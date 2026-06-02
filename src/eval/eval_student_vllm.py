"""
Student / Base 모델 평가 (vLLM 버전) — KorMedMCQA test 정답률.

eval_student.py 의 vLLM 포팅. 채점/파싱/체크포인트/subset 집계 로직은 동일하고,
HF transformers.generate(정적 배칭) → vLLM continuous batching 으로만 교체.
greedy(temperature=0) 평가라 HF 버전과 결과는 사실상 동등.

vLLM 에서는 batch_size 를 직접 정하지 않는다. 프롬프트를 전부 던지면 vLLM 이
--gpu-mem-util 로 잡아둔 KV 캐시 한도 안에서 알아서 최대로 채운다.

★ 동시에 2개 평가를 돌릴 때:
    - 같은 GPU 1장 공유  → 두 프로세스 모두 --gpu-mem-util 0.45 이하 (기본 0.9 면 2번째가 죽음)
    - GPU 2장에 1개씩    → CUDA_VISIBLE_DEVICES 로 분리하고 각각 0.9

사용:
    # base 원래 정답률 (GPU 1장 단독)
    python scripts/eval_student_vllm.py --model Qwen/Qwen3-8B --split test --output eval_base.jsonl
    # LoRA 학습 후 (C1)
    python scripts/eval_student_vllm.py --model Qwen/Qwen3-8B \
        --lora output/qwen3-8b-lora-nofilter --split test --output eval_c1.jsonl
    # full FT 후
    python scripts/eval_student_vllm.py --model output/qwen3-8b-nofilter --split test --output eval_c1.jsonl

    # 같은 GPU 1장에서 2개 동시
    python scripts/eval_student_vllm.py --model A --gpu-mem-util 0.45 --output a.jsonl
    python scripts/eval_student_vllm.py --model B --gpu-mem-util 0.45 --output b.jsonl
    # GPU 2장에 1개씩  (PowerShell:  $env:CUDA_VISIBLE_DEVICES="0";  / "1")
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_student_vllm.py --model A --output a.jsonl
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_student_vllm.py --model B --output b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "generation"))   # generate_traces (구 eval_cot)
_sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root (filters/, utils/)
_sys.path.insert(0, os.path.join(_HERE, "..", "src", "generation"))  # 패키지 구조 대비

from generate_traces import SUBSETS, load_samples, parse_answer

_LETTERS = ["A", "B", "C", "D", "E"]
INSTRUCTION = (
    "다음 한국 의료 자격시험 문제를 풀이하세요. "
    "충분히 추론한 뒤, 마지막 줄에 정확히 \"정답: X\" (X는 A~E 중 하나) 형식으로 답하세요."
)


def q_block(s):
    ch = "\n".join(f"{_LETTERS[i]}. {s[k]}" for i, k in enumerate(_LETTERS) if s.get(k))
    return f"[문제]\n{s['question']}\n\n[선택지]\n{ch}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="base 또는 full-FT 체크포인트 경로")
    ap.add_argument("--lora", default=None, help="LoRA adapter 경로 (있으면 base 위에 로드)")
    ap.add_argument("--split", default="test", choices=["train", "test", "dev"])
    ap.add_argument("--total", type=int, default=-1, help="-1 이면 split 전체")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="프롬프트+생성 최대 길이. KV 캐시 크기를 좌우.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9,
                    help="vLLM 이 잡을 VRAM 비율. 같은 GPU 1장에 2개 띄우면 각각 0.45 이하.")
    ap.add_argument("--max-lora-rank", type=int, default=64,
                    help="LoRA adapter 의 r 보다 크거나 같아야 함.")
    ap.add_argument("--chunk", type=int, default=512,
                    help="이 개수마다 결과를 파일에 flush (크래시 시 체크포인트). "
                         "continuous batching 포화엔 256~512면 충분.")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 thinking 모드 on (기본 off — Student 출력 스타일과 일치)")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 출력 무시하고 처음부터 (기본은 이어서 — 완료분 건너뜀)")
    ap.add_argument("--output", default="eval_results.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_lora=bool(args.lora),
        max_lora_rank=args.max_lora_rank,
    )
    lora_req = LoRARequest("adapter", 1, args.lora) if args.lora else None
    if args.lora:
        print(f"[eval] LoRA adapter 로드: {args.lora}")

    # greedy 평가 (HF do_sample=False 와 동일). stop 은 모델 chat template 의 EOS 사용.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    items = load_samples(args.total, split=args.split)

    def make_prompt(s):
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{q_block(s)}"}]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    # 체크포인트: 기존 출력의 완료분 회수 → (subset, sample_idx) 로 건너뜀 (--overwrite 시 무시)
    done = {}
    if not args.overwrite and os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[(r["subset"], r["sample_idx"])] = r
            except (json.JSONDecodeError, KeyError):
                pass
    pending = [it for it in items if (it["subset"], it["sample_idx"]) not in done]
    print(f"[eval] {args.split} {len(items)}건 | 완료 {len(done)} 건너뜀, 신규 {len(pending)}건 "
          f"(model={args.model}, lora={args.lora})")

    per_subset = defaultdict(lambda: {"total": 0, "correct": 0})
    n_correct = n_unparsed = n_total = 0

    def aggregate(rec):
        nonlocal n_correct, n_unparsed, n_total
        n_total += 1
        n_unparsed += rec.get("predicted") is None
        n_correct += int(rec.get("is_correct", False))
        per_subset[rec["subset"]]["total"] += 1
        per_subset[rec["subset"]]["correct"] += int(rec.get("is_correct", False))

    # 파일 새로 쓰되 완료분(done) 먼저 기록 → 부분파일이 stale 안 되게.
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in done.values():
            aggregate(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()

        # chunk 단위로 vLLM 에 투입 → 각 chunk 결과를 flush (크래시 대비 체크포인트).
        # chunk 안에서는 vLLM 이 continuous batching 으로 알아서 최대 동시 처리.
        for start in range(0, len(pending), args.chunk):
            batch = pending[start:start + args.chunk]
            prompts = [make_prompt(it["sample"]) for it in batch]
            outs = llm.generate(prompts, sp, lora_request=lora_req)

            for it, out in zip(batch, outs):
                s = it["sample"]
                gold = _LETTERS[int(s["answer"]) - 1]
                text = out.outputs[0].text
                pred = parse_answer(text)
                rec = {"subset": it["subset"], "sample_idx": it["sample_idx"],
                       "gold": gold, "predicted": pred, "is_correct": pred == gold,
                       "output": text}
                aggregate(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[eval] +{min(start + args.chunk, len(pending))}/{len(pending)} "
                  f"(누적 {n_total}, 정답 {n_correct})", flush=True)

    n = n_total
    print("\n" + "=" * 56)
    print(f"KorMedMCQA {args.split} 정답률 — {args.model}" + (f" + {args.lora}" if args.lora else ""))
    print("=" * 56)
    print(f"  전체: {n_correct}/{n} = {n_correct / n:.4f}  (파싱실패 {n_unparsed})")
    for sub in SUBSETS:
        d = per_subset[sub]
        if d["total"]:
            print(f"  {sub:8s}: {d['correct']}/{d['total']} = {d['correct'] / d['total']:.4f}")
    print(f"\n저장 → {args.output}")


if __name__ == "__main__":
    main()
