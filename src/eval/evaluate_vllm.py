"""
Student / Base 모델 평가 (vLLM 백엔드) — KorMedMCQA test 정답률.

evaluate.py(HF transformers) 의 vLLM 판. 프롬프트·greedy·출력 스키마·resume 를
모두 동일하게 맞춰 기존 결과(eval_base/C0/C1)와 직접 비교 가능하게 했다.
continuous batching 으로 보통 5~10배 빠르다.

KoMEDMCQA_CoT repo 의 src/eval/ 에 두고 실행하는 것을 가정(generate_traces import).

사용:
    # base
    python src/eval/evaluate_vllm.py --model Qwen/Qwen3-8B --split test \
        --output results/eval_base_vllm.jsonl
    # LoRA (C1)
    python src/eval/evaluate_vllm.py --model Qwen/Qwen3-8B \
        --lora output/qwen3-8b-c1-s42 --split test --output results/eval_C1_s42.jsonl

주의(보고서 §2): test split 은 *Teacher CoT 생성* 에는 금지지만 *모델 평가* 에는 사용한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys as _sys
from collections import defaultdict

from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "generation"))   # generate_traces
_sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root

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
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=6144,
                    help="프롬프트+생성 합(생성 4096 + 프롬프트 여유 2048)")
    ap.add_argument("--max-lora-rank", type=int, default=32, help="train.yaml lora_rank 와 일치")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 thinking 모드 on (기본 off — evaluate.py 와 동일)")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 출력 무시하고 처음부터 (기본은 이어서 — 완료분 건너뜀)")
    ap.add_argument("--output", default="eval_results_vllm.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_lora=bool(args.lora),
        max_lora_rank=args.max_lora_rank,
    )
    lora_req = LoRARequest("adapter", 1, args.lora) if args.lora else None
    if args.lora:
        print(f"[eval] LoRA adapter 로드: {args.lora}")

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    items = load_samples(args.total, split=args.split)

    def make_prompt(s):
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{q_block(s)}"}]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    # 체크포인트: 기존 출력의 완료분 회수 → (subset, sample_idx) 로 건너뜀
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
    print(f"[eval-vllm] {args.split} {len(items)}건 | 완료 {len(done)} 건너뜀, 신규 {len(pending)}건 "
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

    with open(args.output, "w", encoding="utf-8") as fout:
        for r in done.values():
            aggregate(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()

        if pending:
            prompts = [make_prompt(it["sample"]) for it in pending]
            # vLLM 이 내부적으로 continuous batching — 한 번에 넘긴다.
            outputs = llm.generate(prompts, sampling, lora_request=lora_req)
            for it, out in zip(pending, outputs):
                s = it["sample"]
                text = out.outputs[0].text
                gold = _LETTERS[int(s["answer"]) - 1]
                pred = parse_answer(text)
                rec = {"subset": it["subset"], "sample_idx": it["sample_idx"],
                       "gold": gold, "predicted": pred, "is_correct": pred == gold,
                       "output": text}
                aggregate(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

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
