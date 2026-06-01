"""
Student / Base 모델 평가 — KorMedMCQA test 정답률.

base Qwen3-8B 또는 base+LoRA adapter 를 KorMedMCQA test 로 zero-shot 평가한다.
프롬프트는 SFT 학습과 동일(INSTRUCTION + 문제 + 선지). 생성 후 정답 줄 파싱.

비교 기준점:
    base Qwen3-8B(원래) → C1(no-filter) → C2 → C3  모두 같은 방식으로 test 정답률 측정.

주의(보고서 §2): test split 은 *Teacher CoT 생성* 에는 금지지만 *모델 평가* 에는 사용한다.

사용:
    # base 원래 정답률
    python scripts/eval_student.py --model Qwen/Qwen3-8B --split test --output eval_base.jsonl
    # LoRA 학습 후 (C1)
    python scripts/eval_student.py --model Qwen/Qwen3-8B \
        --lora output/qwen3-8b-lora-nofilter --split test --output eval_c1.jsonl
    # full FT 후
    python scripts/eval_student.py --model output/qwen3-8b-nofilter --split test --output eval_c1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "generation"))   # generate_traces (구 eval_cot)
_sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root (filters/, utils/)

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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 thinking 모드 on (기본 off — Student 출력 스타일과 일치)")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 출력 무시하고 처음부터 (기본은 이어서 — 완료분 건너뜀)")
    ap.add_argument("--output", default="eval_results.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        print(f"[eval] LoRA adapter 로드: {args.lora}")
    model.eval()

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

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [make_prompt(it["sample"]) for it in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=4096).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, pad_token_id=tok.pad_token_id,
                )
            gen = gen[:, enc.input_ids.shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)

            for it, text in zip(batch, texts):
                s = it["sample"]
                gold = _LETTERS[int(s["answer"]) - 1]
                pred = parse_answer(text)
                rec = {"subset": it["subset"], "sample_idx": it["sample_idx"],
                       "gold": gold, "predicted": pred, "is_correct": pred == gold,
                       "output": text}
                aggregate(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"[eval] +{min(start + args.batch_size, len(pending))}/{len(pending)} "
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
