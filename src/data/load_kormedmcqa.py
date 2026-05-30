"""
정규화 train CoT 파일에 KorMedMCQA 선지(choices)를 붙인다.

`train_cot_normalized.json` 은 deepseek-r1 의 원본 CoT(reasoning_content+final_content)와
question stem, correct(letter), subset, sample_idx 만 갖고 있고 **선지(choices)가 없다.**
Counterfactual(F6) 재생성은 "특정 오답 선지를 정당화"하는 작업이라 선지가 필요하므로,
HuggingFace `sean0042/KorMedMCQA` train split 에서 (subset, sample_idx) 로 선지를 join 한다.

매핑:
  정규화 subset(영문)  →  KorMedMCQA config(국문)
  doctor→의사 / nurse→간호사 / pharm→약사 / dentist→치과의사
  sample_idx == KorMedMCQA train 의 행 인덱스 i (subset 별 0..N-1, 연속).

출력은 precompute 가 바로 읽는 CoTSample 포맷(id/subset/question/choices/answer/cot...).

사용:
    python scripts/join_choices.py \
        --input  data/regularized/train_cot_normalized.json \
        --output data/regularized/train_cot_with_choices.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# KorMedMCQA config 이름은 영어이며 정규화 파일의 subset 명과 동일 (dentist/doctor/nurse/pharm)
SUBSETS = ["doctor", "nurse", "pharm", "dentist"]
CHOICE_KEYS = ["A", "B", "C", "D", "E"]
_LETTERS = ["A", "B", "C", "D", "E"]


def _letter_to_idx(v):
    """correct/answer 를 0-indexed int 로. 'A'~'E' 또는 1-indexed int 또는 0-indexed int 허용."""
    if isinstance(v, str):
        v = v.strip().upper()
        if v in _LETTERS:
            return _LETTERS.index(v)
        if v.isdigit():
            v = int(v)
    if isinstance(v, int):
        return v - 1 if v >= 1 and v <= 5 else v   # 1-indexed 추정 시 보정
    raise ValueError(f"answer 해석 불가: {v!r}")


def _build_cot(row: dict) -> str:
    """원본 CoT 재구성 — <think>{reasoning}</think>\\n{final}. (regen/judge 가 <think>만 추출)"""
    reasoning = (row.get("reasoning_content") or row.get("reasoning") or "").strip()
    final = (row.get("final_content") or "").strip()
    if reasoning and "<think>" not in reasoning.lower():
        return f"<think>{reasoning}</think>\n{final}".strip()
    # 이미 <think> 포함하거나 reasoning 없음
    return (reasoning + ("\n" + final if final else "")).strip() or final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/regularized/train_cot_normalized.json"))
    p.add_argument("--output", type=Path, default=Path("data/regularized/train_cot_with_choices.json"))
    p.add_argument("--split", default="train")
    args = p.parse_args()

    from datasets import load_dataset

    # 1) KorMedMCQA 선지 로드 → {(subset, idx): {choices, answer_idx}}
    table: dict[tuple[str, int], dict] = {}
    for sub in SUBSETS:
        print(f"[LOAD] {sub} {args.split}")
        ds = load_dataset("sean0042/KorMedMCQA", sub, split=args.split)
        for i, row in enumerate(ds):
            choices = [row[k] for k in CHOICE_KEYS if row.get(k)]
            table[(sub, i)] = {"choices": choices, "answer_idx": int(row["answer"]) - 1}
    print(f"[LOAD] KorMedMCQA {args.split}: {len(table)} 행")

    # 2) 정규화 파일에 join
    rows = json.load(open(args.input, encoding="utf-8"))
    out, miss, mismatch = [], 0, 0
    for r in rows:
        sub = r["subset"]; idx = r.get("sample_idx")
        key = (sub, idx)
        ref = table.get(key)
        if ref is None:
            miss += 1
            continue
        ans_idx = _letter_to_idx(r.get("correct", r.get("answer")))
        if ans_idx != ref["answer_idx"]:
            mismatch += 1   # 정답 letter 불일치 — join 키 어긋남 경고용
        out.append({
            "id": str(r.get("global_idx", f"{sub}_{idx}")),
            "subset": sub,
            "question": r["question"],
            "choices": ref["choices"],
            "answer": ans_idx,
            "teacher_model": r.get("model", "deepseek-r1"),
            "cot": _build_cot(r),
            "predicted_answer": _letter_to_idx(r["predicted"]) if r.get("predicted") else None,
            "filter_history": [],
            "judge_score": None,
            "metadata": {"subset": sub, "sample_idx": idx, "split": args.split},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {len(out)}건 저장 → {args.output}")
    print(f"  선지 매칭 실패(miss): {miss}건 | 정답 letter 불일치(mismatch): {mismatch}건")
    if mismatch > len(out) * 0.05:
        print("  ⚠️ mismatch 5% 초과 — join 키(subset/sample_idx) 정합성 의심. 확인 필요.")
    n_ch = sum(1 for s in out if s["choices"])
    print(f"  choices 보유: {n_ch}/{len(out)}")


if __name__ == "__main__":
    main()
