"""
Student SFT 데이터 변환 — fewshot 생성 출력(reasoning_content)을 학습 포맷으로.

입력: eval_cot.py --prompt-mode fewshot 의 출력 jsonl (train_cot_fewshot.jsonl).
      여기엔 reasoning_content(학습 타겟)는 있으나 question 이 100자로 잘려있고 선지가
      없으므로, KorMedMCQA 에서 (subset, sample_idx) 로 질문+선지를 다시 join 한다.

출력: chat messages jsonl.
  user      = INSTRUCTION + [문제] + [선택지]   (zero-shot, fewshot 미포함)
  assistant = reasoning_content + "\\n정답: X"   (X = Teacher 가 실제 도달한 답=predicted)

비교군:
  - no-filter (baseline): reasoning_content 가 있는 모든 샘플 (정답 틀린 것도 포함).
    → 이후 필터(C2/C3)는 이 모집단에서 부분집합을 골라 같은 방식으로 학습.

사용:
    python scripts/prepare_sft_data.py \
        --input data-raw/train_cot_fewshot.jsonl \
        --output data/sft/train_nofilter.jsonl \
        --answer predicted
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# reasoning_content 끝의 "정답: X"/"Thus, 정답: X" 류 줄 제거용 (canonical 정답 줄과 중복 방지)
_TRAIL_ANSWER_RE = re.compile(
    r"(?:\n+|\s)*(?:thus,?\s*|so,?\s*|따라서\s*|그러므로\s*)?정답\s*[::]\s*\(?[A-E]\)?\.?\s*$",
    re.IGNORECASE,
)

SUBSETS = ["doctor", "nurse", "pharm", "dentist"]
CHOICE_KEYS = ["A", "B", "C", "D", "E"]
_LETTERS = ["A", "B", "C", "D", "E"]

INSTRUCTION = (
    "다음 한국 의료 자격시험 문제를 풀이하세요. "
    "충분히 추론한 뒤, 마지막 줄에 정확히 \"정답: X\" (X는 A~E 중 하나) 형식으로 답하세요."
)


def _q_block(question, choices):
    lines = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"[문제]\n{question}\n\n[선택지]\n{lines}"


def _to_letter(v):
    if isinstance(v, str):
        v = v.strip().upper()
        return v if v in _LETTERS else None
    if isinstance(v, int):
        return _LETTERS[v - 1] if 1 <= v <= 5 else (_LETTERS[v] if 0 <= v < 5 else None)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--answer", choices=["predicted", "gold"], default="predicted",
                   help="completion 의 정답 줄 출처. no-filter 베이스라인은 predicted(추론과 일치) 권장.")
    p.add_argument("--min-reasoning-len", type=int, default=20,
                   help="reasoning_content 가 이보다 짧으면 제외 (빈/깨진 샘플).")
    p.add_argument("--correct-only", action="store_true",
                   help="Teacher 최종답 == gold 인 샘플만 사용 (answer correctness 필터, "
                        "distillation 표준 baseline). 미지정 시 no-filter(오답 포함).")
    args = p.parse_args()

    from datasets import load_dataset

    # KorMedMCQA 에서 (subset, idx) → 질문+선지+gold
    ref = {}
    for sub in SUBSETS:
        ds = load_dataset("sean0042/KorMedMCQA", sub, split=args.split)
        for i, row in enumerate(ds):
            choices = [row[k] for k in CHOICE_KEYS if row.get(k)]
            ref[(sub, i)] = {"question": row["question"], "choices": choices,
                             "gold": _LETTERS[int(row["answer"]) - 1]}
    print(f"[ref] KorMedMCQA {args.split}: {len(ref)} 행")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    out, skip_empty, skip_noref, skip_noans, skip_wrong = [], 0, 0, 0, 0
    for r in rows:
        rc = (r.get("reasoning_content") or "").strip()
        if len(rc) < args.min_reasoning_len:
            skip_empty += 1
            continue
        key = (r.get("subset"), r.get("sample_idx"))
        meta = ref.get(key)
        if meta is None:
            skip_noref += 1
            continue
        ans = _to_letter(r.get("predicted")) if args.answer == "predicted" else None
        if ans is None:
            ans = meta["gold"]  # predicted 없으면 gold 로 폴백
        if ans is None:
            skip_noans += 1
            continue
        if args.correct_only and ans != meta["gold"]:
            skip_wrong += 1
            continue
        user = f"{INSTRUCTION}\n\n{_q_block(meta['question'], meta['choices'])}"
        # reasoning 끝에 모델이 이미 적은 "정답: X" 류 줄 제거 후, canonical 정답 줄 1개만 부착.
        rc_clean = _TRAIL_ANSWER_RE.sub("", rc).strip()
        assistant = f"{rc_clean}\n정답: {ans}"
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"subset": r.get("subset"), "sample_idx": r.get("sample_idx"),
                     "gold": meta["gold"], "answer_used": ans,
                     "is_correct": ans == meta["gold"]},
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n_correct = sum(1 for o in out if o["meta"]["is_correct"])
    mode = "correct-only(표준 baseline)" if args.correct_only else "no-filter"
    print(f"[DONE] {len(out)}건 저장 → {args.output}  [{mode}]")
    print(f"  제외: 빈 reasoning {skip_empty} | ref없음 {skip_noref} | 정답없음 {skip_noans} | 오답(correct-only) {skip_wrong}")
    print(f"  정답 일치(answer_used==gold): {n_correct}/{len(out)} "
          f"({n_correct/len(out)*100:.1f}%) — no-filter 라 오답도 포함됨")


if __name__ == "__main__":
    main()
