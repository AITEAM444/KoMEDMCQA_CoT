"""
build_arms — 통합 데이터셋(스펙 §7) 빌드 + arm 별 SFT export.

서브커맨드:
  unified  generate_traces 출력(jsonl) → 통합 레코드(필터 플래그 + 신호) jsonl.
           KorMedMCQA 에서 질문·선지·gold join, C1(①~④) 판정해 filters.C1 채움.
           C2/C3 는 judge_general / counterfactual 결과로 나중에 채운다(여기선 null).
  export   통합 레코드 → arm(C0/C1/C2/C3/C-rand) 슬라이스 → SFT messages
           (user=질문+선지, assistant=reasoning_content + "정답: X").

스키마(§7): {id, subject, question, choices, gold, think, final_answer, tokens,
            en_ratio, filters:{C0,C1,C2,C3}, signals:{...}}

사용:
  python src/dataset/build_arms.py unified --input data/train_cot_fewshot.jsonl --output data/unified.jsonl
  python src/dataset/build_arms.py export  --input data/unified.jsonl --arm C1 --output data/train_C1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.filters import standard as S  # noqa: E402

SUBSETS = ["doctor", "nurse", "pharm", "dentist"]
CHOICE_KEYS = ["A", "B", "C", "D", "E"]
_L = ["A", "B", "C", "D", "E"]

INSTRUCTION = (
    "다음 한국 의료 자격시험 문제를 풀이하세요. "
    "충분히 추론한 뒤, 마지막 줄에 정확히 \"정답: X\" (X는 A~E 중 하나) 형식으로 답하세요."
)
_TRAIL_ANSWER_RE = re.compile(
    r"(?:\n+|\s)*(?:thus,?\s*|so,?\s*|따라서\s*|그러므로\s*)?정답\s*[::]\s*\(?[A-E]\)?\.?\s*$",
    re.IGNORECASE,
)


def _en_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if re.search(r"[A-Za-z]{2,}", t)) / len(toks)


def _q_block(question, choices):
    lines = "\n".join(f"{_L[i]}. {c}" for i, c in enumerate(choices))
    return f"[문제]\n{question}\n\n[선택지]\n{lines}"


def _load_ref(split):
    from datasets import load_dataset
    ref = {}
    for sub in SUBSETS:
        ds = load_dataset("sean0042/KorMedMCQA", sub, split=split)
        for i, row in enumerate(ds):
            ref[(sub, i)] = {
                "question": row["question"],
                "choices": [row[k] for k in CHOICE_KEYS if row.get(k)],
                "gold": int(row["answer"]) - 1,
            }
    return ref


def cmd_unified(args):
    ref = _load_ref(args.split)
    cfg = S._load_cfg()
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    out, miss = [], 0
    for r in rows:
        meta = ref.get((r.get("subset"), r.get("sample_idx")))
        if meta is None:
            miss += 1
            continue
        gold_letter = _L[meta["gold"]]
        think = S._think(r)
        c1_ok, c1_res = S.passes_c1(r, gold_letter, cfg)
        out.append({
            "id": f'{r.get("subset")}_{int(r.get("sample_idx")):04d}',
            "subject": r.get("subset"),
            "question": meta["question"],
            "choices": meta["choices"],
            "gold": meta["gold"],
            "think": think,
            "final_answer": S._to_letter(r.get("predicted")),
            "tokens": len(think.split()),
            "en_ratio": round(_en_ratio(think), 4),
            "filters": {"C0": True, "C1": c1_ok, "C2": None, "C3": None},
            "signals": {"c1": c1_res, "think_lang": r.get("think_lang")},
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n = len(out)
    c1 = sum(1 for o in out if o["filters"]["C1"])
    print(f"[unified] {n}건 저장 → {args.output} (ref 없음 {miss} 제외)")
    print(f"  arm 크기: C0={n}  C1={c1}  (C2/C3 = judge_general/counterfactual 실행 후 채움)")
    print(f"  평균 en_ratio={sum(o['en_ratio'] for o in out)/max(n,1):.2f}  "
          f"평균 tokens={sum(o['tokens'] for o in out)//max(n,1)}")


def cmd_export(args):
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    arm = args.arm
    out, skip = [], 0
    for r in rows:
        if not r.get("filters", {}).get(arm):     # None/False → arm 미포함
            skip += 1
            continue
        think = (r.get("think") or "").strip()
        ans = r.get("final_answer") or _L[r["gold"]]
        if not think or ans is None:
            skip += 1
            continue
        think_clean = _TRAIL_ANSWER_RE.sub("", think).strip()
        out.append({
            "messages": [
                {"role": "user", "content": f"{INSTRUCTION}\n\n{_q_block(r['question'], r['choices'])}"},
                {"role": "assistant", "content": f"{think_clean}\n정답: {ans}"},
            ],
            "meta": {"id": r["id"], "subject": r["subject"],
                     "gold": _L[r["gold"]], "answer_used": ans},
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[export:{arm}] {len(out)}건 저장 → {args.output} (arm 미포함 {skip} 제외)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("unified", help="generate_traces 출력 → 통합 스키마")
    u.add_argument("--input", required=True)
    u.add_argument("--output", required=True)
    u.add_argument("--split", default="train")
    u.set_defaults(func=cmd_unified)

    e = sub.add_parser("export", help="통합 → arm 슬라이스 → SFT messages")
    e.add_argument("--input", required=True)
    e.add_argument("--arm", required=True, choices=["C0", "C1", "C2", "C3", "C-rand"])
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
