"""
build_arms — 통합 데이터셋(스펙 §7) 빌드 + arm 별 SFT export.

서브커맨드:
  unified    generate_traces 출력(jsonl) → 통합 레코드(필터 플래그 + 신호) jsonl.
             KorMedMCQA 에서 질문·선지·gold join, C1(①~④) 판정해 filters.C1 채움.
             C2/C3/C-rand 는 나중 단계(judge_general / counterfactual / 랜덤)에서 채운다.
  merge-c3   채점된 counterfactual 파일(metadata.counterfactual) → CounterfactualFilter 적용
             → filters.C3 = (C1 통과 AND counterfactual 통과) 채움. (run.py 로직 재사용)
  merge-c2   judge_general 이 채운 signals.c2_score → 상위 |C3|개를 filters.C2=True.
             C3 와 크기 매칭(H3 공정 비교). merge-c3 다음에 실행.
  make-crand C3 의 수량 통제군. C1 통과 풀에서 시드 고정 랜덤 |C3|개 → filters."C-rand"=True.
             merge-c3 다음에 실행해야 |C3| 가 정해진다.
  export     통합 레코드 → arm(C0/C1/C2/C3/C-rand) 슬라이스 → SFT messages
             (user=질문+선지, assistant=reasoning_content + "정답: X").

스키마(§7): {id, subject, question, choices, gold, think, final_answer, tokens,
            en_ratio, filters:{C0,C1,C2,C3,C-rand}, signals:{...}}

사용:
  python src/dataset/build_arms.py unified    --input data/train_cot_fewshot.jsonl --output data/unified.jsonl
  python src/dataset/build_arms.py merge-c3   --unified data/unified.jsonl --cf data/cf_judged.json --output data/unified.jsonl
  python src/dataset/build_arms.py make-crand --unified data/unified.jsonl --output data/unified.jsonl --seed 42
  python src/dataset/build_arms.py export     --input data/unified.jsonl --arm C1 --output data/train_C1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.filters import standard as S  # noqa: E402
from utils.data_loader import load_samples  # noqa: E402

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


def _read_unified(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _write_unified(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _cf_id(sample) -> str | None:
    """채점 파일의 CoTSample → unified id (subset_sampleidx).

    실제 파이프라인의 regen 파일은 metadata.sample_idx(per-subset)를 가지며 이게
    generate_traces→unified 의 id 규약과 일치한다. 구 pilot 파일처럼 sample_idx 가
    없으면 top-level id 로 폴백한다((subset,id) 유일 보장 시에만 유효).
    """
    if not sample.subset:
        return None
    sidx = (sample.metadata or {}).get("sample_idx")
    if sidx is None:
        sidx = sample.id
    try:
        return f"{sample.subset}_{int(sidx):04d}"
    except (TypeError, ValueError):
        return None


def cmd_merge_c3(args):
    """채점된 counterfactual 파일에 CounterfactualFilter 를 적용해 filters.C3 채움.

    C3 = C1 통과 AND counterfactual 통과. run.py 의 판정 로직을 그대로 재사용한다.
    임계값(gap/hedge/min_orig)은 기본적으로 pipeline_config.yaml 에서 읽되, --gap/--hedge/
    --min-orig 로 override 할 수 있다(임계값 검증 verify_thresholds.py 가 후보 컷별 arm 을
    만들 때 사용). on_missing 정책은 config.
    채점 파일에 없는 샘플의 C3 는 None(판정 보류) — cf 채점이 아직 안 된 샘플.
    """
    from filters.counterfactual.run import CounterfactualFilter

    flt = CounterfactualFilter(
        gap_threshold=args.gap,
        hedge_gap_threshold=args.hedge,
        min_orig_score=args.min_orig,
    )
    if any(v is not None for v in (args.gap, args.hedge, args.min_orig)):
        print(f"[merge-c3] 임계값 override: gap={flt.gap_threshold} "
              f"hedge={flt.hedge_gap_threshold} min_orig={flt.min_orig_score}")

    samples = load_samples(args.cf)
    cf_seen, no_idx = set(), 0
    for s in samples:
        rid = _cf_id(s)
        if rid is None:
            no_idx += 1
            continue
        cf_seen.add(rid)
    passed = flt.run(samples, verbose=True)
    cf_pass = {rid for s in passed if (rid := _cf_id(s)) is not None}

    rows = _read_unified(args.unified)
    n_true = n_false = n_none = 0
    for r in rows:
        rid = r["id"]
        c1 = bool(r.get("filters", {}).get("C1"))
        if not c1:
            c3 = False                 # C1 탈락 → C3 불가 (C3 ⊆ C1)
        elif rid in cf_pass:
            c3 = True
        elif rid in cf_seen:
            c3 = False                 # cf 채점됨 + 탈락
        else:
            c3 = None                  # cf 채점 안 됨 → 보류
        r.setdefault("filters", {})["C3"] = c3
        n_true += c3 is True
        n_false += c3 is False
        n_none += c3 is None

    _write_unified(rows, args.output)
    print(f"[merge-c3] {len(rows)}건 → {args.output}")
    print(f"  C3: True={n_true}  False={n_false}  None(cf 미채점)={n_none}")
    print(f"  cf 채점 파일: {len(samples)}건 (sample_idx 없음 {no_idx} 제외, "
          f"통과 {len(cf_pass)})")
    if n_none:
        print(f"  ⚠ {n_none}건은 cf 채점 전 — make-crand 전에 전체 채점 필요 "
              f"(|C3| 가 과소집계됨)")


def cmd_merge_c2(args):
    """C2 = C1 + 범용 judge 상위 N개 (N=|C3|, 크기 매칭) → filters.C2 채움.

    judge_general.py 가 채운 signals.c2_score 를 읽어, C1 통과 풀에서 점수 상위
    |C3| 개를 C2=True 로 표시한다. C3 와 동일 크기로 맞춰 H3(같은 크기에서 judge
    필터보다 반사실이 낫다)를 공정 비교한다. merge-c3 다음에 실행해야 |C3| 가 정해진다.
    """
    rows = _read_unified(args.unified)
    n_c3 = sum(1 for r in rows if r.get("filters", {}).get("C3") is True)
    pool = [r for r in rows
            if r.get("filters", {}).get("C1") is True
            and r.get("signals", {}).get("c2_score") is not None]
    n_unscored = sum(1 for r in rows
                     if r.get("filters", {}).get("C1") is True
                     and r.get("signals", {}).get("c2_score") is None)

    if n_c3 == 0:
        print("[merge-c2] ⚠ C3=True 가 0건 — merge-c3 를 먼저 실행하세요(크기 매칭 불가). 중단.")
        return
    if not pool:
        print("[merge-c2] ⚠ c2_score 가 채워진 C1 통과 행이 없음 — judge_general.py 를 먼저 실행하세요. 중단.")
        return
    if n_unscored:
        print(f"[merge-c2] ⚠ C1 통과 중 c2_score 미채점 {n_unscored}건 — Top-N 후보에서 제외됨 "
              f"(judge_general.py 로 전체 채점 권장).")

    # 점수 내림차순 정렬(동점은 id 로 안정 정렬) → 상위 |C3| 선택
    pool.sort(key=lambda r: (-r["signals"]["c2_score"], str(r["id"])))
    k = min(n_c3, len(pool))
    chosen = {r["id"] for r in pool[:k]}
    cutoff = pool[k - 1]["signals"]["c2_score"]

    for r in rows:
        r.setdefault("filters", {})["C2"] = r["id"] in chosen

    _write_unified(rows, args.output)
    print(f"[merge-c2] {len(rows)}건 → {args.output}")
    print(f"  C1 풀(채점됨)={len(pool)}  |C3|={n_c3}  → C2={len(chosen)}건 "
          f"(c2_score 컷오프 ≥ {cutoff})")
    if k < n_c3:
        print(f"  ⚠ 채점 풀({len(pool)}) < |C3|({n_c3}) — C2 가 {k}건으로 과소. 전체 채점 필요.")


def cmd_make_crand(args):
    """C3 수량 통제군: C1 통과 풀에서 시드 고정 랜덤 |C3|개 → filters['C-rand']=True.

    C3 vs C-rand 는 N 이 같고(둘 다 |C3|) 한쪽은 품질 선별·한쪽은 랜덤 — H2 비교용.
    랜덤 풀은 C1 통과 전체(=C3 모집단). C3 와의 겹침은 랜덤이므로 허용한다.
    """
    rows = _read_unified(args.unified)
    n_c3 = sum(1 for r in rows if r.get("filters", {}).get("C3") is True)
    n_none = sum(1 for r in rows if r.get("filters", {}).get("C3") is None)
    pool = [r["id"] for r in rows if r.get("filters", {}).get("C1") is True]

    if n_c3 == 0:
        print("[make-crand] ⚠ C3=True 가 0건 — merge-c3 를 먼저 실행하세요. 중단.")
        return
    if n_none:
        print(f"[make-crand] ⚠ C3=None {n_none}건 존재 — cf 채점 미완 상태. "
              f"|C3|={n_c3} 가 과소집계일 수 있음.")

    k = min(n_c3, len(pool))
    chosen = set(random.Random(args.seed).sample(pool, k))
    for r in rows:
        r.setdefault("filters", {})["C-rand"] = r["id"] in chosen

    _write_unified(rows, args.output)
    print(f"[make-crand] {len(rows)}건 → {args.output}")
    print(f"  C1 풀={len(pool)}  |C3|={n_c3}  → C-rand={len(chosen)}건 (seed={args.seed})")
    overlap = sum(1 for r in rows
                  if r.get("filters", {}).get("C-rand") and r.get("filters", {}).get("C3"))
    print(f"  C3 와 겹침 {overlap}건 (랜덤이므로 정상)")


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

    m = sub.add_parser("merge-c3", help="채점된 counterfactual → filters.C3 채움")
    m.add_argument("--unified", required=True, help="cmd_unified 출력 jsonl")
    m.add_argument("--cf", required=True, help="채점된 counterfactual json (metadata.counterfactual)")
    m.add_argument("--output", required=True)
    m.add_argument("--gap", type=float, default=None, help="gap 임계값 override (생략=config)")
    m.add_argument("--hedge", type=float, default=None, help="hedge_gap 임계값 override (생략=config)")
    m.add_argument("--min-orig", type=float, default=None, help="min_orig_score override (생략=config)")
    m.set_defaults(func=cmd_merge_c3)

    m2 = sub.add_parser("merge-c2", help="범용 judge 상위 |C3|개 → C2 (크기 매칭)")
    m2.add_argument("--unified", required=True, help="judge_general 으로 c2_score 채운 unified jsonl")
    m2.add_argument("--output", required=True)
    m2.set_defaults(func=cmd_merge_c2)

    c = sub.add_parser("make-crand", help="C1 풀에서 랜덤 |C3|개 → C-rand 수량 통제군")
    c.add_argument("--unified", required=True, help="merge-c3 완료된 unified jsonl")
    c.add_argument("--output", required=True)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_make_crand)

    e = sub.add_parser("export", help="통합 → arm 슬라이스 → SFT messages")
    e.add_argument("--input", required=True)
    e.add_argument("--arm", required=True, choices=["C0", "C1", "C2", "C3", "C-rand"])
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
