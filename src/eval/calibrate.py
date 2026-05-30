"""반사실 필터 임계값 보정 — dev 스윕 표 (연구계획 §3 / §6).

연구계획상 gap/hedge/min_orig 임계값은 **dev 로 보정**해야 한다(test 로 맞추면 부정행위).
이 스크립트는 학습 없이, dev cf 채점 결과(precompute 출력)에서 임계값 조합마다
**reject율 + reject 사유 분해 + 신호 분포**를 표로 뽑아준다. 최종 임계값은 사람이 표를
보고 골라 configs/pipeline_config.yaml 에 적는다(다운스트림 ACC 보정은 학습이 필요하므로
여기선 제외).

판정 로직은 filters/counterfactual/run.py 와 동일하게 재현한다:
  1. orig_score < min_orig            → reject (독립 안전망, OR)
  2. hedge None  → gap < gap_thr      → reject
  3. hedge 설정  → (gap < gap_thr) AND (hedge_gap >= hedge_thr) → reject
                  gap 은 작아도 hedge_gap < thr 이면 rescue(keep)

사용:
    # dev cf 채점 파일(precompute 출력)로 스윕
    python src/eval/calibrate.py --cf data/dev_cf_judged.json \
        --gaps 1.0 1.25 1.5 1.75 2.0 2.5 --hedges none 0.0 0.5 1.0 --min-origs none 2.0 \
        --output results/calibrate_dev.md
    # C1 통과분으로만 한정(권장): dev unified 와 교집합
    python src/eval/calibrate.py --cf data/dev_cf_judged.json --unified data/dev_unified.jsonl
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from filters.counterfactual.run import _compute_hedge_signals
from utils.data_loader import load_samples


def _parse_floats_or_none(vals: list[str]) -> list[float | None]:
    out = []
    for v in vals:
        out.append(None if v.lower() in ("none", "null", "off") else float(v))
    return out


def _cf_id(sample) -> str | None:
    """cf 채점 샘플 → unified id (subset_sampleidx). build_arms._cf_id 와 동일 규약."""
    if not sample.subset:
        return None
    sidx = (sample.metadata or {}).get("sample_idx")
    if sidx is None:
        sidx = sample.id
    try:
        return f"{sample.subset}_{int(sidx):04d}"
    except (TypeError, ValueError):
        return None


def _c1_ids(unified_path: Path) -> set[str]:
    import json
    ids = set()
    for line in open(unified_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("filters", {}).get("C1") is True:
            ids.add(r["id"])
    return ids


def _quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if not xs:
        return {q: None for q in qs}
    s = sorted(xs)
    return {q: round(s[min(int(q * (len(s) - 1) + 0.5), len(s) - 1)], 3) for q in qs}


def collect_stats(samples) -> tuple[list[dict], int]:
    """각 샘플의 (orig, cf_max, gap, hedge_gap) 1회 계산. 점수 없는 건 missing."""
    rows, n_missing = [], 0
    for s in samples:
        meta = (s.metadata or {}).get("counterfactual")
        if not meta:
            n_missing += 1
            continue
        orig = meta.get("orig_score")
        cfs = meta.get("counterfactuals", [])
        cf_scores = [c["cf_score"] for c in cfs if c.get("cf_score") is not None]
        if orig is None or not cf_scores:
            n_missing += 1
            continue
        cf_max = max(cf_scores)
        hedge_gap = _compute_hedge_signals(s.cot, cfs).get("hedge_gap")
        rows.append({"orig": orig, "cf_max": cf_max, "gap": orig - cf_max, "hedge_gap": hedge_gap})
    return rows, n_missing


def decide(r: dict, g: float, h: float | None, m: float | None) -> str:
    """run.py 판정 재현 → 'rej_minorig' | 'rej_gap' | 'rescued' | 'keep'."""
    if m is not None and r["orig"] < m:
        return "rej_minorig"
    gap_trig = r["gap"] < g
    if h is None:
        return "rej_gap" if gap_trig else "keep"
    hedge_trig = r["hedge_gap"] is not None and r["hedge_gap"] >= h
    if gap_trig and hedge_trig:
        return "rej_gap"
    if gap_trig and not hedge_trig:
        return "rescued"   # gap 은 탈락 신호지만 hedge 가 rescue
    return "keep"


def main():
    p = argparse.ArgumentParser(description="반사실 임계값 dev 스윕 표 (학습 없음)")
    p.add_argument("--cf", required=True, help="dev cf 채점 파일(precompute 출력, metadata.counterfactual)")
    p.add_argument("--unified", default=None, help="(선택) dev unified jsonl — C1 통과분으로만 한정")
    p.add_argument("--gaps", nargs="+", default=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5"])
    p.add_argument("--hedges", nargs="+", default=["none", "0.0", "0.5", "1.0"],
                   help="'none' = gap 단독")
    p.add_argument("--min-origs", nargs="+", default=["none", "2.0"])
    p.add_argument("--output", default=None, help="(선택) 마크다운 표 저장 경로")
    args = p.parse_args()

    samples = load_samples(args.cf)
    if args.unified:
        keep_ids = _c1_ids(Path(args.unified))
        before = len(samples)
        samples = [s for s in samples if _cf_id(s) in keep_ids]
        print(f"[calibrate] C1 통과분 한정: {before} → {len(samples)} (dev unified 기준)")

    rows, n_missing = collect_stats(samples)
    n = len(rows)
    if n == 0:
        raise SystemExit("[calibrate] 점수가 있는 샘플이 없습니다. precompute 채점 출력을 입력하세요.")

    gaps = [float(x) for x in args.gaps]
    hedges = _parse_floats_or_none(args.hedges)
    min_origs = _parse_floats_or_none(args.min_origs)

    # ── 신호 분포 (1회) ──────────────────────────────────────────────────────────
    gq = _quantiles([r["gap"] for r in rows])
    oq = _quantiles([r["orig"] for r in rows])
    hvals = [r["hedge_gap"] for r in rows if r["hedge_gap"] is not None]
    hq = _quantiles(hvals)
    dist_lines = [
        "## 신호 분포 (dev, 점수 있는 샘플)",
        f"- n_scored={n}, n_missing={n_missing}",
        f"- gap        min/Q1/med/Q3/max = {gq[0.0]}/{gq[0.25]}/{gq[0.5]}/{gq[0.75]}/{gq[1.0]}  (평균 {statistics.mean(r['gap'] for r in rows):.3f})",
        f"- orig_score min/Q1/med/Q3/max = {oq[0.0]}/{oq[0.25]}/{oq[0.5]}/{oq[0.75]}/{oq[1.0]}",
        f"- hedge_gap  min/Q1/med/Q3/max = {hq[0.0]}/{hq[0.25]}/{hq[0.5]}/{hq[0.75]}/{hq[1.0]}  (n={len(hvals)})",
    ]

    # ── 스윕 표 ──────────────────────────────────────────────────────────────────
    header = ["gap", "hedge", "min_orig", "kept", "rej", "rej%",
              "rej_minorig", "rej_gap", "rescued"]
    table = []
    for m in min_origs:
        for h in hedges:
            for g in gaps:
                cnt = {"keep": 0, "rescued": 0, "rej_minorig": 0, "rej_gap": 0}
                for r in rows:
                    cnt[decide(r, g, h, m)] += 1
                kept = cnt["keep"] + cnt["rescued"]
                rej = cnt["rej_minorig"] + cnt["rej_gap"]
                table.append([
                    f"{g:g}", ("—" if h is None else f"{h:g}"), ("—" if m is None else f"{m:g}"),
                    kept, rej, f"{rej / n * 100:.1f}%",
                    cnt["rej_minorig"], cnt["rej_gap"], cnt["rescued"],
                ])

    # 콘솔 출력
    print("\n".join(dist_lines))
    print("\n## 임계값 스윕 (reject율 = rej / n_scored)")
    widths = [max(len(str(row[i])) for row in ([header] + table)) for i in range(len(header))]
    def fmt(row): return "  ".join(str(c).rjust(widths[i]) for i, c in enumerate(row))
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))
    print("\n현재 config 기본값: gap=1.5, hedge=0.5, min_orig=2.0 → 위 표에서 해당 행 확인 후 선택.")

    if args.output:
        md = list(dist_lines)
        md += ["", "## 임계값 스윕 (reject율 = rej / n_scored)", "",
               "| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        md += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(md), encoding="utf-8")
        print(f"\n표 저장 → {args.output}")


if __name__ == "__main__":
    main()
