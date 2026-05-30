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
import math
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


# ── 비지도 자동 컷 (Otsu + 2-성분 GMM) ──────────────────────────────────────────
# 주의: 이건 *gap 분포 자체*를 두 덩어리로 가르는 경계이지, "필터가 ACC를 올리는" 경계가
# 아니다(feature 분포 분리점 ≠ task 결정경계). 따라서 '데이터가 정했으니 옳다'가 아니라
# "작은 dev 라벨에 과적합하지 않으려는 선택"으로 포지셔닝하고, verify_thresholds.py 로
# 후보 컷의 dev ACC 수렴을 *확인*해야 한다. 분리도가 약할수록 그 ACC 확인이 더 중요하다.

def otsu_threshold(values: list[float]) -> tuple[float | None, float]:
    """1D Otsu — 클래스간 분산 최대화 컷 + separability η²(between/total, 0~1)."""
    n = len(values)
    if n < 2:
        return None, 0.0
    mean = statistics.mean(values)
    total_var = sum((x - mean) ** 2 for x in values) / n
    if total_var == 0:
        return None, 0.0
    uniq = sorted(set(values))
    best_t, best_btw = None, -1.0
    for i in range(len(uniq) - 1):
        t = (uniq[i] + uniq[i + 1]) / 2
        below = [x for x in values if x < t]
        above = [x for x in values if x >= t]
        w0, w1 = len(below) / n, len(above) / n
        btw = w0 * w1 * (statistics.mean(below) - statistics.mean(above)) ** 2
        if btw > best_btw:
            best_btw, best_t = btw, t
    return best_t, (best_btw / total_var)


def _norm_pdf(x: float, mu: float, var: float) -> float:
    return math.exp(-(x - mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)


def gmm2_em(values: list[float], iters: int = 300) -> dict:
    """2-성분 1D GMM (EM, 무의존). 성분 평균/분산/가중치 + 교차컷 + BIC(1 vs 2)."""
    n = len(values)
    xs = sorted(values)
    m = statistics.mean(values)
    v = max(sum((x - m) ** 2 for x in values) / n, 1e-6)
    mu = [xs[n // 4], xs[3 * n // 4]]
    if mu[0] == mu[1]:
        mu = [m - math.sqrt(v), m + math.sqrt(v)]
    var = [v, v]
    w = [0.5, 0.5]
    for _ in range(iters):
        resp = []
        for x in values:
            p = [w[k] * _norm_pdf(x, mu[k], var[k]) for k in (0, 1)]
            s = sum(p) or 1e-12
            resp.append((p[0] / s, p[1] / s))
        for k in (0, 1):
            rk = sum(r[k] for r in resp) or 1e-12
            w[k] = rk / n
            mu[k] = sum(r[k] * x for r, x in zip(resp, values)) / rk
            var[k] = max(sum(r[k] * (x - mu[k]) ** 2 for r, x in zip(resp, values)) / rk, 0.05)
    # 성분을 평균 오름차순으로 정렬
    order = sorted((0, 1), key=lambda k: mu[k])
    mu = [mu[k] for k in order]; var = [var[k] for k in order]; w = [w[k] for k in order]

    def loglik2():
        return sum(math.log(max(sum(w[k] * _norm_pdf(x, mu[k], var[k]) for k in (0, 1)), 1e-300))
                   for x in values)

    def loglik1():
        return sum(math.log(max(_norm_pdf(x, m, v), 1e-300)) for x in values)

    bic2 = -2 * loglik2() + 5 * math.log(n)   # 2μ+2σ²+1weight
    bic1 = -2 * loglik1() + 2 * math.log(n)   # μ+σ²
    # 교차컷: 두 성분 사이에서 책임도가 갈리는 지점
    cross = None
    lo, hi = mu[0], mu[1]
    if hi > lo:
        prev = None
        steps = 400
        for i in range(steps + 1):
            t = lo + (hi - lo) * i / steps
            d = w[0] * _norm_pdf(t, mu[0], var[0]) - w[1] * _norm_pdf(t, mu[1], var[1])
            if prev is not None and (d <= 0 < prev or d >= 0 > prev):
                cross = round(t, 3)
                break
            prev = d
    # Cohen's d 류 분리도
    pooled = math.sqrt((var[0] + var[1]) / 2)
    sep_d = abs(mu[1] - mu[0]) / pooled if pooled > 0 else 0.0
    return {"mu": [round(x, 3) for x in mu], "var": [round(x, 3) for x in var],
            "w": [round(x, 3) for x in w], "crossover": cross,
            "bic1": round(bic1, 1), "bic2": round(bic2, 1),
            "bic_favors_2": bic2 < bic1, "sep_d": round(sep_d, 3)}


def auto_cut(gaps: list[float]) -> dict:
    """비지도 컷 산출 + 이봉성 판정 + 검증강도 권고."""
    otsu_t, eta2 = otsu_threshold(gaps)
    g = gmm2_em(gaps)
    # 이봉 판정: BIC 가 2-성분 선호 + 성분 분리 충분 + 가중치 양쪽 의미있음
    bimodal = (g["bic_favors_2"] and g["sep_d"] >= 1.0
               and min(g["w"]) >= 0.10 and g["crossover"] is not None)
    if bimodal:
        cut, basis = g["crossover"], "GMM crossover (이봉)"
    else:
        cut, basis = 1.0, "의미컷 gap≥1 폴백 (이봉 약함 — 분포가 안 갈림)"
    # 분리도 → ACC 검증강도 (단서 ③: 분리도가 2 vs 3 을 자동 조절)
    if g["sep_d"] >= 2.0:
        intensity = "약하게(seed 1~2): 분리 강함→골짜기 넓고 평평→컷 위치 ACC 둔감(사실상 Path2 충분)"
    elif g["sep_d"] >= 1.0:
        intensity = "보통(seed 2~3): 후보 컷들 dev ACC 확인 권장"
    else:
        intensity = "강하게(seed 3+) 또는 의미컷 폴백: 분리 약함→컷 민감→ACC 확인이 가장 값짐"
    # 검증 후보(중복 제거): 비지도 컷 + otsu + 의미컷(1.0) + 현재 config(1.5)
    cands = sorted({round(c, 2) for c in [cut, otsu_t, 1.0, 1.5] if c is not None})
    return {"otsu": otsu_t, "otsu_eta2": round(eta2, 3), "gmm": g,
            "bimodal": bimodal, "cut": cut, "basis": basis,
            "verify_intensity": intensity, "candidates": cands}


def print_auto(a: dict, n: int) -> list[str]:
    g = a["gmm"]
    L = [
        "## 비지도 자동 컷 (gap 분포) — '데이터가 정한 컷', 단 *분포 분리*이지 ACC 최적이 아님",
        f"- Otsu 컷={a['otsu']}  (separability η²={a['otsu_eta2']})",
        f"- GMM 성분 평균={g['mu']}  분산={g['var']}  가중치={g['w']}  교차컷={g['crossover']}",
        f"- 분리도 Cohen's d={g['sep_d']}  BIC(1성분)={g['bic1']} vs BIC(2성분)={g['bic2']} "
        f"→ 2성분 {'선호' if g['bic_favors_2'] else '비선호'}",
        f"- 이봉 판정: {'YES' if a['bimodal'] else 'NO (단봉/약분리)'}",
        f"- **권고 컷 = {a['cut']}**  ({a['basis']})",
        f"- ACC 검증강도(분리도 기반): {a['verify_intensity']}",
        f"- verify_thresholds.py 후보: {a['candidates']}",
        "",
        "> 포지셔닝: 비지도 컷은 '옳아서'가 아니라 *dev 라벨 과적합 회피*용. 이걸 **결정**으로 두고",
        ">  verify_thresholds.py 로 후보 컷의 dev ACC 가 평평한지/일치하는지 *확인*하라(튜닝 아님).",
    ]
    return L


def main():
    p = argparse.ArgumentParser(description="반사실 임계값 dev 스윕 표 + 비지도 자동 컷 (학습 없음)")
    p.add_argument("--cf", required=True, help="dev cf 채점 파일(precompute 출력, metadata.counterfactual)")
    p.add_argument("--unified", default=None, help="(선택) dev unified jsonl — C1 통과분으로만 한정")
    p.add_argument("--gaps", nargs="+", default=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5"])
    p.add_argument("--hedges", nargs="+", default=["none", "0.0", "0.5", "1.0"],
                   help="'none' = gap 단독")
    p.add_argument("--min-origs", nargs="+", default=["none", "2.0"])
    p.add_argument("--auto", action="store_true",
                   help="gap 분포에 Otsu+GMM 비지도 컷 + 이봉성 진단 + 검증강도 권고 출력")
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

    # 비지도 자동 컷 (선택)
    auto_lines = []
    if args.auto:
        auto = auto_cut([r["gap"] for r in rows])
        auto_lines = print_auto(auto, n)

    # 콘솔 출력
    print("\n".join(dist_lines))
    if auto_lines:
        print("\n" + "\n".join(auto_lines))
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
        if auto_lines:
            md += [""] + auto_lines
        md += ["", "## 임계값 스윕 (reject율 = rej / n_scored)", "",
               "| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        md += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(md), encoding="utf-8")
        print(f"\n표 저장 → {args.output}")


if __name__ == "__main__":
    main()
