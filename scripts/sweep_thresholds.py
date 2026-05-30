"""
임계값 스윕 — (Rejection Rate, Cliff's δ) 곡선 생성 (보고서 §6).

각 필터를 여러 임계값에서 돌려 (Rejection Rate, δ) 점을 찍어 곡선 비교.
좌상단(적게 버리고 효과 큼)에 가까울수록 우월.
결과 JSON에 임계값별 rejected_ids 포함 → Step 2 조합 시 Jaccard 재산출에 재사용.

사용법:
    # KPPL low_threshold 스윕 (high_threshold 현행 유지)
    python scripts/sweep_thresholds.py \\
        --filter kppl --param low_threshold \\
        --values 2.5 3.0 3.5 4.0 4.5 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/sweep/

    # KPPL high_threshold 스윕 (low_threshold 현행 유지)
    python scripts/sweep_thresholds.py \\
        --filter kppl --param high_threshold \\
        --values 5.5 6.0 6.5 7.0 7.5 8.0 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/sweep/

    # Answer Consistency entailment_threshold 스윕
    python scripts/sweep_thresholds.py \\
        --filter answer_consistency --param entailment_threshold \\
        --values 0.5 0.6 0.65 0.7 0.8 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/sweep/

    # Step Coverage coverage_threshold 스윕
    python scripts/sweep_thresholds.py \\
        --filter step_coverage --param coverage_threshold \\
        --values 0.2 0.4 0.6 0.8 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/sweep/

    # Step Coverage min_tokens.step3 스윕 (점 표기: dict 하위 키)
    python scripts/sweep_thresholds.py \\
        --filter step_coverage --param min_tokens.step3 \\
        --values 15 20 25 30 35 40 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/sweep/
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_samples
from evaluation.ablation import (
    measure_independent,
    compute_cliffs_delta,
    run_bootstrap_delta_ci,
)

FILTER_REGISTRY = {
    "kppl":               ("filters.kppl.run",               "KPPLFilter"),
    "step_coverage":      ("filters.step_coverage.run",      "StepCoverageFilter"),
    "answer_consistency": ("filters.answer_consistency.run", "AnswerConsistencyFilter"),
}


def _load_filter(filter_name: str):
    module_path, class_name = FILTER_REGISTRY[filter_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def main():
    parser = argparse.ArgumentParser(description="임계값 스윕 — (Rej, δ) 곡선")
    parser.add_argument("--filter",   required=True, choices=list(FILTER_REGISTRY))
    parser.add_argument("--param",    required=True,
                        help="오버라이드할 필터 속성명 (kppl: low_threshold/high_threshold, answer_consistency: entailment_threshold)")
    parser.add_argument("--values",   nargs="+", type=float, required=True)
    parser.add_argument("--input",    type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/ablation/sweep"))
    parser.add_argument("--bootstrap_n", type=int, default=10_000)
    args = parser.parse_args()

    baseline = load_samples(args.input)
    print(f"baseline: {len(baseline)}개 샘플 로드")
    print(f"필터={args.filter}  파라미터={args.param}  값={args.values}\n")

    f = _load_filter(args.filter)

    rows = []
    print(f"{'─'*74}")
    print(f"{'Threshold':>12} {'N_out':>6} {'Rej%':>7} {'δ':>8} {'CI_lo':>7} {'CI_hi':>7} {'Disc':>6}")
    print(f"{'─'*74}")

    for val in args.values:
        # "a.b" 형태이면 f.a[b] = val (예: min_tokens.step3), 아니면 setattr
        if '.' in args.param:
            attr, key = args.param.split('.', 1)
            getattr(f, attr)[key] = int(val) if val == int(val) else val
        else:
            setattr(f, args.param, val)
        result = measure_independent(baseline, f)

        pass_scores = [
            s.judge_score.minimum for s in baseline
            if s.id not in result.rejected_ids and s.judge_score is not None
        ]
        reject_scores = [
            s.judge_score.minimum for s in baseline
            if s.id in result.rejected_ids and s.judge_score is not None
        ]

        delta = ci_lo = ci_hi = None
        is_disc = None
        if pass_scores and reject_scores:
            delta = round(compute_cliffs_delta(pass_scores, reject_scores), 4)
            ci_lo, ci_hi = run_bootstrap_delta_ci(
                pass_scores, reject_scores, n_bootstrap=args.bootstrap_n,
            )
            ci_lo, ci_hi = round(ci_lo, 3), round(ci_hi, 3)
            is_disc = not (ci_lo <= 0 <= ci_hi)

        d_str  = f"{delta:+.3f}" if delta  is not None else "   N/A"
        lo_str = f"{ci_lo:+.3f}" if ci_lo  is not None else "   N/A"
        hi_str = f"{ci_hi:+.3f}" if ci_hi  is not None else "   N/A"
        disc   = "✓" if is_disc else ("✗" if is_disc is False else "?")

        print(f"{val:>12.3f} {result.n_passed:>6} {result.rejection_rate:>7.1%} "
              f"{d_str:>8} {lo_str:>7} {hi_str:>7} {disc:>6}")

        rows.append({
            "threshold_param": args.param,
            "threshold_value": val,
            "n_in": result.n_baseline,
            "n_out": result.n_passed,
            "rejection_rate": result.rejection_rate,
            "cliffs_delta": delta,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "is_discriminative": is_disc,
            "rejected_ids": sorted(result.rejected_ids),
        })

    print(f"{'─'*74}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"sweep_{args.filter}_{args.param}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"filter": args.filter, "param": args.param, "results": rows},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 → {out_path}")


if __name__ == "__main__":
    main()
