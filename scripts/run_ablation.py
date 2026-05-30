"""
Phase 3 — Ablation Study 실행 스크립트 (보고서 §6).

사용법:
    # Step 1: Ablation 대상 필터(F3·F4·F5) 독립 측정 — Cliff's δ + Bootstrap CI
    python scripts/run_ablation.py step1 \\
        --input data/filtered/02_after_structure.json \\
        --output_dir results/ablation/step1/

    # Step 2: 조합 실험 (Step 1 결과 확인 후 조합 직접 지정)
    python scripts/run_ablation.py step2 \\
        --input data/filtered/02_after_structure.json \\
        --combo kppl step_coverage \\
        --combo kppl step_coverage answer_consistency \\
        --output_dir results/ablation/step2/

    # Step 3: 최적 파이프라인 확정
    python scripts/run_ablation.py step3 \\
        --step2_dir results/ablation/step2/

주의:
  - Step 2의 조합은 Step 1 결과를 보고 결정한다 (보고서 §6 Step 2 원칙).
  - F7 (think_final_divergence) 은 본 Ablation 대상이 아니다 — 정답이 맞는
    샘플 중 추론 흔들림을 제거하므로 Judge Score Δ/δ 로 변별되지 않는다.
    F7 의 효과는 7~8장 C2(F7 제외 최적) vs C3(C2+F7) Student 다운스트림
    성능으로 별도 검증한다 (보고서 §7·§8).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_samples
from utils.schema import FilterName
from evaluation.ablation import (
    FilterAblationResult,
    measure_independent,
    compute_cliffs_delta,
    run_bootstrap_delta_ci,
    print_independent_results,
    print_overlap_matrix,
)


# Ablation 대상 = F3 K-PPL · F4 Step Coverage · F5 Answer Consistency.
# F7(think_final_divergence)은 Judge Score 로 변별되지 않으므로 본 Step 1에서
# 제외한다 (보고서 §6 도입부 + §7 C2 vs C3 설계). 파이프라인 단독 실행은
# `python -m filters.think_final_divergence.run ...` 으로 가능.
ABLATION_FILTER_CLASSES = {
    FilterName.KPPL:               ("filters.kppl.run",               "KPPLFilter"),
    FilterName.STEP_COVERAGE:      ("filters.step_coverage.run",      "StepCoverageFilter"),
    FilterName.ANSWER_CONSISTENCY: ("filters.answer_consistency.run", "AnswerConsistencyFilter"),
}

# Step 2 조합에서 추가로 호출 가능한 필터 (F7 포함). step1 에서는 사용하지 않음.
COMBO_FILTER_CLASSES = {
    **ABLATION_FILTER_CLASSES,
    FilterName.THINK_FINAL_DIVERGENCE: (
        "filters.think_final_divergence.run", "ThinkFinalDivergenceFilter"
    ),
}

# Step 1 에서 측정할 Ablation 필터 순서.
STEP1_FILTERS = [
    FilterName.KPPL,
    FilterName.STEP_COVERAGE,
    FilterName.ANSWER_CONSISTENCY,
]


def _load_filter(filter_name: str, registry: dict = COMBO_FILTER_CLASSES):
    key = FilterName(filter_name)
    if key not in registry:
        raise KeyError(
            f"{filter_name}: 해당 단계에서 사용 가능한 필터가 아닙니다 "
            f"(허용 목록: {[k.value for k in registry]})"
        )
    module_path, class_name = registry[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def _mean_judge(samples: list) -> float | None:
    scores = [s.judge_score.minimum for s in samples if s.judge_score is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


def _result_to_dict(r: FilterAblationResult) -> dict:
    return {
        "filter_name": r.filter_name,
        "n_baseline": r.n_baseline,
        "n_passed": r.n_passed,
        "rejection_rate": r.rejection_rate,
        "score_delta": r.score_delta,
        "filter_efficiency": r.filter_efficiency,
        "cliffs_delta": r.cliffs_delta,
        "delta_ci_lower": r.delta_ci_lower,
        "delta_ci_upper": r.delta_ci_upper,
        "is_discriminative": r.is_discriminative,
        "effect_label": r.effect_label,
        "rejected_ids": sorted(r.rejected_ids),
    }


def step1(args):
    """독립 측정: 각 Ablation 필터를 PRE 통과 데이터에 단독 적용 → Cliff's δ 산출."""
    baseline = load_samples(args.input)
    print(f"[Step 1] baseline: {len(baseline)}개 샘플 로드")

    results = []
    for filter_name in STEP1_FILTERS:
        try:
            f = _load_filter(filter_name.value, registry=ABLATION_FILTER_CLASSES)
            result = measure_independent(baseline, f)
        except NotImplementedError:
            print(f"[SKIP] {filter_name.value} — 미구현")
            continue
        except KeyError as e:
            print(f"[SKIP] {filter_name.value} — {e}")
            continue

        # Cliff's δ + Bootstrap CI (보고서 §6: 통과·제거 두 집단의 Judge Score 분포 차이)
        pass_scores = [
            s.judge_score.minimum for s in baseline
            if s.id not in result.rejected_ids and s.judge_score is not None
        ]
        reject_scores = [
            s.judge_score.minimum for s in baseline
            if s.id in result.rejected_ids and s.judge_score is not None
        ]

        if pass_scores and reject_scores:
            result.cliffs_delta = round(
                compute_cliffs_delta(pass_scores, reject_scores), 4
            )
            lo, hi = run_bootstrap_delta_ci(
                pass_scores, reject_scores, n_bootstrap=args.bootstrap_n,
            )
            result.delta_ci_lower = round(lo, 4)
            result.delta_ci_upper = round(hi, 4)

        cd = f"{result.cliffs_delta:+.3f}" if result.cliffs_delta is not None else "N/A"
        ci = (
            f"[{result.delta_ci_lower:+.3f}, {result.delta_ci_upper:+.3f}]"
            if result.delta_ci_lower is not None else "N/A"
        )
        print(
            f"  [{result.filter_name}] "
            f"Rej={result.rejection_rate:.1%}, δ={cd} {ci}, "
            f"Disc={result.is_discriminative}"
        )
        results.append(result)

    if results:
        print_independent_results(results)
        print_overlap_matrix(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "step1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([_result_to_dict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 → {out_path}")


def step2(args):
    """조합 실험: 지정된 필터 조합들을 순차 적용하며 단계별 메트릭 측정.

    각 단계에서 직전 단계 대비 변화량 (Rejection Rate / Score Δ / 필터 효율) 을
    기록한다. 단계별 Cliff's δ 는 step1 단독 측정의 비교 기준이므로 여기서는
    재계산하지 않는다 — 조합 전체 효과는 7~8장 다운스트림으로 검증.
    """
    baseline = load_samples(args.input)
    print(f"[Step 2] baseline: {len(baseline)}개 샘플 로드")

    if not args.combo:
        print("--combo 를 지정해주세요. 예: --combo kppl step_coverage answer_consistency")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for combo in args.combo:
        combo_name = "_".join(combo)
        print(f"\n[조합] {' → '.join(combo)}")

        current = copy.deepcopy(baseline)
        steps = [{"stage": "baseline", "n": len(current), "mean_judge": _mean_judge(current)}]

        for filter_name_str in combo:
            n_before = len(current)
            before_mean = _mean_judge(current)

            try:
                f = _load_filter(filter_name_str)
                current = f.run(current, verbose=False)
            except NotImplementedError:
                print(f"  [SKIP] {filter_name_str} — 미구현")
                steps.append({"stage": filter_name_str, "skipped": True})
                continue
            except KeyError as e:
                print(f"  [SKIP] {filter_name_str} — {e}")
                steps.append({"stage": filter_name_str, "skipped": True})
                continue

            n_after = len(current)
            after_mean = _mean_judge(current)
            rej_rate = round((n_before - n_after) / n_before, 4) if n_before > 0 else 0.0
            score_delta = (
                round(after_mean - before_mean, 4)
                if after_mean is not None and before_mean is not None else None
            )
            efficiency = (
                round(score_delta / rej_rate, 4)
                if score_delta is not None and rej_rate > 0 else None
            )

            steps.append({
                "stage": filter_name_str,
                "n_before": n_before,
                "n_after": n_after,
                "rejection_rate": rej_rate,
                "mean_judge_before": before_mean,
                "mean_judge_after": after_mean,
                "score_delta": score_delta,
                "filter_efficiency": efficiency,
            })
            print(f"  [{filter_name_str}] {n_before} → {n_after}  Rej={rej_rate:.1%}  Δ={score_delta}")

        out_path = args.output_dir / f"step2_{combo_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"combo": combo, "steps": steps}, f, ensure_ascii=False, indent=2)
        print(f"  저장 → {out_path}")


def step3(args):
    """최적 파이프라인 후보 추리기 — 필터 효율 정렬 및 샘플 수 조건.

    보고서 §6 경고: 효과 크기/효율 임계값만으로 채택을 단정하지 않는다.
    여기서는 조합 후보를 효율 순으로 정렬해 보여주고, 최종 채택은 7~8장
    다운스트림 Student 성능으로 결정한다.
    """
    combo_files = sorted(args.step2_dir.glob("step2_*.json"))
    if not combo_files:
        print(f"step2_*.json 파일을 {args.step2_dir}에서 찾을 수 없습니다.")
        return

    results = []
    for path in combo_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        combo = data["combo"]
        baseline_step = data["steps"][0]
        active_steps = [s for s in data["steps"][1:] if not s.get("skipped") and "n_after" in s]

        if not active_steps:
            continue

        n_baseline = baseline_step["n"]
        n_final = active_steps[-1]["n_after"]
        baseline_mean = baseline_step["mean_judge"]
        final_mean = active_steps[-1]["mean_judge_after"]

        total_rej = round((n_baseline - n_final) / n_baseline, 4) if n_baseline > 0 else 0.0
        total_delta = (
            round(final_mean - baseline_mean, 4)
            if final_mean is not None and baseline_mean is not None else None
        )
        total_efficiency = (
            round(total_delta / total_rej, 4)
            if total_delta is not None and total_rej > 0 else None
        )

        results.append({
            "combo": combo,
            "n_final": n_final,
            "total_rejection_rate": total_rej,
            "total_score_delta": total_delta,
            "total_filter_efficiency": total_efficiency,
            "min_samples_ok": n_final >= args.min_samples,
        })

    results.sort(key=lambda x: x["total_filter_efficiency"] or float("-inf"), reverse=True)

    print("\n" + "=" * 75)
    print("ABLATION STUDY — Step 3: 조합 후보 정렬 (최종 채택은 다운스트림으로)")
    print("=" * 75)
    print(f"{'조합':<35} {'N_final':>8} {'Rej%':>7} {'Δ':>8} {'Eff':>9} {'샘플수':>6}")
    print("-" * 75)
    for r in results:
        combo_str = "→".join(r["combo"])
        delta = f"{r['total_score_delta']:+.4f}" if r["total_score_delta"] is not None else "   N/A"
        eff = f"{r['total_filter_efficiency']:.4f}" if r["total_filter_efficiency"] is not None else "   N/A"
        ok = "✓" if r["min_samples_ok"] else "✗"
        rej = f"{r['total_rejection_rate']:.1%}"
        print(f"{combo_str:<35} {r['n_final']:>8} {rej:>7} {delta:>8} {eff:>9} {ok:>6}")
    print("=" * 75)

    valid = [r for r in results if r["min_samples_ok"]]
    if valid:
        best = valid[0]
        print(f"\n효율 기준 1순위 후보: {' → '.join(best['combo'])}")
        print(f"  필터 효율: {best['total_filter_efficiency']}  최종 샘플 수: {best['n_final']}")
        print(f"  → 최종 채택은 7~8장 Student 다운스트림 성능으로 확정.")
    else:
        best_eff = results[0]
        best_n = max(results, key=lambda x: x["n_final"])
        print(f"\n⚠️  트레이드오프 — 두 기준이 충돌합니다 (보고서 §6 Step 3):")
        print(f"  필터 효율 최대: {' → '.join(best_eff['combo'])}  (효율={best_eff['total_filter_efficiency']}, n={best_eff['n_final']})")
        print(f"  샘플 수 최대:   {' → '.join(best_n['combo'])}  (효율={best_n['total_filter_efficiency']}, n={best_n['n_final']})")
        print(f"  → 논문에서 트레이드오프 명시 후 선택 근거를 서술하세요.")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Ablation Study")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("step1", help="독립 측정 — 각 필터 단독 효과 (Cliff's δ + CI)")
    p1.add_argument("--input", type=Path, required=True,
                    help="PRE 필터(F1·F2) 통과 데이터 (02_after_structure.json)")
    p1.add_argument("--output_dir", type=Path, default=Path("results/ablation/step1"))
    p1.add_argument("--bootstrap_n", type=int, default=10_000,
                    help="Cliff's δ Bootstrap CI 반복 횟수")

    p2 = sub.add_parser("step2", help="조합 실험 — 보완적 필터 조합 효과")
    p2.add_argument("--input", type=Path, required=True,
                    help="PRE 필터(F1·F2) 통과 데이터 (02_after_structure.json)")
    p2.add_argument("--combo", nargs="+", action="append", metavar="FILTER",
                    help="조합 지정 (여러 번 사용 가능). 예: --combo kppl step_coverage answer_consistency")
    p2.add_argument("--output_dir", type=Path, default=Path("results/ablation/step2"))

    p3 = sub.add_parser("step3", help="조합 후보 정렬")
    p3.add_argument("--step2_dir", type=Path, required=True)
    p3.add_argument("--min_samples", type=int, default=200,
                    help="통계적 변별력 확보를 위한 최소 샘플 수 (보고서: 200~300 권장)")

    args = parser.parse_args()

    if args.command == "step1":
        step1(args)
    elif args.command == "step2":
        step2(args)
    elif args.command == "step3":
        step3(args)


if __name__ == "__main__":
    main()
