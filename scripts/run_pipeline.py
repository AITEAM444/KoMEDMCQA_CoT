"""
Phase 4용 순차 파이프라인 실행 스크립트.

Phase 3 Ablation Study 완료 후, 최적 필터 조합이 확정된 상태에서 전체 Train 데이터에 적용.
Phase 3 (Ablation Study) 실행은 scripts/run_ablation.py 사용.

사용법:
    python scripts/run_pipeline.py \\
        --input data/raw/cot_train.json \\
        --output_dir data/filtered/

    # 특정 Ablation 필터 제외 (Phase 3 결과로 제외가 결정된 경우)
    # 또는 C2 산출용으로 think_final_divergence 제외
    python scripts/run_pipeline.py \\
        --input data/raw/cot_train.json \\
        --output_dir data/filtered/ \\
        --skip step_coverage answer_consistency

각 필터 단계의 출력이 numbered JSON으로 저장된다:
    00_raw.json
    01_after_correctness.json
    02_after_structure.json
    03_after_kppl.json
    04_after_step_coverage.json
    05_after_answer_consistency.json
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_samples, save_samples
from evaluation.metrics import compute_pipeline_report, print_report


PIPELINE = [
    # PRE — 항상 적용
    ("01_after_correctness",        "filters.correctness.run",        "CorrectnessFilter"),
    ("02_after_structure",          "filters.structure.run",          "StructureFilter"),
    # Ablation 필터 — Phase 3 결과로 조합 확정 후 사용. --skip으로 제외 가능
    ("03_after_kppl",               "filters.kppl.run",               "KPPLFilter"),
    ("04_after_step_coverage",      "filters.step_coverage.run",      "StepCoverageFilter"),
    ("05_after_answer_consistency",        "filters.answer_consistency.run",        "AnswerConsistencyFilter"),
]


def import_filter(module_path: str, class_name: str):
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def run(input_path: Path, output_dir: Path, skip: list[str] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or []

    # 00_raw 복사
    raw_path = output_dir / "00_raw.json"
    shutil.copy(input_path, raw_path)
    print(f"[START] {len(load_samples(input_path))}개 샘플 로드")

    snapshots = {"00_raw": load_samples(raw_path)}
    current_path = raw_path

    for output_name, module_path, class_name in PIPELINE:
        filter_name = output_name.split("_after_")[-1]
        output_path = output_dir / f"{output_name}.json"

        if filter_name in skip:
            print(f"[SKIP] {filter_name}")
            shutil.copy(current_path, output_path)
            snapshots[output_name] = load_samples(output_path)
            current_path = output_path
            continue

        try:
            f = import_filter(module_path, class_name)
            f.run_from_file(current_path, output_path)
            snapshots[output_name] = load_samples(output_path)
            current_path = output_path
        except ModuleNotFoundError:
            print(f"[SKIP] {filter_name} — 아직 구현되지 않음")
            shutil.copy(current_path, output_path)
            snapshots[output_name] = load_samples(output_path)
            current_path = output_path

    report = compute_pipeline_report(snapshots)
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/filtered"))
    parser.add_argument("--skip", nargs="*", default=[], help="건너뛸 필터 이름")
    args = parser.parse_args()

    run(args.input, args.output_dir, args.skip)
