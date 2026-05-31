"""Student LoRA 학습 자동화 — arm × seed 루프. 연구계획 §5 / §6.5.

각 arm(C0/C1/C2/C3/C-rand)을 ≥3 seed 로 학습해 단일 run 노이즈를 통제한다.
실제 학습은 LLaMA-Factory(configs/train.yaml)가 수행하고, 이 스크립트는

  1. build_arms export 로 arm 별 SFT jsonl 생성 (data-dir/train_<arm>.jsonl)
  2. configs/dataset_info.json 에 komed_<arm> 항목 자동 등록
  3. (arm, seed) 마다 llamafactory-cli train 을 dataset/output_dir/seed/data_seed 오버라이드로 실행
     (seed=가중치 초기화·드롭아웃, data_seed=데이터 셔플/배치 순서 — 둘 다 같은 시드로 묶어 독립 run 보장)

loss 는 train.yaml 설정(질문 마스킹 = sharegpt user 턴 미학습, assistant 만 학습)을
그대로 따른다. seq 4096 도 train.yaml(cutoff_len).

사용:
    # C1·C3·C-rand 를 seed 42,43,44 로 (실행)
    python src/train/train_lora.py --unified data/unified.jsonl \
        --arms C1 C3 C-rand --seeds 42 43 44
    # 명령만 확인
    python src/train/train_lora.py --unified data/unified.jsonl --arms C3 --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_ARMS = _ROOT / "src" / "dataset" / "build_arms.py"
_DATASET_INFO = _ROOT / "configs" / "dataset_info.json"
_SHAREGPT_ENTRY = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"},
}


def export_arm(unified: Path, arm: str, data_dir: Path, dry: bool) -> Path:
    out = data_dir / f"train_{arm}.jsonl"
    cmd = [sys.executable, str(_BUILD_ARMS), "export",
           "--input", str(unified), "--arm", arm, "--output", str(out)]
    print(f"[train] export {arm}: $ {' '.join(cmd)}")
    if not dry:
        subprocess.run(cmd, check=True)
    return out


def register_dataset(arm: str, data_path: Path, dataset_dir: Path, dry: bool) -> str:
    """dataset_info.json 에 komed_<arm> 등록(없으면 추가). LLaMA-Factory 가 dataset_dir 기준
    상대 file_name 을 찾으므로 data_path 를 dataset_dir 기준 상대경로로 기록한다."""
    name = f"komed_{arm.lower().replace('-', '_')}"
    info = json.loads(_DATASET_INFO.read_text(encoding="utf-8")) if _DATASET_INFO.exists() else {}
    try:
        rel = str(data_path.resolve().relative_to(dataset_dir.resolve()))
    except ValueError:
        rel = str(data_path.resolve())
    entry = {"file_name": rel, **_SHAREGPT_ENTRY}
    if info.get(name) != entry:
        info[name] = entry
        print(f"[train] dataset_info 등록: {name} → {rel}")
        if not dry:
            _DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return name


def train_one(config: Path, dataset: str, dataset_dir: Path, output_dir: Path,
              seed: int, extra: list[str], dry: bool) -> None:
    cmd = ["llamafactory-cli", "train", str(config),
           f"dataset={dataset}", f"dataset_dir={dataset_dir}",
           f"output_dir={output_dir}", f"seed={seed}", f"data_seed={seed}", *extra]
    print(f"[train] $ {' '.join(cmd)}")
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="arm × seed LoRA 학습 자동화 (LLaMA-Factory)")
    p.add_argument("--unified", required=True, help="build_arms unified jsonl")
    p.add_argument("--arms", nargs="+", default=["C1", "C3", "C-rand"],
                   choices=["C0", "C1", "C2", "C3", "C-rand"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--config", default=str(_ROOT / "configs" / "train.yaml"))
    p.add_argument("--data-dir", default=str(_ROOT / "data" / "sft"),
                   help="arm jsonl 출력 + LLaMA-Factory dataset_dir")
    p.add_argument("--output-root", default=str(_ROOT / "output"))
    p.add_argument("--dry-run", action="store_true", help="명령만 출력(실행 안 함)")
    p.add_argument("extra", nargs="*", help="llamafactory-cli 에 그대로 전달할 key=value (예: cutoff_len=8192)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    out_root = Path(args.output_root)

    plan = [(arm, seed) for arm in args.arms for seed in args.seeds]
    print(f"[train] 계획: {len(plan)} run = arms{args.arms} × seeds{args.seeds}"
          + ("  (DRY-RUN)" if args.dry_run else ""))

    for arm in args.arms:
        data_path = export_arm(Path(args.unified), arm, data_dir, args.dry_run)
        dataset = register_dataset(arm, data_path, data_dir, args.dry_run)
        for seed in args.seeds:
            out_dir = out_root / f"qwen3-8b-{arm.lower()}-s{seed}"
            train_one(config, dataset, data_dir, out_dir, seed, args.extra, args.dry_run)

    print(f"[train] 완료 ({len(plan)} run){'  — DRY-RUN, 실제 실행 안 됨' if args.dry_run else ''}.")
    print("[train] 다음: 각 output_dir 을 evaluate.py --lora 로 평가 후 stats.py 로 집계.")


if __name__ == "__main__":
    main()
