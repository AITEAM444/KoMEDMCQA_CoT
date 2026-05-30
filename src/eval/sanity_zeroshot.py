"""Sanity zero-shot — base / R1-Distill 하한 점검 (성능 경쟁 아님).

연구계획 §6.4: SFT 가 실제로 작동했는지(C1 > base) 와 벤치 비포화를 확인하기 위한
출발점. evaluate.py 를 zero-shot(미학습)으로 여러 모델에 대해 돌려 비교 표를 만든다.

  · base Qwen3-8B (zero-shot)            : 학습 전 출발점. C1 학습본이 이보다 분명히
                                            높아야 distillation 이 작동한 것.
  · DeepSeek-R1-Distill-Qwen-8B (zero-shot): 절대 위치 참조점(선택).

evaluate.py 를 subprocess 로 호출(LoRA 없이)하므로 평가 로직은 단일 소스로 유지된다.

사용:
    python src/eval/sanity_zeroshot.py --split test --output-dir results/sanity
    # 참조 모델 추가
    python src/eval/sanity_zeroshot.py --split test \
        --models Qwen/Qwen3-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-8B
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_EVALUATE = Path(__file__).resolve().parent / "evaluate.py"


def _slug(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", model).strip("_")


def run_one(model: str, split: str, out_path: Path, total: int, extra: list[str]) -> dict | None:
    """evaluate.py 를 zero-shot 으로 실행하고 결과 jsonl 에서 정확도 집계."""
    cmd = [sys.executable, str(_EVALUATE), "--model", model, "--split", split,
           "--total", str(total), "--output", str(out_path), *extra]
    print(f"\n[sanity] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        print(f"[sanity] ⚠ 결과 파일 없음: {out_path}")
        return None
    n = n_ok = 0
    for line in open(out_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        n_ok += int(bool(r.get("is_correct")))
    return {"model": model, "n": n, "correct": n_ok, "acc": (n_ok / n) if n else 0.0}


def main():
    p = argparse.ArgumentParser(description="zero-shot sanity (base / distill 하한)")
    p.add_argument("--models", nargs="+",
                   default=["Qwen/Qwen3-8B"],
                   help="zero-shot 평가할 모델들 (기본 base Qwen3-8B). 참조용으로 distill 추가 가능.")
    p.add_argument("--split", default="test", choices=["test", "dev", "train"])
    p.add_argument("--total", type=int, default=-1)
    p.add_argument("--output-dir", default="results/sanity")
    p.add_argument("--enable-thinking", action="store_true",
                   help="evaluate.py 에 --enable-thinking 전달 (R1-Distill 류는 켜는 게 자연스러움)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--enable-thinking"] if args.enable_thinking else []

    rows = []
    for model in args.models:
        out_path = out_dir / f"sanity_{_slug(model)}_{args.split}.jsonl"
        res = run_one(model, args.split, out_path, args.total, extra)
        if res:
            rows.append(res)

    print("\n" + "=" * 56)
    print(f"Zero-shot sanity — KorMedMCQA {args.split}")
    print("=" * 56)
    for r in rows:
        print(f"  {r['model']:45s}: {r['correct']}/{r['n']} = {r['acc']:.4f}")
    print("\n주의: 이 수치는 '하한/참조점'일 뿐 성능 경쟁이 아니다. "
          "C1 학습본 ACC 가 base zero-shot 보다 분명히 높아야 distillation 이 작동한 것.")

    summary = out_dir / f"sanity_summary_{args.split}.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"요약 저장 → {summary}")


if __name__ == "__main__":
    main()
