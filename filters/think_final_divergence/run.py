"""
Think-Final Divergence Filter  (⑥ Ablation — 추론 충실성)

<think>의 추론이 최종 Step 4 출력과 어긋난 "사후정당화" 샘플을 탐지해 제거한다.

규칙 기반 hedge 신호만 사용 (NLI 제거):
  - <think> 내 한국어 hedge 표지 수 ≥ hedge_threshold
  - think vs Step4 hedge 밀도 격차(per 100 words) ≥ gap_threshold
  → 둘 중 하나라도 해당하면 탈락

임계값은 02_after_structure.json (n=273) 실측 분포 기반:
  hedge p95=8 → 10 (보수적), gap p95=2.04 → 2.0 / OR 조합 탈락율 7.7%

단독 실행  python -m filters.think_final_divergence.run --input <in.json> --output <out.json>
파이프라인  python scripts/run_pipeline.py --input <in.json> --output_dir <dir>
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

import yaml

from filters.base import BaseFilter
from utils.schema import CoTSample


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STEP4_RE = re.compile(
    r"\[Step\s*4\](.*?)(?=\*{0,2}정답\*{0,2}\s*[：:]|\Z)", re.DOTALL
)

_KO_HEDGE_PATTERNS: list[str] = [
    "하지만", "그러나", "다시 생각", "확실하지", "정정",
    "아닌 것 같", "다른 것 같", "혹시", "잠깐", "재고",
    "다시 보면", "다시 살펴", "재확인", "아닐 수도",
    "맞는지", "맞지 않", "수정", "틀렸", "잘못",
    "오히려", "그렇지 않", "아닌가",
]

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["proposal_filters"]["think_final_divergence"]


def _count_hedges(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(p) for p in patterns)


def _hedge_rate_per_100words(text: str, patterns: list[str]) -> float:
    words = text.split()
    return _count_hedges(text, patterns) / max(len(words), 1) * 100


def _en_token_ratio(text: str) -> float:
    """영어 토큰 비율 — 교란 변수 계측용, 판정에 사용하지 않음."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if re.search(r"[A-Za-z]{2,}", t)) / len(tokens)


class ThinkFinalDivergenceFilter(BaseFilter):
    name = "think_final_divergence"

    def __init__(self):
        cfg = _load_config()
        self.hedge_threshold: int = cfg.get("hedge_threshold", 10)
        self.gap_threshold: float = cfg.get("gap_threshold", 2.0)

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:

        think_match = _THINK_RE.search(sample.cot)
        if not think_match:
            return True, None, None
        think_text = think_match.group(1).strip()

        step4_match = _STEP4_RE.search(sample.cot)
        step4_text = step4_match.group(1).strip() if step4_match else ""

        hedge_count = _count_hedges(think_text, _KO_HEDGE_PATTERNS)
        think_rate = _hedge_rate_per_100words(think_text, _KO_HEDGE_PATTERNS)
        final_rate = _hedge_rate_per_100words(step4_text, _KO_HEDGE_PATTERNS) if step4_text else 0.0
        think_final_gap = think_rate - final_rate

        sample.metadata.update({
            "think_hedge_count":    hedge_count,
            "think_final_gap":      round(think_final_gap, 4),
            "think_en_token_ratio": round(_en_token_ratio(think_text), 4),
        })

        hedge_flagged = hedge_count >= self.hedge_threshold
        gap_flagged = think_final_gap >= self.gap_threshold

        if not (hedge_flagged or gap_flagged):
            return True, think_final_gap, None

        triggered = []
        if hedge_flagged:
            triggered.append(f"hedge={hedge_count}(≥{self.hedge_threshold})")
        if gap_flagged:
            triggered.append(f"gap={think_final_gap:.3f}(≥{self.gap_threshold})")
        return False, think_final_gap, ", ".join(triggered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f = ThinkFinalDivergenceFilter()
    f.run_from_file(args.input, args.output)
