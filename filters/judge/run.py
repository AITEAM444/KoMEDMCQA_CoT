"""
LLM-as-a-Judge Filter  (F8 TAIL — 항상 적용, Ablation 대상 아님)

G-Eval 채점 로직을 BaseFilter 인터페이스에 연결하는 어댑터.
채점: 3차원 (step_coherence / korean_fluency / step_coverage) × N회 반복 → min 집계.
루브릭·프롬프트 상세는 data/evaluated/score_faithfulness.py 참조.

담당: D
참고: Liu et al. 2023 (G-Eval) + DC-CoT arXiv:2505.18759
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from filters.base import BaseFilter
from filters.judge.score_faithfulness import DIM_ORDER, make_client, score_one_dim
from utils.schema import CoTSample, JudgeScore


_CHOICE_LABELS = ["A", "B", "C", "D", "E"]
_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"

# Reasoning 모델(DeepSeek-R1 등)이 출력하는 <think>...</think> 블록은
# 최종 4-Step 답안과 별개의 내부 추론이므로 G-Eval 채점 대상에서 제외한다.
# (think 자체의 흔들림은 ⑥ think_final_divergence 필터가 별도로 다룬다.)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _load_judge_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["tail_filters"]["judge"]


class JudgeFilter(BaseFilter):
    name = "judge"

    def __init__(self):
        cfg = _load_judge_config()
        self._temperature: float = cfg["temperature"]
        self._reps: int = cfg["n_eval_per_sample"]
        self._threshold: float = cfg["min_score_threshold"]
        self._client = make_client()

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        sample_row = {
            "question": sample.question,
            "subset": sample.subset,
            **{
                label: (sample.choices[i] if i < len(sample.choices) else "")
                for i, label in enumerate(_CHOICE_LABELS)
            },
        }
        reasoning = _THINK_BLOCK_RE.sub("", sample.cot or "").strip()
        result = {
            "correct": _CHOICE_LABELS[sample.answer],
            "predicted": (
                _CHOICE_LABELS[sample.predicted_answer]
                if sample.predicted_answer is not None
                else "미확인"
            ),
            "reasoning": reasoning,
        }

        dim_scores: dict[str, float | None] = {}
        for dim in DIM_ORDER:
            res = score_one_dim(
                self._client, sample_row, result, dim,
                self._reps, self._temperature, sleep_on_error=2.0,
            )
            dim_scores[dim] = res["mean"]

        failed_dims = [d for d, v in dim_scores.items() if v is None]
        if failed_dims:
            return False, None, f"채점 파싱 실패: {', '.join(failed_dims)}"

        judge_score = JudgeScore(
            step_coherence=dim_scores["step_coherence"],
            korean_fluency=dim_scores["korean_fluency"],
            step_coverage=dim_scores["step_coverage"],
        )
        sample.judge_score = judge_score

        score_min = judge_score.minimum
        passed = score_min >= self._threshold
        reason = (
            None
            if passed
            else (
                f"G-Eval min {score_min:.2f} < {self._threshold} ("
                f"SC={dim_scores['step_coherence']:.1f}, "
                f"KF={dim_scores['korean_fluency']:.1f}, "
                f"CV={dim_scores['step_coverage']:.1f})"
            )
        )
        return passed, score_min, reason


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f = JudgeFilter()
    f.run_from_file(args.input, args.output)
