"""
Step Coverage Filter  (④ Ablation)

추론이 충분히 상세한지, Step 3에서 A~E 선택지를 충분히 검토했는지 2계층으로 판단.

체크 기준:
  계층 1 — Step 3 선택지 커버리지: 5지선다 중 coverage_threshold 이상 언급 (기본 0.6 = 3/5)
  계층 2 — Step별 내용 밀도: 각 Step의 공백 분리 토큰 수가 최솟값 이상

score 반환값: Step 3 선택지 커버리지 비율 (0.0 ~ 1.0)

담당: B
참고: 이 연구 자체 설계 — 5지선다 특화, 계획서 §5-2 ④
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

import yaml

from filters.base import BaseFilter
from utils.schema import CoTSample


COVERAGE_THRESHOLD = 0.4  # DeepSeek-R1 재보정: GPT-4/4o 기준 0.6 → 13% 거부율 + judge J-gap=-0.054 불일치 → 0.4(2/5)로 하향
MIN_TOKENS = {            # DeepSeek-R1 생성 패턴 재보정 (GPT-4/4o 실측값에서 하향 조정)
    "step1": 8,           # <8:  문제 재진술 one-liner 제거 (GPT-4/4o 실측 최솟값 8tok — 이전 10은 9tok 정상 샘플 제거)
    "step2": 15,          # <15: ~0% — 안전망 (유지)
    "step3": 25,          # <25: 5지선다 최소 기술 미달 제거 (이전 40은 13% 과다 거부 — judge 불일치)
    "step4": 15,          # <15: ~0% — 안전망 (유지)
}

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["ablation_filters"]["step_coverage"]


def check_option_coverage(cot: str) -> dict:
    """
    Step 3 구간에서 A~E 선택지 언급 여부 집계.

    의학 텍스트는 "vitamin C", "hepatitis B", "type A" 등 알파벳 단독 토큰이
    흔하므로 너무 느슨한 패턴은 false positive 가 많다. 다음 세 가지 명시적
    "선택지 마커" 패턴만 허용:
      1) 줄머리에 등장하는 옵션 라벨: "A.", "A)", "A:", "- A.", "* A:" 등
      2) 직접 참조: "선택지 A", "선택지A"
      3) 한국어 번호 표기: "A번"
    """
    step3_match = re.search(
        r'\[Step 3\](.*?)(?=\[Step 4\]|\Z)', cot, re.DOTALL
    )
    if not step3_match:
        return {"coverage": 0.0, "covered": [], "missing": list("ABCDE")}

    step3_text = step3_match.group(1)
    covered, missing = [], []

    for opt in "ABCDE":
        patterns = [
            # 줄머리(옵션 본문 마커): 선택적 bullet/공백 후 옵션 라벨 + 구분자
            rf'(?:^|\n)[ \t]*(?:[-*•]\s*)?{opt}[\.\):]',
            # 명시 참조
            rf'선택지\s*{opt}\b',
            # 한국어 번호 표기 — 단어 경계로 vitamin B 등 단순 알파벳 배제
            rf'\b{opt}번\b',
        ]
        if any(re.search(p, step3_text) for p in patterns):
            covered.append(opt)
        else:
            missing.append(opt)

    return {
        "coverage": len(covered) / 5,
        "covered": covered,
        "missing": missing,
    }


def check_step_density(cot: str) -> dict:
    """각 Step의 공백 분리 토큰 수를 계산하고 최솟값 미달 Step을 violations로 반환."""
    step_pattern = re.compile(
        r'\[Step (\d)\](.*?)(?=\[Step \d\]|\*{0,2}정답\*{0,2}\s*[：:]|$)', re.DOTALL
    )
    steps = step_pattern.findall(cot)

    step_lengths = {}
    for step_num, content in steps:
        tokens = len(content.strip().split())
        step_lengths[f"step{step_num}"] = tokens

    violations = {
        k: v for k, v in step_lengths.items()
        if v < MIN_TOKENS.get(k, 10)
    }
    return {"step_lengths": step_lengths, "violations": violations}


class StepCoverageFilter(BaseFilter):
    name = "step_coverage"

    def __init__(
        self,
        coverage_threshold: float | None = None,
        min_tokens: dict | None = None,
    ):
        # yaml 우선, 명시 인자가 들어오면 override. 다른 ablation 필터와 동일한 패턴.
        cfg = _load_config()
        self.coverage_threshold = (
            coverage_threshold if coverage_threshold is not None
            else cfg.get("coverage_threshold", COVERAGE_THRESHOLD)
        )
        if min_tokens is None:
            min_tokens = cfg.get("min_tokens") or dict(MIN_TOKENS)
        self.min_tokens = dict(min_tokens)

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        cot = sample.cot or ""

        # 계층 1: Step 3 선택지 커버리지
        cov = check_option_coverage(cot)
        score = round(cov["coverage"], 4)

        if cov["coverage"] < self.coverage_threshold:
            missing_str = ",".join(cov["missing"])
            return False, score, f"Step3 커버리지 부족({cov['coverage']:.0%}): {missing_str} 미언급"

        # 계층 2: Step별 내용 밀도
        density = check_step_density(cot)
        if density["violations"]:
            viol_str = ", ".join(
                f"{k}({v}tok)" for k, v in density["violations"].items()
            )
            return False, score, f"Step 밀도 부족: {viol_str}"

        return True, score, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f = StepCoverageFilter()
    f.run_from_file(args.input, args.output)
