"""
Teacher Correctness Filter  (① PRE — 항상 적용)

CoT의 최종 예측 답안이 정답과 일치하지 않는 샘플을 제거한다.
Teacher 프롬프트가 강제하는 "정답: X" (X ∈ A-E) 형식을 우선 파싱하며,
4단계 폴백 패턴과 파싱 신뢰도 기반 필터링을 지원한다.

담당: B
참고: DC-CoT (arXiv:2505.18759, ICLR 2026)
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

from filters.base import BaseFilter
from utils.schema import CoTSample


# ── 정규식 4단계 폴백 ─────────────────────────────────────────────────────────
# 프롬프트 제약: 마지막 줄이 정확히 "정답: X" 여야 함.
# 모델이 제약을 어길 경우를 대비해 순차 폴백 적용.
#
# 다른 필터(structure, answer_consistency, step_coverage)와 동일하게
# 마크다운 볼드(**), 전각 콜론(：) 을 strict/loose 모두에서 허용한다.
# (예: "**정답: A**" 가 structure 는 통과하면서 correctness 에서만 0.5 신뢰도로
#  떨어져 정답이 탈락하던 silent false negative 방지)

# 1순위: 줄 끝에 정답 표기 — 프롬프트 준수 케이스 (볼드/전각 콜론 허용)
_STRICT = re.compile(
    r"\*{0,2}정답\*{0,2}\s*[：:]\s*([A-E])\*{0,2}[\s.]*$",
    re.MULTILINE,
)

# 2순위: 괄호·따옴표 등 추가 토큰이 붙은 경우
_LOOSE = re.compile(
    r"\*{0,2}정답\*{0,2}\s*[：:]\s*[\[\(\"']?([A-E])[\]\)\"'.*]?[\s.*]*$",
    re.MULTILINE,
)

# 3·4순위 공통 패턴: "정답: X" 를 위치 제약 없이 추출
# 3순위는 마지막 "정답:" 줄에만 적용(사족 허용), 4순위는 전체 텍스트에 적용
_ANYWHERE = re.compile(
    r"정답\*{0,2}\s*[：:]\s*\*{0,2}([A-E])",
    re.IGNORECASE,
)

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# 파싱 방법별 신뢰도 — 필터링 및 로그용
_PARSE_META = {
    "strict":    1.0,
    "loose":     0.8,
    "last_line": 0.7,
    "anywhere":  0.5,
    "fail":      0.0,
}


def _get_last_answer_line(cot: str) -> str:
    """CoT에서 '정답:' 포함 마지막 줄을 반환. 없으면 빈 문자열."""
    for line in reversed(cot.splitlines()):
        if re.search(r"\*{0,2}정답\*{0,2}\s*[：:]", line):
            return line
    return ""


def extract_predicted_answer(cot: str) -> tuple[int | None, str]:
    """
    CoT 텍스트에서 최종 선지를 추출한다.

    Returns
    -------
    (answer_idx, parse_method)
        answer_idx  : 0-indexed int (A=0 … E=4), 추출 실패 시 None
        parse_method: "strict" | "loose" | "last_line" | "anywhere" | "fail"
    """
    # 1·2순위: 줄 끝 앵커 기반
    for method, pattern in [("strict", _STRICT), ("loose", _LOOSE)]:
        matches = pattern.findall(cot)
        if matches:
            return LETTER_TO_IDX[matches[-1].upper()], method

    # 3순위: 마지막 "정답:" 줄에 한정해 추출 (같은 줄 사족 허용)
    last_line = _get_last_answer_line(cot)
    if last_line:
        m = _ANYWHERE.search(last_line)
        if m:
            return LETTER_TO_IDX[m.group(1).upper()], "last_line"

    # 4순위: 전체 텍스트 어디서든
    matches = _ANYWHERE.findall(cot)
    if matches:
        return LETTER_TO_IDX[matches[-1].upper()], "anywhere"

    return None, "fail"


class CorrectnessFilter(BaseFilter):
    name = "correctness"

    def __init__(self, min_confidence: float = 0.8):
        """
        Parameters
        ----------
        min_confidence : float
            이 값 미만의 파싱 신뢰도를 가진 샘플은 정답이어도 제거.
            기본값 0.8 → loose(0.8)까지 허용, last_line(0.7)·anywhere(0.5)는 제거.
        """
        super().__init__()
        self.min_confidence = min_confidence

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        predicted, method = extract_predicted_answer(sample.cot)
        confidence = _PARSE_META[method]

        if sample.predicted_answer is None:
            sample.predicted_answer = predicted

        if predicted is None:
            return False, 0.0, "답안 추출 실패"

        if confidence < self.min_confidence:
            return (
                False,
                confidence,
                f"파싱 신뢰도 미달 (method={method}, confidence={confidence})",
            )

        if predicted != sample.answer:
            return (
                False,
                confidence,
                f"오답 (예측={predicted}, 정답={sample.answer}, method={method})",
            )

        return True, confidence, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-confidence", type=float, default=0.8,
        help="파싱 신뢰도 임계값 (기본 0.8, last_line=0.7·anywhere=0.5 제거)",
    )
    args = parser.parse_args()

    f = CorrectnessFilter(min_confidence=args.min_confidence)
    f.run_from_file(args.input, args.output)
