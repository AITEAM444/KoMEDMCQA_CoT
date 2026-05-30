"""
Structure Format Filter  (② PRE — 항상 적용)

CoT 출력이 Teacher 프롬프트에서 강제한 4-Step 형식을 준수하는지 정규식으로 검사.

체크 기준 (모두 충족해야 통과):
  1. [Step 1] ~ [Step 4] 라벨 4개 모두 존재
  2. "정답: X" 형식 라인 존재
     (마크다운 볼드, 전각 콜론, 줄 끝 공백·마침표 허용 / 줄 앞뒤 위치 무관)

담당: B
참고: 이 연구 자체 설계 — 계획서 §5-1 ② Structure Format
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

from filters.base import BaseFilter
from utils.schema import CoTSample


# 느슨한 정답 패턴: 볼드(**), 전각 콜론(：), 줄 끝 공백·마침표 허용
ANSWER_LINE_RE = re.compile(
    r'\*{0,2}정답\*{0,2}\s*[:\：]\s*[ABCDE]\*{0,2}[\s.]*$',
    re.MULTILINE,
)


def check_structure_hard(cot: str) -> dict[str, bool]:
    positions = {
        i: m.start()
        for i in range(1, 5)
        for m in [re.search(rf'\[Step {i}\]', cot)]
        if m
    }
    in_order = (
        len(positions) == 4
        and positions[1] < positions[2] < positions[3] < positions[4]
    )
    return {
        "step1": 1 in positions,
        "step2": 2 in positions,
        "step3": 3 in positions,
        "step4": 4 in positions,
        "step_order": in_order,
        "answer_line": bool(ANSWER_LINE_RE.search(cot)),
    }


class StructureFilter(BaseFilter):
    name = "structure"

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        cot = sample.cot or ""
        checks = check_structure_hard(cot)
        missing = [k for k, v in checks.items() if not v]
        if missing:
            return False, None, "형식 누락: " + ", ".join(missing)
        return True, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f = StructureFilter()
    f.run_from_file(args.input, args.output)
