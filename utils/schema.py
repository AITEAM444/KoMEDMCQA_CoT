from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Subset(str, Enum):
    DOCTOR = "doctor"
    NURSE = "nurse"
    PHARMACIST = "pharm"
    DENTIST = "dentist"


class FilterName(str, Enum):
    # C1 — DeepSeek-R1 표준 baseline 정제 (①~④, 고정값)
    CORRECTNESS = "correctness"        # ① 정답 게이트
    READABILITY = "readability"        # ② 혼합언어/가독성 (한글비율 필터 비채택)
    STRUCTURE = "structure"            # ③ 형식/파싱 게이트
    LENGTH = "length"                  # ④ 길이 필터
    # C2 — 범용 LLM Judge (대조군)
    JUDGE_GENERAL = "judge_general"
    # C3 — 반사실 충실성 필터 (본 연구 기여, Inverse-MATCHA)
    COUNTERFACTUAL = "counterfactual"


# C1 표준 baseline = ①~④ (R1 2501.12948 정제 절차 고정값)
C1_FILTERS = [
    FilterName.CORRECTNESS,
    FilterName.READABILITY,
    FilterName.STRUCTURE,
    FilterName.LENGTH,
]


@dataclass
class FilterRecord:
    filter_name: str
    passed: bool
    score: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class JudgeScore:
    step_coherence: float         # 단계 간 논리적 연결 (reference-guided)
    korean_fluency: float         # 번역체 없는 자연스러운 한국어 의학 표현
    step_coverage: float          # Step 1~4 각 역할 실질 수행 여부

    @property
    def minimum(self) -> float:
        return min(self.step_coherence, self.korean_fluency, self.step_coverage)


@dataclass
class CoTSample:
    id: str
    subset: str
    question: str
    choices: list[str]
    answer: int                        # 0-indexed 정답 인덱스
    teacher_model: str                 # "deepseek-r1" (확정), contamination check 용 "gpt-5"
    cot: str                           # 생성된 CoT 텍스트
    predicted_answer: Optional[int] = None
    filter_history: list[FilterRecord] = field(default_factory=list)
    judge_score: Optional[JudgeScore] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_correct(self) -> Optional[bool]:
        if self.predicted_answer is None:
            return None
        return self.predicted_answer == self.answer

    @property
    def passed_filters(self) -> list[str]:
        return [r.filter_name for r in self.filter_history if r.passed]

    @property
    def failed_filters(self) -> list[str]:
        return [r.filter_name for r in self.filter_history if not r.passed]

    def add_filter_result(
        self,
        filter_name: str,
        passed: bool,
        score: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.filter_history.append(
            FilterRecord(
                filter_name=filter_name,
                passed=passed,
                score=score,
                reason=reason,
            )
        )
