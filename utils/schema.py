from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Subset(str, Enum):
    DOCTOR = "doctor"
    NURSE = "nurse"
    PHARMACIST = "pharm"
    DENTIST = "dentist"


class FilterName(str, Enum):
    # PRE — 항상 적용 (Ablation 대상 아님)
    CORRECTNESS = "correctness"            # ①
    STRUCTURE = "structure"                # ②
    # Ablation 대상 필터 (Step 1 독립 측정)
    KPPL = "kppl"                          # ③
    STEP_COVERAGE = "step_coverage"        # ④
    ANSWER_CONSISTENCY = "answer_consistency"  # ⑤
    # 제안 층위 — Ablation 뒤 별도 단계로 적용 (C2 → C3 비교)
    THINK_FINAL_DIVERGENCE = "think_final_divergence"  # ⑥
    MATCHA = "matcha"                      # ⑦ (Answer-Conditioned Input Perturbation)
    COUNTERFACTUAL = "counterfactual"      # ⑧ (Inverse MATCHA — Answer Counterfactual)
    # TAIL — 항상 적용 (Ablation 대상 아님)
    JUDGE = "judge"                        # ⑨


PRE_FILTERS = [FilterName.CORRECTNESS, FilterName.STRUCTURE]
ABLATION_FILTERS = [
    FilterName.KPPL,
    FilterName.STEP_COVERAGE,
    FilterName.ANSWER_CONSISTENCY,
]
PROPOSAL_FILTERS = [FilterName.THINK_FINAL_DIVERGENCE, FilterName.MATCHA, FilterName.COUNTERFACTUAL]
TAIL_FILTERS = [FilterName.JUDGE]

FILTER_ORDER = PRE_FILTERS + ABLATION_FILTERS + PROPOSAL_FILTERS + TAIL_FILTERS


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
