"""
PRISM — Perturbation-Robustness Inference Stability Monitor  (⑦ Proposal)

Jiang et al., "Robust Answers, Fragile Logic" (2025) 의 MATCHA 방법론을 기반으로 한
Answer-Conditioned Perturbation 스트레스 테스트 필터.

핵심 원리 (Decoupling Hypothesis):
  LLM의 정답과 추론 경로는 강하게 인과로 묶여 있지 않고 '느슨하게 상관'될 뿐이다.
  모델이 답을 먼저 직관적으로 선택하고, 그 답을 정당화하는 추론을 사후에 짜는 경우
  ('Right for the Wrong Reasons'), 입력에 미세한 교란을 줬을 때 추론 체인만 무너진다.

작동 방식:
  1. 정답을 먼저 고정 (answer-conditioned prompt: "정답은 X입니다")
  2. 문제 입력에 의미 보존 교란 3종 적용
  3. 교란 후 새로운 CoT 재생성 (probe 모델 or heuristic)
  4. CoT가 무너졌는지 판정 → stability_score < threshold 이면 fragile → 탈락

두 가지 운영 모드:
  heuristic (기본값)
    모델 호출 없이 원본 CoT 내부 신호로 fragility를 추정.
    세 가지 신호를 0~1 스케일로 통합:
      S1. Step4 본문이 정답 레이블을 명시적으로 언급하는가 (사후정당화 탐지)
      S2. <think>–Step4 hedge 밀도 격차 (F6 검출 이후에도 남은 잠재 fragility)
      S3. Step3 내부 역전·수정 마커 밀도 (선택지 검토 단계 불안정성)

  api_probe
    OpenAI-compatible 엔드포인트에 교란 프롬프트를 보내 CoT를 재생성하고
    (a) 예측 정답 일치 여부, (b) hedge 밀도 급등 여부로 안정성을 투표.
    로컬 R1-distill(vLLM 서빙) 또는 외부 API 모두 지원.
    gradient 접근 불필요 — 완전 black-box.

교란 전략 (의미 보존, black-box):
  distractor_inject   문제 줄기에 무관한 의학적 문장 1개 삽입
  choice_order_swap   오답 선택지 2개의 순서 교환 (정답 위치 고정)
  stem_marker         문제 끝에 중립 재독 마커 추가

파이프라인 위치:
  C2 (기존): PRE → Ablation → Think-Final-Divergence → Judge
  C3 (제안): PRE → Ablation → Think-Final-Divergence → PRISM → Judge

  C2 vs C3 다운스트림 Student 성능으로만 효과를 보고하고,
  Judge Score로는 주장하지 않는다 (순환 평가 방지).

주의:
  api_probe 모드는 white-box(gradient)가 아닌 black-box 교란이므로
  완전 MATCHA 재현은 아니다. 논문 §4 Figure 1의 token-level adversarial
  perturbation 대비 경량 근사 버전임을 명시할 것.

confound 통제 (§6.3):
  - 교란-reject 샘플의 문제 난이도 편향 확인 (길이 프록시)
  - "답 바뀐 군 vs 유지 군"의 hedge 증가폭 비교 (진짜 fragility vs noise)
  - 의미보존 judge로 의미 변경된 교란 제거

단독 실행   python -m filters.prism.run --input <in.json> --output <out.json>
파이프라인  python scripts/run_pipeline.py --input <in.json> --output_dir <dir>
"""

from __future__ import annotations

import re
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import os

import yaml

from filters.base import BaseFilter
from utils.schema import CoTSample


# ── 상수 ────────────────────────────────────────────────────────────────────
CHOICE_LABELS = ["A", "B", "C", "D", "E"]

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"

# F6(think_final_divergence)과 동일한 한국어 hedge 표지 목록 재사용
_KO_HEDGE_PATTERNS: list[str] = [
    "하지만", "그러나", "다시 생각", "확실하지", "정정",
    "아닌 것 같", "다른 것 같", "혹시", "잠깐", "재고",
    "다시 보면", "다시 살펴", "재확인", "아닐 수도",
    "맞는지", "맞지 않", "수정", "틀렸", "잘못",
    "오히려", "그렇지 않", "아닌가",
]

# S3 신호용 — Step3 내 역전/수정 마커 (hedge보다 더 강한 불안정 신호)
_BACKTRACK_PATTERNS: list[str] = [
    "아닌 것 같", "다시 생각", "오히려", "정정", "수정", "틀렸",
    "재고", "잘못", "다시 보면", "재확인",
]

# 교란(distractor_inject)용 의학적으로 그럴싸하지만 무관한 문장 풀
_DISTRACTOR_POOL: list[str] = [
    "단, 이 환자는 최근 6개월간 특별한 약물 복용 이력이 없었다.",
    "해당 검사는 공복 상태에서 시행되었으며 결과는 당일 보고되었다.",
    "환자의 혈압은 정상 범위에 있었으며 체온은 36.8°C였다.",
    "보호자는 증상 발현 시점을 정확히 기억하지 못했다고 진술하였다.",
    "입원 당시 시행된 12유도 심전도 검사는 정상 동리듬을 보였다.",
    "환자는 알레르기 병력이 없다고 진술하였고 가족력도 특이사항이 없었다.",
    "이전 수술 이력은 없으며 직업은 사무직 종사자이다.",
    "의무기록에 따르면 증상 발현 후 48시간이 경과한 시점에 내원하였다.",
    "해당 병원은 지역 거점 종합병원으로 관련 전문의가 상주하고 있다.",
    "시행된 혈액검사에서 CRP 및 ESR은 정상 수치를 보였다.",
]

# ── 정규식 ───────────────────────────────────────────────────────────────────
_ANSWER_RE = re.compile(
    r"(?:정답|답)\s*[：:]\s*([A-Ea-e])", re.IGNORECASE
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STEP4_RE = re.compile(
    r"\[Step\s*4\](.*?)(?=\*{0,2}정답\*{0,2}\s*[：:]|\Z)", re.DOTALL
)
_STEP3_RE = re.compile(r"\[Step\s*3\](.*?)\[Step\s*4\]", re.DOTALL)


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def _count_patterns(text: str, patterns: list[str]) -> int:
    return sum(text.count(p) for p in patterns)


def _hedge_rate(text: str, patterns: list[str] = _KO_HEDGE_PATTERNS) -> float:
    """hedge 표지 수 / 단어 수 × 100."""
    words = text.split()
    return _count_patterns(text, patterns) / max(len(words), 1) * 100


def _extract_predicted_answer(cot: str) -> str | None:
    """CoT 텍스트에서 최종 정답 레이블(A-E)을 추출."""
    m = _ANSWER_RE.search(cot)
    return m.group(1).upper() if m else None


def _format_choices(choices: list[str]) -> str:
    labels = CHOICE_LABELS[: len(choices)]
    return "\n".join(f"{lbl}. {ch}" for lbl, ch in zip(labels, choices))


# ── 교란 전략 ─────────────────────────────────────────────────────────────────

def _perturb_distractor_inject(
    question: str, choices: list[str], seed: int = 42
) -> tuple[str, list[str]]:
    """
    문제 줄기 끝에 의학적으로 그럴싸하지만 무관한 문장을 1개 삽입한다.
    질문의 핵심 의미는 변경되지 않으나 reasoning이 해당 distractor에
    이끌리거나 무시하는 방식이 달라진다 → fragile한 CoT는 혼동을 보임.
    """
    rng = random.Random(seed)
    distractor = rng.choice(_DISTRACTOR_POOL)
    perturbed_q = question.rstrip() + " " + distractor
    return perturbed_q, choices


def _perturb_choice_order_swap(
    question: str, choices: list[str], correct_idx: int, seed: int = 42
) -> tuple[str, list[str]]:
    """
    오답 선택지 2개의 순서(위치)를 교환한다.
    정답 선택지의 내용과 위치는 고정되므로 의미가 보존된다.
    표면적 레이블 연결이 달라지므로 '특정 레이블에 고착된' 취약한 추론이
    다른 레이블을 정답으로 선언하거나 혼동 신호를 내놓는지 확인한다.
    """
    rng = random.Random(seed)
    wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
    if len(wrong_indices) < 2:
        return question, choices

    i, j = rng.sample(wrong_indices, 2)
    new_choices = choices[:]
    new_choices[i], new_choices[j] = new_choices[j], new_choices[i]
    return question, new_choices


def _perturb_stem_marker(
    question: str, choices: list[str]
) -> tuple[str, list[str]]:
    """
    문제 끝에 '가장 적절한 것을 고르시오'라는 중립 마커를 추가한다.
    질문의 의미는 완전히 동일하나 문장 길이·구조가 변하여 토큰 수준의
    민감도를 간접 측정한다. 가장 경량한 교란.
    """
    marker = " (아래 선택지 중 가장 적절한 것을 고르시오.)"
    if marker in question:
        return question, choices
    return question.rstrip() + marker, choices


# ── Answer-conditioned 프롬프트 ─────────────────────────────────────────────

def _answer_conditioned_prompt(
    question: str, choices: list[str], answer_letter: str
) -> str:
    """
    MATCHA의 핵심: 정답을 먼저 '힌트'로 노출하고,
    그 상태에서 교란된 입력에 대해 추론만 생성하도록 유도한다.
    정답이 고정됐는데도 추론이 무너지면 → '사후 정당화' 패턴 → fragile.
    """
    return (
        f"[PRISM 힌트: 이 문제의 정답은 {answer_letter}입니다. "
        "아래 추론 단계를 통해 왜 그 선택지가 옳은지 논리적으로 설명하시오.]\n\n"
        f"문제: {question}\n\n"
        f"선택지:\n{_format_choices(choices)}\n\n"
        "[Step 1] 환자/문제 상황 정리\n"
        "[Step 2] 관련 의학적 원리\n"
        "[Step 3] 선택지별 검토\n"
        "[Step 4] 결론 도출\n"
        "정답:"
    )


# ── API probe (black-box) ────────────────────────────────────────────────────

def _call_probe_api(
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    api_key: str = "EMPTY",
    timeout: int = 60,
) -> str | None:
    """
    OpenAI-compatible 엔드포인트에 answer-conditioned 교란 프롬프트를 전송하고
    CoT 텍스트를 반환. DeepSeek-R1 API의 reasoning_content 필드를 처리함.
    네트워크 오류 시 None 반환 → heuristic fallback.
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
            timeout=timeout,
        )
        msg = resp.choices[0].message
        # DeepSeek-R1 API: reasoning은 reasoning_content, 답은 content로 분리됨
        reasoning = getattr(msg, "reasoning_content", None) or ""
        content = msg.content or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content
    except Exception:
        return None


# ── 교란 결과 데이터클래스 ────────────────────────────────────────────────────

@dataclass
class _ProbedResult:
    strategy: str
    predicted_answer: str | None   # 재생성 CoT에서 추출한 정답 레이블
    hedge_rate: float               # 재생성 CoT의 hedge 밀도
    coherent: bool                  # 안정적(True) vs 붕괴(False)
    cot_head: str                   # 디버그용 앞 200자


# ── heuristic 안정성 추정 ────────────────────────────────────────────────────

def _heuristic_stability(sample: CoTSample) -> tuple[float, str]:
    """
    모델 호출 없이 원본 CoT 내부 세 가지 신호로 fragility를 추정.

    S1 (Step4 명시성): Step4 본문이 정답 레이블을 명시적으로 언급하는가.
        없으면 사후정당화 가능성 높음. → 0.0 / 있으면 → 1.0
    S2 (hedge 격차): <think>–Step4 hedge 밀도 격차.
        gap ≥ 4.0 → 0.0, gap ≥ 2.0 → 0.4, else → 1.0
        (F6에서 이미 걸러진 심각한 케이스 이후 잔류 신호 측정)
    S3 (Step3 역전): Step3 내 backtrack 마커 수.
        ≥ 3 → 0.0, 1~2 → 0.5, 0 → 1.0

    stability_score = mean(S1, S2, S3) ∈ [0, 1]
    """
    answer_letter = (
        CHOICE_LABELS[sample.answer]
        if sample.answer is not None and sample.answer < len(CHOICE_LABELS)
        else None
    )

    # --- S1: Step4 정답 레이블 명시 ---
    step4_m = _STEP4_RE.search(sample.cot)
    step4_text = step4_m.group(1).strip() if step4_m else ""

    if answer_letter and step4_text and answer_letter in step4_text:
        s1 = 1.0
        s1_tag = f"s4_mention=Y({answer_letter})"
    elif step4_text:
        s1 = 0.0
        s1_tag = f"s4_mention=N({answer_letter})"
    else:
        # Step4 자체가 없으면 F2에서 걸러져야 했으나 방어적으로 중립 처리
        s1 = 0.5
        s1_tag = "s4_missing"

    # --- S2: <think>–Step4 hedge 격차 ---
    think_m = _THINK_RE.search(sample.cot)
    think_text = think_m.group(1) if think_m else ""

    think_hr = _hedge_rate(think_text) if think_text else 0.0
    step4_hr = _hedge_rate(step4_text) if step4_text else 0.0
    gap = think_hr - step4_hr

    if gap >= 4.0:
        s2 = 0.0
    elif gap >= 2.0:
        s2 = 0.4
    else:
        s2 = 1.0
    s2_tag = f"think_gap={gap:.2f}"

    # --- S3: Step3 내 역전·수정 마커 ---
    step3_m = _STEP3_RE.search(sample.cot)
    step3_text = step3_m.group(1) if step3_m else ""
    backtrack_n = _count_patterns(step3_text, _BACKTRACK_PATTERNS)

    if backtrack_n >= 3:
        s3 = 0.0
    elif backtrack_n >= 1:
        s3 = 0.5
    else:
        s3 = 1.0
    s3_tag = f"s3_backtrack={backtrack_n}"

    score = (s1 + s2 + s3) / 3.0
    summary = f"{s1_tag}|{s2_tag}|{s3_tag}"
    return round(score, 4), summary


# ── API probe 안정성 계산 ────────────────────────────────────────────────────

def _api_probe_stability(
    sample: CoTSample,
    answer_letter: str,
    probe_base_url: str,
    probe_model: str,
    temperature: float,
    strategies: list[str],
    n_perturb: int,
    hedge_spike_threshold: float,
    api_key: str = "EMPTY",
) -> tuple[float, str]:
    """
    교란 프롬프트를 probe 모델에 보내 CoT를 재생성하고 안정성을 투표.
    API 오류 등으로 결과가 없으면 heuristic fallback.
    """
    perturbations: list[tuple[str, str, list[str]]] = []

    if "distractor_inject" in strategies:
        pq, pc = _perturb_distractor_inject(sample.question, sample.choices)
        perturbations.append(("distractor_inject", pq, pc))

    if "choice_order_swap" in strategies and sample.answer is not None:
        pq, pc = _perturb_choice_order_swap(
            sample.question, sample.choices, sample.answer
        )
        perturbations.append(("choice_order_swap", pq, pc))

    if "stem_marker" in strategies:
        pq, pc = _perturb_stem_marker(sample.question, sample.choices)
        perturbations.append(("stem_marker", pq, pc))

    results: list[_ProbedResult] = []
    for strat, pq, pc in perturbations[:n_perturb]:
        prompt = _answer_conditioned_prompt(pq, pc, answer_letter)
        cot_text = _call_probe_api(
            prompt, probe_base_url, probe_model, temperature, api_key
        )
        if cot_text is None:
            continue

        pred = _extract_predicted_answer(cot_text)
        hr = _hedge_rate(cot_text)
        # 안정: 정답 일치 AND hedge 급등 없음
        coherent = (pred == answer_letter) and (hr < hedge_spike_threshold)
        results.append(
            _ProbedResult(
                strategy=strat,
                predicted_answer=pred,
                hedge_rate=hr,
                coherent=coherent,
                cot_head=cot_text[:200],
            )
        )

    if not results:
        # API 완전 실패 → heuristic fallback
        return _heuristic_stability(sample)

    n_coherent = sum(1 for r in results if r.coherent)
    score = n_coherent / len(results)
    details = "|".join(
        f"{r.strategy}:{'OK' if r.coherent else 'FAIL'}"
        f"(pred={r.predicted_answer},hr={r.hedge_rate:.1f})"
        for r in results
    )
    return round(score, 4), details


# ── 메인 필터 클래스 ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["proposal_filters"]["prism"]


class PRISMFilter(BaseFilter):
    """
    PRISM — Perturbation-Robustness Inference Stability Monitor

    Answer-Conditioned Probing으로 '정답은 맞히지만 추론이 취약한(brittle)' 샘플을 탐지.

    stability_score ∈ [0, 1]:
        1.0 — 교란에 완전히 안정적 (robust CoT)
        0.0 — 교란 즉시 붕괴 (fragile CoT)
    threshold 미만 → 탈락 (학습셋에서 제외)

    heuristic 모드 (기본):
        원본 CoT 내부 3가지 신호 (Step4 명시성, hedge 격차, Step3 역전 밀도)
        모델 호출 없음 — 서브샘플 파일럿 및 비용 제로 실험에 적합.

    api_probe 모드:
        교란 프롬프트 → OpenAI-compatible probe 모델 → 재생성 CoT 안정성 투표.
        vLLM 서빙 로컬 R1-distill 또는 외부 API와 연동.
        효과 확인 후 전체 스케일로 확장하는 것을 권장.
    """

    name = "prism"

    def __init__(self) -> None:
        cfg = _load_config()

        self.mode: Literal["heuristic", "api_probe"] = cfg.get("mode", "heuristic")
        self.stability_threshold: float = cfg.get("stability_threshold", 0.5)
        self.n_perturb: int = cfg.get("n_perturb", 3)
        self.temperature: float = cfg.get("temperature", 0.4)

        # api_probe 전용
        self.probe_base_url: str = cfg.get(
            "probe_base_url", "https://api.deepseek.com"
        )
        self.probe_model: str = cfg.get(
            "probe_model", "deepseek-reasoner"
        )
        self.api_key: str = cfg.get(
            "api_key", os.getenv("DEEPSEEK_API_KEY", "EMPTY")
        )
        self.hedge_spike_threshold: float = cfg.get("hedge_spike_threshold", 5.0)
        self.perturbation_strategies: list[str] = cfg.get(
            "perturbation_strategies",
            ["distractor_inject", "choice_order_swap", "stem_marker"],
        )

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:

        # F1에서 이미 탈락한 샘플(is_correct=False)은 PRISM 대상 아님
        # (PRISM은 "정답은 맞는데 추론이 취약한" 케이스를 탐지한다)
        if sample.is_correct is False:
            return True, None, None

        answer_letter = (
            CHOICE_LABELS[sample.answer]
            if sample.answer is not None and sample.answer < len(CHOICE_LABELS)
            else None
        )

        if self.mode == "api_probe" and answer_letter:
            score, summary = _api_probe_stability(
                sample=sample,
                answer_letter=answer_letter,
                probe_base_url=self.probe_base_url,
                probe_model=self.probe_model,
                temperature=self.temperature,
                strategies=self.perturbation_strategies,
                n_perturb=self.n_perturb,
                hedge_spike_threshold=self.hedge_spike_threshold,
                api_key=self.api_key,
            )
        else:
            score, summary = _heuristic_stability(sample)

        # confound 통제를 위한 메타데이터 기록
        sample.metadata.update(
            {
                "prism_stability_score": score,
                "prism_signal_summary": summary,
                "prism_mode": self.mode,
                # 난이도 편향 확인용 프록시 (6.3-1)
                "prism_cot_length": len(sample.cot.split()),
            }
        )

        if score < self.stability_threshold:
            return (
                False,
                score,
                f"fragile(score={score:.3f}<{self.stability_threshold}): {summary}",
            )

        return True, score, None


# ── 독립 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRISM — Perturbation-Robustness Inference Stability Monitor"
    )
    parser.add_argument("--input", type=Path, required=True, help="입력 JSON")
    parser.add_argument("--output", type=Path, required=True, help="출력 JSON")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "api_probe"],
        default=None,
        help="운영 모드 (미지정 시 config 값 사용)",
    )
    args = parser.parse_args()

    f = PRISMFilter()
    if args.mode:
        f.mode = args.mode
        print(f"[PRISM] mode override → {f.mode}")

    f.run_from_file(args.input, args.output)
