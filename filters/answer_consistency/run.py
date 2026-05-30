"""
Answer Consistency Filter  (⑤ Ablation — 한국어 특화)

[Step 4] 결론 단락이 선언된 정답 선택지를 명시적으로 언급하는지 규칙 기반으로 검사.
추론 본문과 최종 정답 라인 사이의 내부 정합성(CoT Mismatch)을 탐지.

── 방식 ────────────────────────────────────────────────────────────────────────
규칙 기반 Answer Consistency (NLI 모델 없음):
    1. 양성(score ≥ 0.8): Step 4에 declared_answer 레터 또는 선택지 텍스트가
       결론 문맥에서 명시됨 → 통과
    2. 음성(score = 0.1): Step 4가 다른 선택지 레터를 결론으로 명시 → 탈락 (CoT 불일치)
    3. 불확실(score = 0.6): 명시적 언급 없음 → 기본 통과 (보수적 기본값)

── 배경 ────────────────────────────────────────────────────────────────────────
기존 NLI 방식(klue-roberta-base-nli)이 n=273 데이터에서 75.1% 탈락을 유발,
분석 결과 모델이 의료 도메인 한국어 추론의 entailment를 과소평가하는 것이 원인.
규칙 기반 전환 후 약 85% 통과, 나머지 15%는 선택지 텍스트 추출 실패로 기본 통과.

── 실행 ────────────────────────────────────────────────────────────────────────
단독 실행  python -m filters.answer_consistency.run --input <in.json> --output <out.json>
파이프라인  python scripts/run_pipeline.py --input <in.json> --output_dir <dir>

담당: B
참고: PROF (arXiv:2509.03403), Critique of Impure Reason (arXiv:2412.15748)
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

import yaml

from filters.base import BaseFilter
from utils.schema import CoTSample


STEP4_RE = re.compile(r"\[Step 4\](.*?)(?=\*{0,2}정답\*{0,2}\s*[：:]|\Z)", re.DOTALL)
STEP3_RE = re.compile(r"\[Step 3\](.*?)(?=\[Step 4\]|\Z)", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"\*{0,2}정답\*{0,2}\s*[：:]\s*([ABCDE])\*{0,2}[\s.]*$", re.MULTILINE)
_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pipeline_config.yaml"

_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# 결론 문맥 도입 어구
_CONCLUSION_PREFIX = r"(?:따라서|그러므로|결론(?:적으로)?|최종(?:적으로)?|결국|요약하면|정리하면)"


def _extract_choice_from_step3(cot: str, letter: str) -> str:
    """
    choices 필드가 비어있을 때 Step 3 텍스트에서 해당 선택지(A~E)의 핵심 명칭을 추출.

    Step 3는 보통 다음 형태로 작성된다:
      A. lysozyme – 선세포에서 분비되는 ...
      B. 수마트립탄(sumatriptan) - 트립탄 계열 ...
      C. α-amylase: 소화 효소 ...

    설명(대시/콜론 뒤)을 제외하고 선택지 명칭 부분만 반환한다.
    """
    step3_m = STEP3_RE.search(cot)
    if not step3_m:
        return ""
    step3 = step3_m.group(1)

    pattern = (
        rf'(?:^|\n)[ \t]*(?:[-*•]\s*)?{letter}[\.\):]\s*'
        rf'(.+?)(?=\n[ \t]*(?:[-*•]\s*)?[ABCDE][\.\):]|\Z)'
    )
    m = re.search(pattern, step3, re.DOTALL)
    if not m:
        return ""
    full_text = m.group(1).strip()

    first_line = full_text.split('\n')[0].strip()

    # 대시(–/—/-) 또는 콜론(:) 앞의 명칭 부분만 추출
    name_part = re.split(r'\s+[-–—]\s+|\s*:\s+', first_line, maxsplit=1)[0].strip()
    return name_part[:40] if name_part else first_line[:40]


def _rule_based_score(
    step4_text: str, declared_answer: str, choice_text: str
) -> tuple[float, str | None]:
    """
    Step 4 텍스트가 declared_answer 와 일치하는지 규칙으로 판정.

    Returns:
        score  : 0.1(불일치) / 0.6(불확실, 기본 통과) / 0.8~1.0(일치)
        reason : 탈락 시 이유, 통과 시 None
    """
    letter = declared_answer

    # ── 양성: declared answer 레터 명시 ────────────────────────────────────────
    pos_letter_pats = [
        rf'{_CONCLUSION_PREFIX}[^\n]*?{letter}',            # 따라서 ... B
        rf'정답[은이으로:]\s*{letter}',                       # 정답은 B
        rf'{letter}[이가은는을]?\s*(?:정답|맞|옳|적절|이다|입니다)',  # B가 정답, B이다
        rf'선택지\s*{letter}',                               # 선택지 B
        rf'답[은이:]\s*{letter}',                            # 답은 B
        rf'{letter}(?:번)?(?:이다|입니다)',                   # B이다, B번입니다
    ]
    if any(re.search(p, step4_text) for p in pos_letter_pats):
        return 1.0, None

    # ── 양성: 선택지 텍스트 명시 ────────────────────────────────────────────────
    if choice_text and len(choice_text) >= 3:
        # 괄호 내용 제거 후 핵심 명칭
        key = re.sub(r'\(.*?\)', '', choice_text).strip()

        if key and key in step4_text:
            return 0.9, None

        # "(ABBR)" 형태 약어 추출 후 매칭  예: "activated partial … (aPTT)"
        abbr_m = re.search(r'\(([A-Za-z]{2,10})\)', choice_text)
        if abbr_m and abbr_m.group(1) in step4_text:
            return 0.85, None

        # 가장 적절한/맞는 + 선택지 명칭 앞 15자
        key_short = key[:15]
        if key_short:
            conclusion_text_pats = [
                rf'(?:가장\s+)?(?:적절한|좋은|옳은|맞는)\s+\S+[^.\n]*?{re.escape(key_short)}',
                rf'정답[은이]\s+{re.escape(key_short)}',
                rf'{_CONCLUSION_PREFIX}[^\n]*?{re.escape(key_short)}',
            ]
            if any(re.search(p, step4_text) for p in conclusion_text_pats):
                return 0.8, None

    # ── 음성: 다른 선택지 레터가 결론 맥락에 명시 → CoT 불일치 ──────────────────
    # lookbehind (?<![A-Za-z0-9]) 로 영문 약어 내부(IgE, NSAIDs, CD3 등)를 제외한다.
    other_letters = [l for l in "ABCDE" if l != letter]
    og = "|".join(other_letters)
    # 단독 선택지 레터: 앞이 영숫자가 아니고 뒤가 한국어 조사·종결어미
    _sl = rf'(?<![A-Za-z0-9])(?:{og})(?=[이가은는을번\s]|이다|입니다|이므로)'
    neg_pats = [
        rf'{_CONCLUSION_PREFIX}[^\n]*?{_sl}',
        rf'정답[은이으로:]\s*(?<![A-Za-z0-9])(?:{og})',
        rf'(?<![A-Za-z0-9])(?:{og})(?:번)?(?:이다|입니다)',
        rf'(?<![A-Za-z0-9])(?:{og})[이가][^。\n]*?(?:정답|맞|옳|적절)',
    ]
    if any(re.search(p, step4_text) for p in neg_pats):
        return 0.1, f"Step 4가 정답 {letter} 아닌 다른 선택지를 결론으로 명시 (CoT 불일치)"

    # ── 불확실: 명시적 언급 없음 → 보수적 기본 통과 ───────────────────────────
    return 0.6, None


def _load_ac_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["ablation_filters"]["answer_consistency"]


class AnswerConsistencyFilter(BaseFilter):
    name = "answer_consistency"

    def __init__(self):
        ac_cfg = _load_ac_config()
        self.threshold: float = ac_cfg.get("entailment_threshold", 0.5)

    def filter_sample(
        self, sample: CoTSample
    ) -> tuple[bool, float | None, str | None]:
        answer_match = FINAL_ANSWER_RE.search(sample.cot)
        if not answer_match:
            return False, None, "정답 라인 파싱 실패"
        declared_answer = answer_match.group(1)

        step4_match = STEP4_RE.search(sample.cot)
        if not step4_match:
            return False, None, "Step 4 파싱 실패"
        step4_text = step4_match.group(1).strip()
        if not step4_text:
            return False, None, "Step 4 내용 없음"

        idx = _LETTER_TO_IDX[declared_answer]
        if sample.choices and len(sample.choices) > idx:
            choice_text = sample.choices[idx]
        else:
            choice_text = _extract_choice_from_step3(sample.cot, declared_answer)

        score, reason = _rule_based_score(step4_text, declared_answer, choice_text)
        passed = score >= self.threshold
        return passed, score, reason


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f = AnswerConsistencyFilter()
    f.run_from_file(args.input, args.output)
