"""
Counterfactual Answer Probing judge.

Scores how strongly a CoT supports a target answer letter. The judge first
identifies which option the body of the reasoning actually supports, ignoring a
possibly spurious final "answer" line, then scores target support from 1 to 5.
"""

from __future__ import annotations

import json
import re
import time


def parse_geval_response(text):
    """G-Eval/JSON judge 응답에서 {"score":..,"supported_letter":..,..} 추출.
    (구 filters.judge.score_faithfulness 에서 inline — counterfactual judge 전용)"""
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    candidates = re.findall(r'\{[^{}]*"score"\s*:\s*[0-9]+[^{}]*\}', text)
    if candidates:
        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
                if isinstance(obj.get("score"), (int, float)) and 1 <= obj["score"] <= 5:
                    return obj
            except json.JSONDecodeError:
                continue

    m = re.findall(r'"score"\s*:\s*([1-5])', text)
    if m:
        return {"score": int(m[-1]), "reason": "recovered_from_broken_json"}

    m = re.findall(r'(\d)\s*점\s*(?:기준|에 해당|입니다|[:.])', text)
    if m:
        sc = int(m[-1])
        if 1 <= sc <= 5:
            return {"score": sc, "reason": "recovered_from_korean_text"}

    return None


_LETTERS = ["A", "B", "C", "D", "E"]

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _format_choices(choices) -> str:
    if not choices:
        return "(선지 정보 없음)"
    lines = []
    for i, choice in enumerate(list(choices)[: len(_LETTERS)]):
        lines.append(f"{_LETTERS[i]}. {choice}")
    return "\n".join(lines)


def _extract_internal_reasoning(cot: str) -> str:
    """Extract the internal reasoning stage from a full CoT if present.

    For Counterfactual evaluation, only the internal <think> block should be
    judged. If no <think> block exists, fall back to the full CoT text.
    """
    if not cot:
        return ""
    m = _THINK_BLOCK_RE.search(cot)
    if m:
        return m.group(1).strip()
    return cot.strip()


def _call_judge_with_model(client, model: str, prompt: str, temperature: float, max_tokens: int = 1024) -> str:
    """Call a judge model through an OpenAI-compatible chat client."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


_CF_RUBRIC = """너는 엄격한 의료시험 추론 감사관이다. 문제, 선지 전체, 평가 대상 선지(target),
그리고 추론(CoT)이 주어진다. 네 임무는 두 단계다.

[1단계 — 본문이 실제로 지지하는 선지 판정]
- CoT 본문의 *임상적 내용*(기전·해부·약리·감별 배제)만 보고, 이 추론이
  실제로 도달하는 선지 하나를 골라라. 이것을 supported_letter 라 한다.
- 맨 끝의 "정답: X" 선언 줄은 무시하라. 그 줄은 본문과 무관하게 붙어 있을 수 있다.
  반드시 본문 논증이 가리키는 선지로 판정하라.
  (예: 본문이 '좌측와위로 눕힌다'를 논증하면, 끝에 '정답: C'라 적혀 있어도
   좌측와위에 해당하는 선지가 supported_letter 다.)

[2단계 — target 지지 강도 채점]
- target 선지가 의학적으로 옳은 답인지 여부는 절대 판단하지 마라.
  오직 '추론이 target 선지의 내용을 결정적으로 지지하는가'만 본다.
- supported_letter ≠ target 이면: 이 추론은 target을 정당화하지 못한 것이다.
  prose가 아무리 유창하고 의학적으로 옳아도 score = 1.
- supported_letter == target 일 때만 아래 기준으로 채점:
    5 = target에 대한 결정적 임상 근거 + 모든 distractor 명시적 배제
    4 = target에 대한 강한 정당화 + 주요 distractor 배제
    3 = 합리적 정당화는 있으나 distractor 배제가 약하거나 일반적
    2 = 결론은 명시했으나 임상 근거가 빈약(재진술 수준)
    1 = target 정당화가 사실상 실패

[Input]
문제:
{question}

선지 전체:
{choices}

평가 대상 선지(target): {target_letter}

CoT:
{cot}

오직 아래 JSON만 출력하라. 다른 텍스트·코드블록·서두 금지:
{{"supported_letter": "<A~E>", "matches_target": <true|false>, "score": <1~5>, "reason": "<한 문장>"}}
"""


def score_convincingness(
    client,
    question: str,
    cot: str,
    target_idx: int,
    *,
    model: str = "solar-pro3",
    n_reps: int = 1,
    temperature: float = 0.3,
    sleep_on_error: float = 2.0,
    max_chars: int = 8000,
    choices=None,
) -> dict:
    """Score how strongly a single CoT supports target_idx as the answer."""
    if not (0 <= target_idx < len(_LETTERS)):
        return {"target": None, "runs": [], "mean": None, "raws": [], "errors": [f"invalid target_idx={target_idx}"]}

    target_letter = _LETTERS[target_idx]
    internal_cot = _extract_internal_reasoning((cot or "")[:max_chars])
    prompt = _CF_RUBRIC.format(
        target_letter=target_letter,
        question=(question or "").strip(),
        choices=_format_choices(choices),
        cot=internal_cot,
    )

    runs, raws, errors = [], [], []
    for _ in range(max(n_reps, 1)):
        text, parsed, last_err = None, None, None
        for _ in range(3):
            try:
                text = _call_judge_with_model(client, model, prompt, temperature)
                parsed = parse_geval_response(text)
                if parsed is not None:
                    break
                last_err = "empty_response" if not (text and text.strip()) else "parse_failed"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            if sleep_on_error:
                time.sleep(sleep_on_error)
        raws.append(text)
        if parsed is not None:
            runs.append(int(parsed["score"]))
            errors.append(None)
        else:
            errors.append(last_err)

    mean = sum(runs) / len(runs) if runs else None
    return {"target": target_letter, "runs": runs, "mean": mean, "raws": raws, "errors": errors}
