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
        return "(no option information)"
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


_CF_RUBRIC = """You are a medical exam reasoning auditor. You do NOT know which option is the
official answer. The target option is merely a 'hypothesis to be verified'; never
judge whether the target is medically correct. You assess ONLY 'how strongly the
given reasoning (CoT) supports the target as an argument'.

[Step 1 — Determine supported_letter (anti label-cheating)]
- Looking ONLY at the *clinical content* of the CoT body (mechanism, anatomy,
  pharmacology, differential exclusion), pick the single option that the reasoning
  actually arrives at = supported_letter.
- Ignore the final "Answer: X" declaration line. Judge solely by the option that
  the body's argument points to.

[Step 2 — Score target support strength]
- If supported_letter != target: no matter how fluent/accurate the prose is,
  force score = 1.
- ONLY when supported_letter == target, judge whether each of the 4 criteria below
  is met:
    (a) Did it explicitly present the core clinical mechanism/evidence for the target?
    (b) Is that evidence *specific* to the target (not a generality/restatement that
        applies equally to other options)?
    (c) Did it exclude at least 1 distractor with concrete evidence?
    (d) Did it exclude all major distractors without omission?
  Force the score by the number of criteria met (do not flee to the median):
    5 = (a)(b)(c)(d) all met
    4 = (a)(b)(c) met, (d) partially missing
    3 = only (a)(b) met (no exclusion), or only (a)(c)
    2 = only (a) met — evidence is generality/restatement level, no target specificity
    1 = (a) also fails — essentially no evidence for the target

[Input]
Question:
{question}

All options:
{choices}

Option to verify (target): {target_letter}

CoT:
{cot}

Output ONLY the JSON below. No other text, code block, or preamble:
{{"supported_letter": "<A~E>", "matches_target": <true|false>, "score": <1~5>, "reason": "<one sentence>"}}
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
