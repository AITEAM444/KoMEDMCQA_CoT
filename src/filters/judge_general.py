"""C2 — 범용 LLM Judge 필터 (대조군).

연구계획 §6.2 / H3: "정답이 맞는 trace 안에서 *범용* 채점 AI 는 변별을 못 한다"를
입증하기 위한 대조군. C1 통과 trace 를 DC-CoT(2505.18759) 류 **범용 품질 루브릭**
(coherence / correctness-justification / clarity)으로 1~5 점화한다.

반사실 필터(C3)의 judge 와는 *개념상 별개*다:
  - C2 judge_general : "이 풀이가 (답 무관하게) 일반적으로 잘 쓰였나" — 범용 품질
  - C3 counterfactual : "답을 바꿨을 때 정당화가 무너지나" — 답 종속성(faithfulness)
API 인프라(OpenAI GPT-5 judge)는 precompute(반사실 judge)와 동일하게 재사용한다.

이 스크립트는 build_arms 의 unified jsonl 을 읽어 C1=True 행의 reasoning(think)을
채점하고 `signals.c2_score` 를 채워 같은 스키마로 다시 쓴다. arm 선택(상위 N=|C3|)은
`build_arms.py merge-c2` 가 이 점수를 읽어 수행한다(반사실의 precompute→run 분리와 동형).

판정엔 추가 호출이 없으므로(점수만 박아 둠) 임계값/Top-N 을 바꿔 가며 여러 번 돌려도
비용이 안 든다.

사용:
    export OPENAI_API_KEY=...    # GPT-5 judge
    python src/filters/judge_general.py \
        --unified data/unified.jsonl --output data/unified.jsonl \
        --workers 8 --judge-reps 1
    # 이어서: build_arms.py merge-c2 로 상위 |C3| → filters.C2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_LETTERS = ["A", "B", "C", "D", "E"]


# 추론(reasoning) 모델 — temperature 미지원 + max_completion_tokens 사용 + thinking 토큰 소비.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _make_judge() -> tuple[object, str]:
    """judge backend = OpenAI GPT-5 (OPENAI_API_KEY) — precompute._make_judge 와 동일 규약.

    모델은 OPENAI_JUDGE_MODEL 환경변수로 override 가능(default "gpt-5").
    """
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 가 필요합니다 (GPT-5 C2 judge용).")
    model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5")
    return OpenAI(api_key=key), model


# 범용 품질 루브릭 — 답의 옳고 그름이 아니라 *추론 글 자체의 품질* 만 본다.
# (C3 와 달리 답을 바꿔보지 않는다 = 일반 judge 가 변별 못 함을 보이려는 대조군이므로
#  의도적으로 표준적인 품질 축만 사용한다.)
_QUALITY_RUBRIC = """You are a STRICT grader of medical reasoning quality. Most exam explanations
have real flaws — do NOT be lenient and actively hunt for weaknesses. If you find yourself
giving 4–5 to most explanations, you are grading too softly. The official answer is
{gold_letter} and the CoT reaches it; judge ONLY the *reasoning prose quality*, NOT whether
a better answer exists. Score each of three axes independently on 1–5 with these anchors:

(1) justification — concrete, answer-SPECIFIC clinical evidence?
    5 = decisive mechanism/pharmacology/anatomy AND explicitly excludes the main distractors with evidence
    4 = solid specific mechanism, but ≥1 relevant distractor left unaddressed
    3 = correct but GENERIC evidence that could equally support other options; no exclusion
    2 = mostly restates the answer / shallow assertion
    1 = no real clinical evidence
(2) coherence — do steps connect with no leaps, contradictions, or post-hoc jumps?
    5 = every step strictly follows the previous; 3 = one unjustified leap; 1 = incoherent / hand-wavy
(3) clarity — organized, precise terminology, free of garbled/run-on/padding?
    5 = clean and precise; 3 = readable but verbose/loose; 1 = garbled or hard to follow

Rules: grade each axis strictly and INDEPENDENTLY. Do NOT default to the middle, and do
NOT copy one axis onto another. A fluent but generic explanation MUST score low on (1).
Reserve 5 for genuinely excellent reasoning.

[Question]
{question}

[Options]
{choices}

[Correct answer]
{gold_letter}

[CoT reasoning]
{cot}

Output ONLY this JSON, no other text (reason = the single biggest weakness):
{{"justification": <1-5>, "coherence": <1-5>, "clarity": <1-5>, "reason": "<one sentence on the main weakness>"}}
"""

_AXES = ("justification", "coherence", "clarity")


def _composite(obj: dict) -> float | None:
    """세 축(1~5) 평균을 c2_score 로. 세 축이 다 있으면 평균(연속값 → 동점↓ → 랭킹 변별↑),
    없으면 과거 'score' 단일값으로 폴백."""
    vals = [obj[a] for a in _AXES if isinstance(obj.get(a), (int, float)) and 1 <= obj[a] <= 5]
    if len(vals) == len(_AXES):
        return sum(vals) / len(vals)
    if isinstance(obj.get("score"), (int, float)) and 1 <= obj["score"] <= 5:
        return float(obj["score"])
    return None


def _parse_quality(text: str) -> dict | None:
    """judge 응답에서 세 축(justification/coherence/clarity) 추출 (JSON → 정규식 폴백)."""
    if not text or not text.strip():
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    for cand in reversed(re.findall(r'\{[^{}]*\}', t)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _composite(obj) is not None:
            return obj
    # 정규식 폴백 — 깨진 JSON 에서 축 점수만 회수
    obj = {}
    for a in _AXES:
        m = re.findall(rf'"{a}"\s*:\s*([1-5])', t)
        if m:
            obj[a] = int(m[-1])
    if len(obj) == len(_AXES):
        obj["reason"] = "recovered_from_broken_json"
        return obj
    m = re.findall(r'"score"\s*:\s*([1-5])', t)
    if m:
        return {"score": int(m[-1]), "reason": "recovered_from_broken_json"}
    return None


def _format_choices(choices) -> str:
    if not choices:
        return "(no option information)"
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(list(choices)[: len(_LETTERS)]))


def score_quality(
    client, model: str, question: str, cot: str, gold_idx: int, choices,
    *, n_reps: int = 1, temperature: float = 0.3, sleep_on_error: float = 2.0,
    max_chars: int = 8000,
) -> dict:
    """단일 CoT 의 범용 품질 점수(1~5, 세 축 평균). n_reps>1 이면 평균."""
    gold_letter = _LETTERS[gold_idx] if 0 <= gold_idx < len(_LETTERS) else "?"
    prompt = _QUALITY_RUBRIC.format(
        gold_letter=gold_letter,
        question=(question or "").strip(),
        choices=_format_choices(choices),
        cot=(cot or "")[:max_chars],
    )
    runs, last_err = [], None
    reasoning = _is_reasoning_model(model)
    for _ in range(max(n_reps, 1)):
        parsed = None
        for _ in range(3):
            try:
                if reasoning:
                    # GPT-5/o-series: temperature 미지원 + max_completion_tokens + thinking 여유
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=4000,
                    )
                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=512,
                    )
                parsed = _parse_quality(resp.choices[0].message.content)
                if parsed is not None:
                    break
                last_err = "parse_failed"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if sleep_on_error:
                time.sleep(sleep_on_error)
        if parsed is not None:
            comp = _composite(parsed)
            if comp is not None:
                runs.append(round(comp, 4))
    return {"mean": (sum(runs) / len(runs)) if runs else None, "runs": runs, "last_err": last_err}


def run(unified_path: Path, output_path: Path, *, workers: int, judge_reps: int,
        judge_temp: float, limit: int | None) -> None:
    rows = [json.loads(l) for l in open(unified_path, encoding="utf-8") if l.strip()]
    # C1 통과 + c2_score 미채점 행만 채점 대상 (C2 ⊆ C1)
    targets = [
        i for i, r in enumerate(rows)
        if r.get("filters", {}).get("C1") is True
        and r.get("signals", {}).get("c2_score") is None
    ]
    if limit is not None:
        targets = targets[:limit]

    judge_client, judge_model = _make_judge()
    print(f"[judge_general:C2] backend={judge_model} workers={workers} "
          f"C1통과미채점={len(targets)}건 / 전체 {len(rows)}건")

    # 체크포인트 — id 별 점수 append. 재실행 시 건너뜀.
    ckpt_path = output_path.with_suffix(".c2.ckpt.jsonl")
    scored: dict[str, float] = {}
    if ckpt_path.exists():
        for line in open(ckpt_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("score") is not None:
                    scored[str(rec["id"])] = rec["score"]
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[judge_general:C2] 체크포인트 {len(scored)}건 재사용 ← {ckpt_path}")

    todo = [i for i in targets if rows[i]["id"] not in scored]
    print(f"[judge_general:C2] 신규 채점 {len(todo)}건")

    lock = threading.Lock()
    done = {"n": len(scored)}
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    def _process(i: int) -> tuple[int, dict]:
        r = rows[i]
        try:
            res = score_quality(
                judge_client, judge_model, r["question"], r.get("think") or "",
                int(r["gold"]), r.get("choices") or [],
                n_reps=judge_reps, temperature=judge_temp,
            )
        except Exception as e:
            res = {"mean": None, "runs": [], "last_err": f"{type(e).__name__}: {e}"}
        return i, res

    def _log(i: int, res: dict) -> None:
        with lock:
            done["n"] += 1
            rid = rows[i]["id"]
            if res["mean"] is not None:
                scored[str(rid)] = res["mean"]
                ckpt_f.write(json.dumps({"id": rid, "score": res["mean"], "runs": res["runs"]},
                                        ensure_ascii=False) + "\n")
                ckpt_f.flush()
                print(f"[judge_general:C2] {done['n']}/{len(targets)} id={rid} c2_score={res['mean']}")
            else:
                print(f"[judge_general:C2] {done['n']}/{len(targets)} id={rid} 채점실패(재시도대상): {res['last_err']}")

    if workers <= 1:
        for i in todo:
            idx, res = _process(i)
            _log(idx, res)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_process, i) for i in todo]
            for fut in as_completed(futs):
                idx, res = fut.result()
                _log(idx, res)
    ckpt_f.close()

    # 점수를 signals.c2_score 에 적재 (C1 통과 + 채점된 행만)
    n_filled = 0
    for r in rows:
        s = scored.get(str(r["id"]))
        if s is not None:
            r.setdefault("signals", {})["c2_score"] = s
            n_filled += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_missing = len(targets) - sum(1 for i in targets if rows[i]["id"] in scored)
    print(f"[judge_general:C2] {len(rows)}건 → {output_path} (c2_score 채움 {n_filled}건)")
    if n_missing:
        print(f"[judge_general:C2] ⚠ 미채점 {n_missing}건 — 같은 명령 재실행 시 이어서 처리 "
              f"(체크포인트: {ckpt_path})")


def main():
    p = argparse.ArgumentParser(description="C2 범용 LLM Judge — unified 의 C1 통과 trace 품질 채점")
    p.add_argument("--unified", required=True, help="build_arms unified jsonl")
    p.add_argument("--output", required=True, help="signals.c2_score 채운 unified jsonl (in-place 가능)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--judge-reps", type=int, default=1, help="채점 반복(평균)")
    p.add_argument("--judge-temp", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None, help="파일럿용")
    args = p.parse_args()
    run(Path(args.unified), Path(args.output), workers=args.workers,
        judge_reps=args.judge_reps, judge_temp=args.judge_temp, limit=args.limit)


if __name__ == "__main__":
    main()
