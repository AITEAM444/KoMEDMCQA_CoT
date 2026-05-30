import json
from pathlib import Path
from typing import Union
from utils.schema import CoTSample, FilterRecord, JudgeScore


_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _dict_to_sample(d: dict) -> CoTSample:
    filter_history = [
        FilterRecord(**r) for r in d.get("filter_history", [])
    ]
    judge_raw = d.get("judge_score")
    if judge_raw:
        judge_score = JudgeScore(**judge_raw)
    elif "scores" in d:
        scores = d["scores"]
        judge_score = JudgeScore(
            step_coherence=scores["step_coherence"]["mean"],
            korean_fluency=scores["korean_fluency"]["mean"],
            step_coverage=scores["step_coverage"]["mean"],
        )
    else:
        judge_score = None

    sample_id = d.get("id") or str(d.get("global_idx", ""))
    if not sample_id:
        raise KeyError("id 또는 global_idx 중 하나는 반드시 있어야 합니다")

    # answer: int(processed) 또는 letter string(raw) 모두 허용
    answer_raw = d.get("answer")
    if answer_raw is None:
        correct = d.get("correct", "")
        answer_raw = _LETTER_TO_IDX.get(correct)
        if answer_raw is None:
            raise KeyError("answer 또는 correct 필드가 필요합니다")
    elif isinstance(answer_raw, str):
        answer_raw = _LETTER_TO_IDX.get(answer_raw, answer_raw)

    # cot: 직접 필드 또는 raw 필드(reasoning_content + final_content)에서 재구성
    cot = d.get("cot")
    if cot is None:
        final_content = d.get("final_content", "")
        reasoning_content = d.get("reasoning_content", "")
        if reasoning_content:
            cot = f"<think>{reasoning_content}</think>\n{final_content}"
        else:
            cot = final_content or d.get("reasoning", "")

    # teacher_model: 직접 필드 또는 raw 'model' 필드
    teacher_model = d.get("teacher_model") or d.get("model", "")

    # predicted_answer: int(processed) 또는 letter string(raw) 모두 허용
    predicted_raw = d.get("predicted_answer")
    if predicted_raw is None:
        predicted = d.get("predicted")
        if predicted is not None and isinstance(predicted, str):
            predicted_raw = _LETTER_TO_IDX.get(predicted)
    elif isinstance(predicted_raw, str):
        predicted_raw = _LETTER_TO_IDX.get(predicted_raw, predicted_raw)

    # choices: 리스트 또는 A/B/C/D/E 필드로부터 생성
    choices = d.get("choices", [])
    if not choices and all(k in d for k in ["A", "B", "C", "D", "E"]):
        choices = [d["A"], d["B"], d["C"], d["D"], d["E"]]

    return CoTSample(
        id=sample_id,
        subset=d["subset"],
        question=d.get("question", ""),
        choices=choices,
        answer=answer_raw,
        teacher_model=teacher_model,
        cot=cot,
        predicted_answer=predicted_raw,
        filter_history=filter_history,
        judge_score=judge_score,
        metadata=d.get("metadata", {}),
    )


def _sample_to_dict(s: CoTSample) -> dict:
    return {
        "id": s.id,
        "subset": s.subset,
        "question": s.question,
        "choices": s.choices,
        "answer": s.answer,
        "teacher_model": s.teacher_model,
        "cot": s.cot,
        "predicted_answer": s.predicted_answer,
        "filter_history": [
            {
                "filter_name": r.filter_name,
                "passed": r.passed,
                "score": r.score,
                "reason": r.reason,
            }
            for r in s.filter_history
        ],
        "judge_score": {
            "step_coherence": s.judge_score.step_coherence,
            "korean_fluency": s.judge_score.korean_fluency,
            "step_coverage": s.judge_score.step_coverage,
        } if s.judge_score else None,
        "metadata": s.metadata,
    }


def load_samples(path: Union[str, Path]) -> list[CoTSample]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    return [_dict_to_sample(d) for d in data]


def save_samples(samples: list[CoTSample], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [_sample_to_dict(s) for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for d in dicts:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        else:
            json.dump(dicts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} samples → {path}")
