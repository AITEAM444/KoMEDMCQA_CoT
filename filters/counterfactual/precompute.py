"""
Counterfactual Answer Probing 사전 계산기 (Inverse MATCHA).

각 샘플에 대해:
  (1) wrong choice K 개 선택 (default K=1, 재현 가능한 seed)
  (2) 각 wrong_idx 마다 R1 에 "정답=wrong_idx 라는 전제로 CoT 작성" 요청
      → 인라인 TeacherRegenerator (forced 모드) 사용
  (3) Solar G-Eval 로 "정당화 강도" 채점
      · 원본 CoT 는 target=gold 로 채점
      · CF CoT 는 target=해당 wrong 로 채점
  (4) metadata["counterfactual"] 에 적재:
        {
          "k": int,
          "orig_score": float | None,
          "counterfactuals": [
            {"target_idx": int, "target_letter": "A"~"E",
             "cf_cot": str, "cf_score": float | None,
             "error": str | None},
            ...
          ]
        }

이 결과는 추가 API 호출 없이 run.py 에서 gap = orig_score - max(cf_score)
임계값으로 필터링된다.

사용:
    python -m filters.counterfactual.precompute \
      --input  data/filtered/02_after_structure.json \
      --output data/filtered/counterfactual_output_precomputed.json \
      --k 1 --limit 30 --sleep 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from filters.counterfactual.judge import score_convincingness
from utils.data_loader import load_samples, save_samples
from utils.schema import CoTSample


def _make_judge(upstage_key: str | None, deepseek_key: str | None) -> tuple[object, str]:
    """judge backend 자동 선택.

    UPSTAGE_API_KEY 가 있으면 Solar Pro 3 (원본 F8 인프라와 일치),
    아니면 DeepSeek-Chat (싸고 빠름, R1 보다 채점 안정).
    """
    from openai import OpenAI
    if upstage_key:
        return OpenAI(api_key=upstage_key, base_url="https://api.upstage.ai/v1"), "solar-pro3"
    if deepseek_key:
        return OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"), "deepseek-chat"
    raise RuntimeError("UPSTAGE_API_KEY 또는 DEEPSEEK_API_KEY 중 하나가 필요합니다 (G-Eval judge용).")


_LETTERS = ["A", "B", "C", "D", "E"]

# content 안에 모델이 끼워 넣은 <think>...</think> 블록 제거용 (R1 의 진짜 reasoning_content 와 중복 방지)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── Teacher CoT 재생성기 (구 filters/counterfactual/regenerator.py 인라인) ──────────
# counterfactual 은 forced 모드만 사용한다(정답을 prompt 에 박고 그 정답이 왜 맞는지
# R1 이 단계별 추론하게 함). MATCHA 전용이던 free 모드는 함께 제거했다.
_REGEN_PROMPT_FORCED = """당신은 한국 의료 자격시험 풀이를 4단계로 작성하는 전문가입니다.
정답은 이미 확정되었으며, 그 정답이 왜 맞는지를 단계별로 추론하여 보여 주는 것이 과제입니다.

[제약]
- 정답은 {answer_letter} 입니다. 정답을 절대 변경하지 마십시오.
- 다음 4단계 라벨을 그대로 사용하시오: [Step 1] / [Step 2] / [Step 3] / [Step 4]
- 마지막 줄은 반드시 "정답: {answer_letter}" 로 끝내시오.
- [Step 1]~[Step 4]와 "정답:" 줄(최종 답변)은 모두 한국어로 작성하시오.

[문제]
{question}

[선지]
{choices_block}

[출력 — 아래 4단계와 정답 줄만 작성. 내부 사고는 적지 말 것(모델이 자체적으로 수행)]
[Step 1] ...
[Step 2] ...
[Step 3] ...
[Step 4] ...
정답: {answer_letter}
"""


@dataclass
class RegenResult:
    cot: str
    model: str
    error: str | None = None


class TeacherRegenerator:
    """정답을 prompt 에 고정하고 그 정답의 4-step 정당화 CoT 를 R1 에 생성시킨다.

    환경변수:
        MATCHA_TEACHER_BASE_URL  (default: https://api.deepseek.com)
        MATCHA_TEACHER_MODEL     (default: deepseek-reasoner)
        MATCHA_TEACHER_API_KEY   (default: DEEPSEEK_API_KEY)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url or os.environ.get(
            "MATCHA_TEACHER_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model or os.environ.get("MATCHA_TEACHER_MODEL", "deepseek-reasoner")
        self.api_key = api_key or os.environ.get(
            "MATCHA_TEACHER_API_KEY"
        ) or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Teacher API key가 필요합니다. "
                "MATCHA_TEACHER_API_KEY 또는 DEEPSEEK_API_KEY 환경변수를 설정하세요."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _format_choices(self, choices: list[str]) -> str:
        if not choices:
            return "(선지는 문제 본문에 포함되어 있습니다)"
        return "\n".join(
            f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices) if i < len(_LETTERS)
        )

    def regenerate(
        self,
        question: str,
        answer_idx: int,
        choices: list[str],
    ) -> RegenResult:
        if not (0 <= answer_idx < len(_LETTERS)):
            return RegenResult(cot="", model=self.model, error=f"invalid answer_idx={answer_idx}")

        prompt = _REGEN_PROMPT_FORCED.format(
            answer_letter=_LETTERS[answer_idx],
            question=question.strip(),
            choices_block=self._format_choices(choices),
        )

        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            return RegenResult(cot="", model=self.model, error=f"{type(e).__name__}: {e}")

        msg = resp.choices[0].message
        # deepseek-reasoner(R1)는 *진짜* 내부추론을 reasoning_content 로 분리 제공한다.
        # 이것이 원본 Teacher CoT(eval_cot)의 <think> 와 동일한 신호원이므로 **우선 사용**한다.
        # 프롬프트는 더 이상 content 에 <think> 를 요구하지 않지만, 모델이 혹시 content 에
        # <think> 를 끼워 넣었으면 제거해 진짜 reasoning_content 와 중복되지 않게 한다.
        reasoning = getattr(msg, "reasoning_content", None)
        content = (msg.content or "").strip()
        content_no_think = _THINK_BLOCK_RE.sub("", content).strip() if "<think>" in content.lower() else content
        if reasoning:
            cot = f"<think>{reasoning}</think>\n{content_no_think}"
        elif "<think>" in content.lower():
            cot = content        # 비-R1 모델: 프롬프트가 없어도 content 에 <think> 가 있으면 사용
        else:
            cot = content
        return RegenResult(cot=cot.strip(), model=self.model, error=None)


def _pick_wrong_indices(gold: int, k: int, n_choices: int, rng: random.Random) -> list[int]:
    """gold 를 제외한 wrong choice 인덱스를 k 개 비복원 추출."""
    pool = [i for i in range(n_choices) if i != gold]
    rng.shuffle(pool)
    return pool[:k]


def precompute_sample(
    sample: CoTSample,
    k: int,
    wrong_idxs: list[int],
    regenerator: TeacherRegenerator,
    judge_client,
    judge_model: str,
    judge_reps: int,
    judge_temp: float,
    n_choices: int,
    regen_only: bool = False,
) -> dict:
    # regen_only(F6 1단계): 채점 생략, cf_cot 생성만. judge는 별도 단계에서 수행.
    if not regen_only:
        orig = score_convincingness(
            judge_client, sample.question, sample.cot, sample.answer,
            model=judge_model, n_reps=judge_reps, temperature=judge_temp,
            choices=sample.choices,
        )

    # CF CoT 생성 (+ regen_only 가 아니면 채점). wrong_idxs는 외부에서 미리 결정 → 재현성.
    cfs = []
    for w_idx in wrong_idxs:
        regen = regenerator.regenerate(sample.question, w_idx, sample.choices)
        if regen.error or not regen.cot:
            cfs.append({
                "target_idx": w_idx, "target_letter": _LETTERS[w_idx],
                "cf_cot": regen.cot, "cf_score": None,
                "judge_runs": [], "regen_model": regen.model,
                "error": regen.error or "empty_cot",
            })
            continue
        if regen_only:
            cfs.append({
                "target_idx": w_idx, "target_letter": _LETTERS[w_idx],
                "cf_cot": regen.cot, "cf_score": None,
                "judge_runs": [], "regen_model": regen.model, "error": None,
            })
            continue
        cf = score_convincingness(
            judge_client, sample.question, regen.cot, w_idx,
            model=judge_model, n_reps=judge_reps, temperature=judge_temp,
            choices=sample.choices,
        )
        cfs.append({
            "target_idx": w_idx, "target_letter": _LETTERS[w_idx],
            "cf_cot": regen.cot,
            "cf_score": cf["mean"],
            "judge_runs": cf["runs"],
            "regen_model": regen.model,
            "error": None,
        })

    return {
        "k": k,
        "orig_target_letter": _LETTERS[sample.answer],
        "orig_score": None if regen_only else orig["mean"],
        "orig_judge_runs": [] if regen_only else orig["runs"],
        "counterfactuals": cfs,
    }


def run(
    input_path: Path,
    output_path: Path,
    k: int,
    limit: int | None,
    seed: int,
    sleep: float,
    judge_reps: int,
    judge_temp: float,
    n_choices: int,
    workers: int = 1,
    regen_only: bool = False,
) -> None:
    samples = load_samples(input_path)
    if limit is not None:
        samples = samples[:limit]

    # rng 는 순차적으로 모두 굴려 wrong_idxs 를 미리 결정 — 병렬 실행 시 재현성 보장
    rng = random.Random(seed)
    wrong_idxs_per_sample = [
        _pick_wrong_indices(s.answer, k, n_choices, rng) for s in samples
    ]

    regenerator = TeacherRegenerator()
    # 쓰레드 풀 launch 전에 client 강제 초기화 (lazy init race 방지)
    regenerator._get_client()

    if regen_only:
        judge_client, judge_model = None, "(regen-only: 채점 생략)"
    else:
        judge_client, judge_model = _make_judge(
            upstage_key=os.environ.get("UPSTAGE_API_KEY"),
            deepseek_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    print(
        f"[counterfactual:precompute] judge backend = {judge_model}, "
        f"workers = {workers}, n_samples = {len(samples)}, regen_only = {regen_only}"
    )

    def _process(i: int) -> tuple[int, dict]:
        sample = samples[i]
        try:
            if not sample.choices:
                return i, {"k": k, "counterfactuals": [], "error": "empty_choices"}
            return i, precompute_sample(
                sample, k, wrong_idxs_per_sample[i],
                regenerator, judge_client, judge_model,
                judge_reps, judge_temp, n_choices, regen_only=regen_only,
            )
        except Exception as e:
            return i, {"k": k, "counterfactuals": [], "error": f"{type(e).__name__}: {e}"}
        finally:
            if sleep > 0:
                time.sleep(sleep)  # 워커당 호출 후 sleep — rate-limit 완충

    # 체크포인트 — 완료 샘플을 인덱스별 JSONL 로 append. 재실행 시 끝난 인덱스 건너뜀.
    # (장시간 deepseek 재생성 중 Colab 끊김 등에서 진행분 보존)
    ckpt_path = output_path.with_suffix(".ckpt.jsonl")
    results: dict[int, dict] = {}
    if ckpt_path.exists():
        for line in open(ckpt_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                results[int(rec["i"])] = rec["result"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        print(f"[counterfactual:precompute] 체크포인트 {len(results)}건 재사용 ← {ckpt_path}")
    todo = [i for i in range(len(samples)) if i not in results]
    print(f"[counterfactual:precompute] 신규 처리 {len(todo)}건 (체크포인트 {len(results)}건 건너뜀)")

    done_counter = {"n": len(results)}
    log_lock = threading.Lock()
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    def _is_good(result: dict) -> bool:
        """성공 판정 — 에러 없고 cf 가 모두 cf_cot(생성 성공)을 가질 때만 True.
        실패/빈 것은 체크포인트에 안 남겨 재실행 시 재시도되게 한다."""
        if result.get("error"):
            return False
        cfs = result.get("counterfactuals", [])
        if not cfs:
            return False
        return all(c.get("cf_cot") and not c.get("error") for c in cfs)

    def _log(i: int, result: dict) -> None:
        with log_lock:
            done_counter["n"] += 1
            if _is_good(result):
                ckpt_f.write(json.dumps({"i": i, "id": str(samples[i].id), "result": result},
                                        ensure_ascii=False) + "\n")
                ckpt_f.flush()
            sample = samples[i]
            err = result.get("error")
            if err:
                print(
                    f"[counterfactual:precompute] {done_counter['n']}/{len(samples)} "
                    f"id={sample.id} ERROR(미저장,재시도대상): {err}"
                )
            else:
                cfs = result.get("counterfactuals", [])
                cf_scores = [c.get("cf_score") for c in cfs if c.get("cf_score") is not None]
                print(
                    f"[counterfactual:precompute] {done_counter['n']}/{len(samples)} "
                    f"id={sample.id} orig={result.get('orig_score')} cf_scores={cf_scores}"
                )

    if workers <= 1:
        for i in todo:
            idx, result = _process(i)
            results[idx] = result
            _log(idx, result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, i) for i in todo]
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result
                _log(idx, result)
    ckpt_f.close()

    # 결과를 입력 순서대로 sample.metadata 에 적재 (미처리분은 placeholder)
    for i, sample in enumerate(samples):
        sample.metadata = {**sample.metadata,
                           "counterfactual": results.get(i, {"k": k, "counterfactuals": [], "error": "not_processed"})}

    save_samples(samples, output_path)
    n_missing = sum(1 for i in range(len(samples)) if i not in results)
    if n_missing:
        print(f"[counterfactual:precompute] ⚠️ 미처리 {n_missing}건 — 같은 명령 재실행 시 이어서 처리 (체크포인트: {ckpt_path})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=1, help="샘플당 생성할 wrong-answer CF 개수 (default 1)")
    p.add_argument("--limit", type=int, default=None, help="초소형 파일럿용 (e.g. 30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--judge-reps", type=int, default=1, help="채점 반복 횟수 (편향 완화)")
    p.add_argument("--judge-temp", type=float, default=0.3)
    p.add_argument("--n-choices", type=int, default=5, help="MCQ 선지 개수 (KorMedMCQA 기본 5)")
    p.add_argument("--workers", type=int, default=1,
                   help="동시 API 호출 쓰레드 수 (default 1=순차). 8~16 권장, rate-limit 초과 시 낮춰라.")
    p.add_argument("--regen-only", action="store_true",
                   help="F6 1단계: 채점(judge) 생략, cf_cot 재생성만 수행. UPSTAGE/DEEPSEEK judge 키 불필요. "
                        "채점은 별도 단계(GPT-5)에서. cf_score/orig_score 는 null 로 저장됨.")
    args = p.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        k=args.k,
        limit=args.limit,
        seed=args.seed,
        sleep=args.sleep,
        judge_reps=args.judge_reps,
        judge_temp=args.judge_temp,
        n_choices=args.n_choices,
        workers=args.workers,
        regen_only=args.regen_only,
    )
