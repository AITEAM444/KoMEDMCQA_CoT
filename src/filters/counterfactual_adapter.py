"""
F6 Counterfactual CoT 생성 — train CoT 생성(eval_cot.py fewshot 모드)과 *동일한 설계*로,
정답 대신 **오답 하나를 강제**해 그 오답을 정당화하는 추론을 DeepSeek-R1 으로 생성한다.

설계 일관성 (eval_cot fewshot 과 동일):
  - 영어 instruction + 분야별 한국어 fewshot (추론 퀄리티↑, 내부추론 언어는 비강제)
  - 신호원 = reasoning_content (R1 의 진짜 내부추론)
차이 (counterfactual):
  - "정답은 {오답}" 으로 *오답* 을 강제 → 그 오답을 정당화하는 cf 추론을 생성
  - F6 판정: gap = orig_score - cf_score (채점은 별도 단계, GPT-5). 오답도 원본만큼
    그럴듯하게 정당화되면(gap 작음) = 답 무관 사후합리화 의심 → 탈락

오답 선택은 (seed + global_idx) 로 재현 가능. checkpoint(출력파일 이어쓰기)·parallel 내장.

사용:
    export DEEPSEEK_API_KEY=...
    # 4분야 전체 (각 분야 자기 fewshot 자동 적용)
    python scripts/gen_counterfactual.py --total -1 --split train \
        --workers 16 --output data/raw/train_cf.jsonl
    # 한 분야만
    python scripts/gen_counterfactual.py --total -1 --split train \
        --subset doctor --workers 16 --output data/raw/cf_doctor.jsonl
    # 중단 후 재개: 같은 명령 재실행 (성공분 건너뜀). 처음부터: --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "generation"))   # generate_traces (구 eval_cot)
_sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root (filters/, utils/)

from generate_traces import (  # 동일 인프라 재사용 (구 eval_cot)
    DEEPSEEK_MODEL_IDS,
    SUBSETS,
    _LANGDETECT_AVAILABLE,
    _format_q_block,
    detect_think_language,
    load_done,
    load_fewshot_blocks,
    load_samples,
    parse_answer,
    parse_shard,
    shard_output_path,
)

_LETTERS = ["A", "B", "C", "D", "E"]

# eval_cot 의 fewshot instruction 과 동일 톤. 단 *정답* 이 아니라 *주어진(오답) 선지* 를
# 정당화하게 한다. 추론 방식·언어는 지시하지 않는다 (fewshot 만 한국어 노출).
COUNTERFACTUAL_INSTRUCTION = """You are a medical expert taking the Korean medical licensing examination.
For the following question, treat {wrong_letter} as the correct answer.
Provide careful clinical reasoning that justifies why {wrong_letter} is the correct answer.
Write the final answer on the last line exactly as: 정답: {wrong_letter}
Do not write anything after the answer line.
===== Example =====
{fewshot}
===== Now answer the following =====
{question_block}"""


def pick_wrong_idx(gold_idx, n_choices, seed):
    """gold 를 제외한 오답 인덱스 1개를 seed 로 재현 가능하게 선택."""
    rng = random.Random(seed)
    pool = [i for i in range(n_choices) if i != gold_idx]
    rng.shuffle(pool)
    return pool[0]


def build_cf_prompt(sample, wrong_idx, fewshot_blocks):
    subset = sample.get("subject") or sample.get("subset")
    return COUNTERFACTUAL_INSTRUCTION.format(
        wrong_letter=_LETTERS[wrong_idx],
        fewshot=fewshot_blocks[subset],
        question_block=_format_q_block(sample),
    )


def generate_cf_one(client, model, sample, wrong_idx, fewshot_blocks, seed):
    prompt = build_cf_prompt(sample, wrong_idx, fewshot_blocks)
    api_model = DEEPSEEK_MODEL_IDS.get(model, model)
    gold_idx = int(sample["answer"]) - 1
    out = {"gold_letter": _LETTERS[gold_idx], "wrong_letter": _LETTERS[wrong_idx]}
    try:
        resp = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=seed,
            max_tokens=8000,
        )
        msg = resp.choices[0].message
        content = msg.content
        # R1 의 진짜 내부추론 = reasoning_content. 이것이 cf 의 <think> 신호원.
        reasoning_content = getattr(msg, "reasoning_content", None)
        think = detect_think_language(reasoning_content)
        out.update({
            "predicted": parse_answer(content),
            "cf_content": content,
            "cf_reasoning_content": reasoning_content,
            "think_lang": think["label"],
            "think_lang_probs": think["probs"],
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-r1")
    p.add_argument("--total", type=int, default=100, help="-1 이면 split 전체")
    p.add_argument("--split", default="train", choices=["train", "test", "fewshot", "dev"])
    p.add_argument("--subset", default=None, choices=SUBSETS,
                   help="한 분야만 생성 (생략 시 4분야 전체).")
    p.add_argument("--seed", type=int, default=42, help="오답 선택 + 디코딩 seed 베이스")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", default=None, help="'i/N' 분산 실행")
    p.add_argument("--overwrite", action="store_true", help="기존 출력 무시하고 새로 시작")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.split == "test":
        raise SystemExit("test split 생성 금지 (보고서 §2). --split train 사용.")

    model = args.model
    workers = max(1, args.workers)
    shard = parse_shard(args.shard)
    base_output = args.output or f"cf_{model}_{args.split}_{args.total}.jsonl"
    output_file = shard_output_path(base_output, shard)

    if model.startswith("deepseek"):
        client = OpenAI(base_url="https://api.deepseek.com",
                        api_key=os.environ["DEEPSEEK_API_KEY"])
    else:
        client = OpenAI()

    run_subsets = [args.subset] if args.subset else SUBSETS
    samples = load_samples(args.total, split=args.split, subsets=run_subsets)
    fewshot_blocks = load_fewshot_blocks(run_subsets)
    print(f"=== Counterfactual 생성 ({model}, fewshot, n={len(samples)}) ===")
    print(f"분야: {run_subsets} | workers={workers} | seed={args.seed}")
    if not _LANGDETECT_AVAILABLE:
        print("⚠ langdetect 미설치 — think 언어 분포 비활성 (pip install langdetect).")

    indexed = list(enumerate(samples))
    if shard is not None:
        i, n = shard
        indexed = [(g, it) for g, it in indexed if g % n == i]

    done = {} if args.overwrite else load_done(output_file)
    pending = [(g, it) for g, it in indexed if g not in done]
    if done:
        print(f"체크포인트: 기존 {len(done)}건 완료 → 건너뜀, {len(pending)}건 남음")
    print(f"이번 실행 대상: {len(pending)}건\n")

    def run_one(global_idx, item):
        sample = item["sample"]
        gold_idx = int(sample["answer"]) - 1
        n_choices = sum(1 for k in _LETTERS if sample.get(k)) or 5
        wrong_idx = pick_wrong_idx(gold_idx, n_choices, args.seed + global_idx)
        res = generate_cf_one(client, model, sample, wrong_idx, fewshot_blocks, args.seed)
        res["global_idx"] = global_idx
        res["subset"] = item["subset"]
        res["sample_idx"] = item["sample_idx"]
        res["question"] = sample["question"][:100] + "..."
        res["model"] = model
        return res

    lang_counter = Counter()
    n_err = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in done.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        def write(res):
            nonlocal n_err
            if res.get("error"):
                n_err += 1
            elif res.get("think_lang"):
                lang_counter[res["think_lang"]] += 1
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

        if workers == 1:
            for g, it in tqdm(pending, desc="cf 생성"):
                write(run_one(g, it))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_one, g, it): g for g, it in pending}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="cf 생성"):
                    write(fut.result())

    print(f"\n완료 → {output_file} | 에러 {n_err}건")
    if lang_counter:
        print("cf think(reasoning_content) 언어 분포:", dict(lang_counter))


if __name__ == "__main__":
    main()
