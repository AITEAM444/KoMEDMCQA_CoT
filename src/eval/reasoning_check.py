"""
개방형 데이터셋의 reasoning-bound 여부 사전 확인.

질문 단독으로 답하게 했을 때(no-reasoning) vs gold CoT 를 준 뒤 답하게 했을 때
(with-reasoning) 정답률을 비교한다. 자유서술이라 정답 여부는 LLM judge 로 채점.

  reasoning_gain = acc(with CoT) - acc(no CoT)
    - 크면  → 추론이 답을 좌우 = reasoning-bound  (필터가 효과 낼 여지 있음)
    - 작으면 → 지식 주도(knowledge-bound)          = KorMedMCQA 와 같은 막다른 길

데이터셋: ChuGyouk/medical-o1-reasoning-SFT-Ko (Question / Complex_Cot / Response).

사용:
    python src/eval/reasoning_check.py \
        --model Qwen/Qwen3-8B \
        --dataset ChuGyouk/medical-o1-reasoning-SFT-Ko \
        --total 200 --output results/reasoning_check_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

ANS_INSTRUCTION = (
    "다음 의료 질문에 답하세요. 마지막 줄에 \"답:\" 으로 시작하는 한 줄에 "
    "핵심 결론(진단명·답)을 간결히 쓰세요."
)
JUDGE_INSTRUCTION = (
    "두 의료 답변이 핵심 결론에서 같은지 판단하세요. "
    "표현이 달라도 핵심 진단/답이 일치하면 YES, 다르면 NO 만 출력하세요."
)
_YES = re.compile(r"\b(YES|Y|예|일치)\b", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """생성문에서 '답:' 뒤를 우선 추출, 없으면 전체."""
    i = text.rfind("답:")
    return (text[i + 2:] if i >= 0 else text).strip()[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--dataset", default="ChuGyouk/medical-o1-reasoning-SFT-Ko")
    ap.add_argument("--total", type=int, default=200, help="표본 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--answer-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--output", default="results/reasoning_check.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ds = load_dataset(args.dataset, split="train")
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[:args.total]
    rows = [ds[i] for i in idxs]
    print(f"[check] {args.dataset}: 표본 {len(rows)} / 전체 {len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enable_lora=bool(args.lora), max_lora_rank=args.max_lora_rank)
    lora_req = LoRARequest("adapter", 1, args.lora) if args.lora else None
    gen_sp = SamplingParams(temperature=0.0, max_tokens=args.answer_tokens)
    judge_sp = SamplingParams(temperature=0.0, max_tokens=4)

    def chat(user, prefill=""):
        base = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=args.enable_thinking)
        return base + prefill

    # 두 조건의 프롬프트
    p_noreason, p_withreason = [], []
    for r in rows:
        q = r["Question"]
        user = f"{ANS_INSTRUCTION}\n\n[질문]\n{q}"
        p_noreason.append(chat(user, prefill="\n\n답:"))                       # f=0
        p_withreason.append(chat(user, prefill=f"{r['Complex_Cot']}\n\n답:"))  # f=1.0

    print("[check] no-reasoning 생성...")
    out_a = llm.generate(p_noreason, gen_sp, lora_request=lora_req)
    print("[check] with-reasoning 생성...")
    out_b = llm.generate(p_withreason, gen_sp, lora_request=lora_req)
    ans_a = [extract_answer(o.outputs[0].text) for o in out_a]
    ans_b = [extract_answer(o.outputs[0].text) for o in out_b]

    # judge (같은 모델 self-judge — 사전체크용. 본실험은 별도/강한 judge 권장)
    def judge_prompts(answers):
        ps = []
        for r, a in zip(rows, answers):
            user = (f"{JUDGE_INSTRUCTION}\n\n[기준답]\n{r['Response'][:1500]}\n\n"
                    f"[채점대상]\n{a}")
            ps.append(chat(user))
        return ps
    print("[check] judge 채점...")
    j_a = llm.generate(judge_prompts(ans_a), judge_sp, lora_request=lora_req)
    j_b = llm.generate(judge_prompts(ans_b), judge_sp, lora_request=lora_req)
    ok_a = [bool(_YES.search(o.outputs[0].text)) for o in j_a]
    ok_b = [bool(_YES.search(o.outputs[0].text)) for o in j_b]

    acc_a = sum(ok_a) / len(rows)
    acc_b = sum(ok_b) / len(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fo:
        for r, a, b, ka, kb in zip(rows, ans_a, ans_b, ok_a, ok_b):
            fo.write(json.dumps({
                "question": r["Question"][:300], "gold": r["Response"][:300],
                "ans_noreason": a, "ok_noreason": ka,
                "ans_withreason": b, "ok_withreason": kb,
            }, ensure_ascii=False) + "\n")

    print("\n" + "=" * 56)
    print(f"[reasoning-bound 사전확인] {args.dataset}  (n={len(rows)}, model={args.model})")
    print(f"  no-reasoning  정답률(f=0)   = {100*acc_a:.2f}%")
    print(f"  with-reasoning 정답률(f=1.0) = {100*acc_b:.2f}%")
    print(f"  reasoning_gain = {100*(acc_b-acc_a):+.2f}%p")
    print(f"  early_commit_ratio = {acc_a/acc_b:.3f}   (낮을수록 reasoning-bound)" if acc_b else "")
    print("\n  해석: gain 크면(예: +10%p↑) reasoning-bound → 필터 효과 여지 O")
    print("        gain 작으면 knowledge-bound → MCQ 와 같은 막다른 길")
    print(f"\n저장 → {args.output}")


if __name__ == "__main__":
    main()
