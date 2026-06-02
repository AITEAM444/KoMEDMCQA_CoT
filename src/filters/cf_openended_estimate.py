"""
개방형(주관식) 데이터셋용 반사실 필터 — 로컬 vLLM self-judge 탈락률 추정.

기존 counterfactual 필터는 MCQ 선지에 묶여 있다(틀린 '선지'를 뽑아 정당화).
개방형(medical-o1: Question/Complex_Cot/Response)에는 선지가 없으므로:
  1) gold 답을 보고 '그럴듯하지만 틀린 대안 답'을 로컬 모델이 생성
  2) 그 오답을 정당화하는 CF CoT 를 로컬 모델이 생성
  3) self-judge 로 원본 CoT(→gold) / CF CoT(→오답) 정당화 강도 1~5 채점
  4) gap = orig_score - cf_score; gap<gap_thr 또는 orig<min_orig → 탈락(post-hoc 의심)

목적: '이 트레이스 풀에 거를 나쁜 추론이 있는가'(필터 여지 = 품질 편차) 확인.
  - 탈락률 적당(예 15~40%) + gap 분포 퍼짐 → 필터가 할 일 있음 (green)
  - 탈락률 ~0% (gap 다 큼) → 이미 깨끗(s1K 문제), 필터 무의미
  - 탈락률 ~100% → 임계/judge 과민 — 점검 필요

⚠ self-judge 라 절대 점수엔 bias. '여지 있나' 추정용. 본실험은 더 강한 judge 권장.

사용:
    python src/filters/cf_openended_estimate.py \
        --model Qwen/Qwen3-8B --dataset ChuGyouk/medical-o1-reasoning-SFT-Ko \
        --total 100 --output results/cf_estimate_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter

_WRONG_INSTR = (
    "다음 의료 질문의 정답은 \"{gold}\" 입니다. 이와 다르면서도 임상적으로 "
    "그럴듯해 보이는 **틀린 대안 답**(다른 진단명/처방/수치 등)을 딱 하나만, "
    "한 줄로 제시하세요. 설명 없이 답만 쓰세요.\n\n[질문]\n{q}"
)
_CF_INSTR = (
    "당신은 의료 전문가입니다. 다음 질문의 정답이 \"{wrong}\" 라고 가정하고, "
    "왜 그것이 옳은지 단계적으로 임상 추론하세요. 마지막 줄에 \"정답: {wrong}\" 라고 쓰세요."
    "\n\n[질문]\n{q}"
)
_JUDGE_INSTR = (
    "아래 추론이 목표답을 임상적으로 얼마나 강하게 정당화하는지 1~5 정수로만 채점하세요. "
    "맨 끝의 '정답:' 선언 줄은 무시하고 본문 임상 내용만 보세요. "
    "본문이 실제로 그 목표답을 지지하지 못하면 1점.\n"
    "5=결정적 근거+대안 배제, 4=강한 정당화, 3=합리적이나 약함, 2=재진술 수준, 1=정당화 실패.\n\n"
    "[목표답]\n{target}\n\n[추론]\n{cot}\n\n점수(1~5 정수만):"
)
_DIGIT = re.compile(r"[1-5]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--dataset", default="ChuGyouk/medical-o1-reasoning-SFT-Ko")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gap-threshold", type=float, default=1.5)
    ap.add_argument("--min-orig-score", type=float, default=2.0)
    ap.add_argument("--cf-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--output", default="results/cf_estimate.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ds = load_dataset(args.dataset, split="train")
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    rows = [ds[i] for i in idxs[:args.total]]
    print(f"[cf-est] {args.dataset}: 표본 {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enable_lora=bool(args.lora), max_lora_rank=args.max_lora_rank)
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None

    def chat(user):
        return tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=args.enable_thinking)

    def gen(prompts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp, lora_request=lora)]

    def judge(cots, targets):
        sp = SamplingParams(temperature=0.0, max_tokens=4)
        ps = [chat(_JUDGE_INSTR.format(target=t[:400], cot=c[:4000]))
              for c, t in zip(cots, targets)]
        outs = llm.generate(ps, sp, lora_request=lora)
        scores = []
        for o in outs:
            m = _DIGIT.search(o.outputs[0].text)
            scores.append(int(m.group()) if m else 1)
        return scores

    # 1) 오답 생성
    print("[cf-est] 1/4 오답 생성...")
    wrongs = gen([chat(_WRONG_INSTR.format(gold=r["Response"][:300], q=r["Question"]))
                  for r in rows], 64)
    wrongs = [w.splitlines()[0][:200] if w else "알 수 없음" for w in wrongs]

    # 2) CF CoT 생성
    print("[cf-est] 2/4 CF CoT 생성...")
    cf_cots = gen([chat(_CF_INSTR.format(wrong=w, q=r["Question"]))
                   for r, w in zip(rows, wrongs)], args.cf_tokens)

    # 3) 채점: orig(→gold) / cf(→wrong)
    print("[cf-est] 3/4 orig 채점...")
    orig_scores = judge([r["Complex_Cot"] for r in rows], [r["Response"] for r in rows])
    print("[cf-est] 4/4 cf 채점...")
    cf_scores = judge(cf_cots, wrongs)

    # 4) 판정
    n_drop = n_safety = n_gap = 0
    gaps = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fo:
        for r, w, oc, cf_sc, cf_cot in zip(rows, wrongs, orig_scores, cf_scores, cf_cots):
            gap = oc - cf_sc
            gaps.append(gap)
            safety = oc < args.min_orig_score
            gapcut = gap < args.gap_threshold
            drop = safety or gapcut
            n_drop += drop; n_safety += safety; n_gap += (gapcut and not safety)
            fo.write(json.dumps({
                "question": r["Question"][:200], "gold": r["Response"][:150],
                "wrong": w, "orig_score": oc, "cf_score": cf_sc, "gap": gap,
                "cf_cot": cf_cot[:400],
                "drop": drop, "reason": "orig<min" if safety else ("gap<thr" if gapcut else ""),
            }, ensure_ascii=False) + "\n")

    n = len(rows)
    print("\n" + "=" * 56)
    print(f"[개방형 반사실 필터 탈락률 추정] n={n}, model={args.model}")
    print(f"  orig_score 분포: {dict(Counter(orig_scores))}  (평균 {statistics.mean(orig_scores):.2f})")
    print(f"  cf_score   분포: {dict(Counter(cf_scores))}  (평균 {statistics.mean(cf_scores):.2f})")
    print(f"  gap 평균 {statistics.mean(gaps):.2f}, median {statistics.median(gaps):.1f}, "
          f"min {min(gaps)}, max {max(gaps)}")
    print(f"\n  탈락률 = {n_drop}/{n} = {100*n_drop/n:.1f}%")
    print(f"    - 안전망(orig<{args.min_orig_score}) = {n_safety}")
    print(f"    - gap<{args.gap_threshold}          = {n_gap}")
    print("\n  해석: 탈락 15~40% + gap 퍼짐 → 필터 여지 O (green)")
    print("        탈락 ~0% → 이미 깨끗(s1K 문제) / ~100% → judge·임계 점검")
    print(f"\n저장 → {args.output}")


if __name__ == "__main__":
    main()
