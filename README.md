# KoMED — Korean Medical CoT Distillation & Counterfactual Faithfulness Filter

DeepSeek-R1 교사로 KorMedMCQA train 에 long-CoT(reasoning) trace 를 생성하고, **반사실 추론
충실성 필터(Inverse-MATCHA)**로 "답을 바꿔도 똑같이 그럴듯하게 정당화하는 사후합리화 CoT"를
걸러낸 뒤 Qwen3-8B student(LoRA)를 학습·평가한다.

**주장**: 정답이 맞는 trace 안에서 범용 LLM Judge는 변별력을 잃지만, 반사실 프로빙은 *답을
바꿨을 때의 정당화 강도 격차*라는 독립축에서 사후합리화 trace 를 잡아낸다. 더 적은 데이터로
student 정확도를 유지·향상시키는 것이 증거.

비교군: **C0**(무필터) · **C1**(표준 baseline) · **C2**(C1+범용 Judge) · **C-rand**(C1 무작위 축소) · **C3**(C1+반사실).
핵심: C3≥C1 / C3>C-rand / C3>C2.

## 절대 규칙
- **test split 에는 Teacher CoT 를 생성하지 않는다** — 최종 평가(정확도)에만. calibration/early-stop 은 dev.

## 실제 구성 (현재 구현된 것)
```
configs/
  pipeline_config.yaml        # 필터 파라미터 (counterfactual gap/hedge/min_orig 등)
  train.yaml / qwen3_8b_*.yaml # Qwen3-8B LoRA/full SFT (LLaMA-Factory)
  ds_zero3.json               # full FT DeepSpeed ZeRO-3
  dataset_info.json           # LLaMA-Factory 데이터 등록
src/
  generation/generate_traces.py     # R1 trace 생성 (영어 instruction + 분야별 한국어 fewshot, free, reasoning_content 캡처)
  filters/counterfactual_adapter.py  # 반사실 CF 생성 (오답 강제 정당화, reasoning_content 신호)
  dataset/build_arms.py              # SFT 데이터 export (reasoning_content + 정답; no-filter / correct-only)
  eval/evaluate.py                   # KorMedMCQA test 정확도 (전체/과목별, resume 내장)
  eval/compare_judges.py             # judge 모델 동등성 비교 (ref vs cand)
  data/load_kormedmcqa.py            # KorMedMCQA 선지 join 유틸
filters/                             # 작동 패키지: counterfactual{precompute,run,judge}(기여) + correctness/structure(C1) + base/think_final_divergence/judge(의존)
utils/                               # CoTSample 스키마 + 데이터 로더
data/ , results/                     # 데이터·체크포인트 (.gitignore)
```

## 환경
```bash
pip install torch transformers datasets openai tqdm langdetect peft   # 생성/필터/평가
pip install -U "llamafactory[torch,deepspeed,metrics]"                 # 학습
export DEEPSEEK_API_KEY=...   # R1 생성 / 반사실 CF
export OPENAI_API_KEY=...     # GPT-5 judge (선택)
```
> **GPU**: Blackwell(RTX 5090/5080)은 `torch==2.11.0+cu128`. 8B full FT 는 A100 80GB×2(ZeRO-3),
> 단일 GPU 는 LoRA(QLoRA/8bit).

## 파이프라인
```bash
# 1) R1 trace 생성 (train, fewshot)
python src/generation/generate_traces.py --model deepseek-r1 --total -1 --split train \
  --prompt-mode fewshot --workers 16 --output data/train_cot_fewshot.jsonl
# 2) 반사실 CF 생성 (C3 신호)
python src/filters/counterfactual_adapter.py --total -1 --split train --workers 16 --output data/train_cf.jsonl
# 3) SFT 데이터 build (no-filter / correct-only)
python src/dataset/build_arms.py --input data/train_cot_fewshot.jsonl --output data/train_nofilter.jsonl --answer predicted
python src/dataset/build_arms.py --input data/train_cot_fewshot.jsonl --output data/train_correct.jsonl --answer predicted --correct-only
# 4) 학습 (LoRA)
llamafactory-cli train configs/train.yaml
# 5) 평가 (test, base/학습본)
python src/eval/evaluate.py --model Qwen/Qwen3-8B [--lora output/...] --split test --output results/eval.jsonl
```

## 라이선스
KorMedMCQA: cc-by-nc-2.0 → Non-Commercial 전파. 연구·교육 목적.
