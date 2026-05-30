# KoMED — Korean Medical CoT Distillation & Filtering

한국어 의료 CoT(Chain-of-Thought) distillation 파이프라인. Teacher(DeepSeek-R1)가 KorMedMCQA
train split 에 추론(reasoning) trace 를 생성하고, 단계적 필터링으로 저품질 샘플을 제거한 뒤
Student(Qwen3-8B)를 학습·평가한다.

**핵심 주장**: counterfactual 기반 추론충실성 필터(F6)가 **한국어 의료 CoT distillation**에서
효과를 보인다 — C2(필터) vs C3(+F6) 다운스트림 Student 정확도로 검증.

> 도메인·과제·평가는 한국어(KorMedMCQA). 내부추론(reasoning_content)은 영어 instruction 으로
> 품질을 끌어올려 생성하므로 언어가 섞일 수 있으며, 이는 의도된 설계다(품질 우선, 필터로 충실성 보장).

## 절대 규칙
- **test split 에는 Teacher CoT 를 절대 생성하지 않는다.** test 는 최종 평가(정확도)에만 사용.
- 생성·필터·학습은 train split, 검증은 dev split.

## 구조
```
configs/
  filters.yaml        # 필터 파라미터(C1 ~ C2 judge)
  train.yaml          # Qwen3-8B LoRA 하이퍼파라미터 (no-filter baseline)
  ds_zero3.json       # full FT 용 DeepSpeed ZeRO-3
  dataset_info.json   # LLaMA-Factory 데이터 등록
src/
  generation/generate_traces.py   # R1 trace 생성 (영어 instruction + 분야별 한국어 fewshot, reasoning_content 캡처)
  filters/                         # 표준 필터(correctness/structure/kppl/step_coverage/answer_consistency) + judge
    counterfactual_adapter.py      # F6: 오답 강제 → cf 추론 생성 (Inverse-MATCHA 계열)
  dataset/build_arms.py            # 비교군(C0~C3) SFT 데이터 export (reasoning_content + 정답)
  eval/evaluate.py                 # KorMedMCQA test 정확도 (전체/과목별, resume 내장)
  eval/compare_judges.py           # judge 모델 동등성 비교 (ref vs cand)
  data/load_kormedmcqa.py          # KorMedMCQA 선지 join 유틸
filters/ , utils/                  # 작동 패키지(필터 본체·스키마·로더) — 위 src 코드가 import
scripts/                           # run_pipeline / run_ablation / sweep
data/ , results/                   # 데이터·체크포인트(.gitignore, 대용량)
```

## 환경
```bash
pip install -r requirements.txt   # torch(+cu), transformers 4.5x, datasets, openai, peft, vllm 등
# 학습: pip install -U "llamafactory[torch,deepspeed,metrics]"
export DEEPSEEK_API_KEY=...   # R1 생성 / counterfactual
export OPENAI_API_KEY=...     # GPT-5 judge (선택)
```
> **GPU 주의**: Blackwell(RTX 5090/5080, sm_120)은 `torch==2.11.0+cu128` 필요. transformers 는
> 4.5x 대. 8B full FT 는 단일 16~24GB 불가 → A100 80GB×2(ZeRO-3) 또는 LoRA(QLoRA/8bit).

## 재현 절차
```bash
# 1) Teacher trace 생성 (train, fewshot, reasoning_content)
python src/generation/generate_traces.py --model deepseek-r1 --total -1 --split train \
  --prompt-mode fewshot --workers 16 --output data/train_cot_fewshot.jsonl
# 2) counterfactual cf 생성 (F6 신호)
python src/filters/counterfactual_adapter.py --total -1 --split train \
  --workers 16 --output data/train_cf.jsonl
# 3) 비교군 데이터 build (no-filter / correct-only / ...)
python src/dataset/build_arms.py --input data/train_cot_fewshot.jsonl \
  --output data/train_nofilter.jsonl --answer predicted
# 4) 학습 (LoRA, no-filter baseline)
llamafactory-cli train configs/train.yaml
# 5) 평가 (test 정확도)
python src/eval/evaluate.py --model Qwen/Qwen3-8B --lora output/... --split test --output results/eval.jsonl
```

## 라이선스
Non-Commercial (NC). 연구·교육 목적에 한해 사용.
