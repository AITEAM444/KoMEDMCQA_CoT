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
  pipeline_config.yaml        # 필터 파라미터 (C1 c1_filters + C3 counterfactual gap/hedge/min_orig)
  train.yaml / qwen3_8b_*.yaml # Qwen3-8B LoRA/full SFT (LLaMA-Factory)
  ds_zero3.json               # full FT DeepSpeed ZeRO-3
  dataset_info.json           # LLaMA-Factory 데이터 등록
src/
  generation/generate_traces.py      # R1 trace 생성 (영어 instruction + 분야별 한국어 fewshot, reasoning_content 캡처)
  filters/standard.py                # C1 표준 baseline ①~④ (reasoning 기준)
  filters/counterfactual_adapter.py  # 반사실 CF 생성 (오답 강제 정당화, reasoning_content 신호)
  filters/judge_general.py           # C2 범용 LLM Judge 채점 (대조군) → signals.c2_score
  dataset/build_arms.py              # 통합 스키마 + arm(C0/C1/C2/C3/C-rand) merge·export
  eval/evaluate.py                   # KorMedMCQA test 정확도 (전체/과목별, resume 내장)
  eval/stats.py                      # seed mean±std + 부트스트랩 CI + McNemar
  eval/sanity_zeroshot.py            # base/distill zero-shot 하한 점검
  eval/compare_judges.py             # judge 모델 동등성 비교 (ref vs cand)
  train/train_lora.py                # arm × seed 학습 자동화 (LLaMA-Factory 래퍼)
  data/load_kormedmcqa.py            # KorMedMCQA 선지 join 유틸
filters/                             # 작동 패키지: counterfactual{precompute,run,judge}(기여) + base(의존)
scripts/run_pipeline.sh / make_report.py  # end-to-end 오케스트레이션 + 마크다운 리포트
utils/                               # CoTSample 스키마 + 데이터 로더
data/ , results/                     # 데이터·체크포인트 (.gitignore)
```

## 환경
```bash
pip install torch transformers datasets openai tqdm langdetect peft   # 생성/필터/평가
pip install -U "llamafactory[torch,deepspeed,metrics]"                 # 학습

# Teacher(R1) — trace 생성 + 반사실 CF 생성 (필수)
export DEEPSEEK_API_KEY=...
# C2/C3 judge 채점 — OpenAI GPT-5 (필수). 모델은 OPENAI_JUDGE_MODEL 로 override 가능(default gpt-5)
export OPENAI_API_KEY=...
# judge 동등성 검증(compare_judges.py)용 게이트웨이 (선택)
export GATEWAY_API_KEY=...
```
> **GPU**: Blackwell(RTX 5090/5080)은 `torch==2.11.0+cu128`. 8B full FT 는 A100 80GB×2(ZeRO-3),
> 단일 GPU 는 LoRA(QLoRA/8bit).

## 파이프라인

전체를 한 번에:
```bash
bash scripts/run_pipeline.sh          # 1~11 단계 전부 (산출물 있으면 건너뜀)
# 옵션: WORKERS=16 SEEDS="42 43 44" ARMS="C0 C1 C2 C3 C-rand" bash scripts/run_pipeline.sh
```

아래는 같은 흐름의 **단계별 수동 실행**. 각 단계 = `실행 파일 → 하는 일 → 산출물`.
데이터는 `data/unified.jsonl` 한 파일에 누적되며(필터 플래그 `filters.{C0,C1,C2,C3,C-rand}`
+ 신호 `signals.*`), arm 은 마지막에 슬라이스한다.

### 0) (선택) 학습이 작동할 환경인지 sanity 점검 — §6.4
```bash
# src/eval/sanity_zeroshot.py : 학습 전 base 모델 zero-shot 정답률(하한).
#   → 나중에 C1 학습본이 이보다 분명히 높아야 distillation 이 작동한 것.
python src/eval/sanity_zeroshot.py --split test --models Qwen/Qwen3-8B
```

### 1) Teacher(R1) trace 생성 — §4
```bash
# src/generation/generate_traces.py : KorMedMCQA train 3,401문항에 R1 long-CoT 생성.
#   영어 instruction + 분야별 한국어 fewshot, R1 의 reasoning_content 캡처. 체크포인트 내장.
#   ⚠ test split 은 절대 생성하지 않는다(평가 전용). dev 는 임계값 보정용으로만.
python src/generation/generate_traces.py --model deepseek-r1 --total -1 --split train \
  --prompt-mode fewshot --workers 16 --output data/train_cot_fewshot.jsonl
# → data/train_cot_fewshot.jsonl  (문항별 reasoning_content + 최종답)
```

### 2) 통합 스키마 빌드 + C0/C1 판정 — §6.1
```bash
# src/dataset/build_arms.py unified : trace 를 KorMedMCQA(질문·선지·gold)와 join 하고
#   C1 표준 baseline ①정답 ②가독성 ③형식/파싱 ④길이(src/filters/standard.py)를 판정.
python src/dataset/build_arms.py unified \
  --input data/train_cot_fewshot.jsonl --output data/unified.jsonl
# → data/unified.jsonl  (filters.C0=True 전부 / filters.C1=①~④통과 여부)
```

### 3) C3 — 반사실 필터(제안, 기여) — §2
```bash
# 3-1) src/filters/counterfactual_adapter.py : 정답 대신 '오답 하나를 강제'해 그 오답을
#      정당화하는 cf-CoT 를 R1 으로 생성(생성 인프라는 trace 생성과 동일). seed 로 오답 재현.
python src/filters/counterfactual_adapter.py \
  --total -1 --split train --workers 16 --output data/train_cf.jsonl

# 3-2) filters/counterfactual/precompute.py : 원본 CoT(target=gold)와 cf-CoT(target=오답)를
#      judge 로 1~5점 채점해 metadata.counterfactual 에 박는다(모든 API 호출이 여기서 끝).
#      judge 백엔드: OpenAI GPT-5 (OPENAI_API_KEY 필요).
python -m filters.counterfactual.precompute \
  --input data/train_cf.jsonl --output data/cf_judged.json --k 1 --workers 16

# 3-3) build_arms.py merge-c3 : filters/counterfactual/run.py 의 판정 로직(gap/hedge/
#      min_orig, AND-rule)을 적용해 filters.C3 = (C1통과 AND 반사실통과) 를 채운다.
python src/dataset/build_arms.py merge-c3 \
  --unified data/unified.jsonl --cf data/cf_judged.json --output data/unified.jsonl
# → unified 에 filters.C3 + signals.counterfactual(gap, orig_score, hedge ...) 적재
```

### 4) C2 — 범용 LLM Judge(대조군) — §6.2 / H3
```bash
# 4-1) src/filters/judge_general.py : C1 통과 trace 를 '범용 품질' 루브릭(coherence/
#      justification/clarity)으로 1~10 채점 → signals.c2_score. (반사실과 달리 답을 안 바꿈)
python src/filters/judge_general.py \
  --unified data/unified.jsonl --output data/unified.jsonl --workers 8

# 4-2) build_arms.py merge-c2 : c2_score 상위 N=|C3| 개를 filters.C2=True.
#      C3 와 '같은 크기'로 맞춰야 H3(같은 크기에서 반사실>일반judge) 비교가 공정.
python src/dataset/build_arms.py merge-c2 \
  --unified data/unified.jsonl --output data/unified.jsonl
```

### 5) C-rand — 수량 통제군 — §6.2 / H2 통제
```bash
# build_arms.py make-crand : C1 통과 풀에서 시드 고정 랜덤 |C3|개 → filters['C-rand'].
#   "이득이 크기 축소가 아니라 선별 품질에서 옴"을 통제. (merge-c3 다음에 실행)
python src/dataset/build_arms.py make-crand \
  --unified data/unified.jsonl --output data/unified.jsonl --seed 42
```
> 여기까지 끝나면 `data/unified.jsonl` 한 파일에 5개 arm 플래그가 모두 채워진다.

### 6) (선택) arm 데이터 export — 직접 학습/검수용
```bash
# build_arms.py export : 특정 arm 만 SFT 형식(messages)으로 슬라이스.
#   (train_lora.py 가 내부에서 자동 호출하므로 보통 직접 실행 불필요)
python src/dataset/build_arms.py export --input data/unified.jsonl --arm C3 \
  --output data/sft/train_C3.jsonl
```

### 7) 학습 — arm × seed, ≥3 seed — §5 / §6.5
```bash
# src/train/train_lora.py : 각 arm 을 export→dataset 등록→LLaMA-Factory(configs/train.yaml)로
#   LoRA 학습(질문 마스킹, seq 4096). 단일 run 노이즈 통제 위해 최소 3 seed.
python src/train/train_lora.py --unified data/unified.jsonl \
  --arms C0 C1 C2 C3 C-rand --seeds 42 43 44
#   먼저 명령만 확인:  ... --arms C3 --seeds 42 --dry-run
# → output/qwen3-8b-<arm>-s<seed>/  (LoRA adapter)
```

### 8) 평가 — test 정답률 — §6.3
```bash
# src/eval/evaluate.py : base+LoRA 를 KorMedMCQA test 3,009문항으로 평가(전체/과목별, resume).
for arm in C0 C1 C2 C3 C-rand; do for s in 42 43 44; do
  python src/eval/evaluate.py --model Qwen/Qwen3-8B \
    --lora output/qwen3-8b-$(echo $arm|tr A-Z a-z)-s$s \
    --split test --output results/eval_${arm}_s${s}.jsonl
done; done
```

### 9) 통계 — mean±std + CI + McNemar — §6.5
```bash
# src/eval/stats.py : arm 별 seed 평균±std + 부트스트랩 95% CI, arm 간 짝지은 McNemar 검정.
python src/eval/stats.py \
  --arm C0 results/eval_C0_s*.jsonl  --arm C1 results/eval_C1_s*.jsonl \
  --arm C2 results/eval_C2_s*.jsonl  --arm C3 results/eval_C3_s*.jsonl \
  --arm C-rand results/eval_C-rand_s*.jsonl \
  --mcnemar C3 C1 --mcnemar C3 C-rand --mcnemar C3 C2 \
  --output results/stats.json
#   H2: C3>C1, C3>C-rand   |   H3: C3>C2 (같은 크기)
```

### 10) 리포트 — §6
```bash
# scripts/make_report.py : arm 크기 + 신호 분포(gap/orig_score/c2_score) + 통계/가설 점검 →
#   results/report.md (H1 분석 대상 = C1통과 ∩ C3탈락 건수 포함).
python scripts/make_report.py \
  --unified data/unified.jsonl --stats results/stats.json --output results/report.md
```

### (선택) judge 모델 동등성 검증
```bash
# src/eval/compare_judges.py : 반사실 judge 를 ref 모델 대비 후보 모델로 교체해도
#   동일하게 작동하는지(supported_letter/reject 결정 일치율) 검증. GATEWAY_API_KEY 필요.
python src/eval/compare_judges.py \
  --input data/cf_judged.json --ref-model claude-opus-4-8 --cand-model gpt-5 \
  --output results/judge_compare.json
```

### 가설 ↔ 비교 매핑 (§6.2)
| 가설 | 비교 | 실행 단계 |
|---|---|---|
| H1 (별개의 에러를 잡음) | C1통과 ∩ C3탈락 분석 (학습 불필요) | 5 → 10 (report 의 H1 건수/신호분포) |
| H2 (성능 향상) | C3 > C1, C3 > C-rand | 7~9 (`--mcnemar C3 C1`, `C3 C-rand`) |
| H3 (judge로는 못 잡음) | C3 > C2 (같은 크기) | 4 → 7~9 (`--mcnemar C3 C2`) |

## 라이선스
KorMedMCQA: cc-by-nc-2.0 → Non-Commercial 전파. 연구·교육 목적.
