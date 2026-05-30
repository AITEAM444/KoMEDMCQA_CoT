# KoMED — Korean Medical CoT Distillation & Counterfactual Faithfulness Filter

DeepSeek-R1 교사로 KorMedMCQA train 에 long-CoT(reasoning) trace 를 생성하고, **반사실 추론
충실성 필터(Inverse-MATCHA)**로 "답을 바꿔도 똑같이 그럴듯하게 정당화하는 사후합리화 CoT"를
걸러낸 뒤 Qwen3-8B student(LoRA)를 학습·평가한다.

**주장**: 정답이 맞는 trace 안에서 범용 LLM Judge는 변별력을 잃지만, 반사실 프로빙은 *답을
바꿨을 때의 정당화 강도 격차*라는 독립축에서 사후합리화 trace 를 잡아낸다. 더 적은 데이터로
student 정확도를 유지·향상시키는 것이 증거.

비교군: **C0**(무필터) · **C1**(표준 baseline) · **C2**(C1+범용 Judge) · **C-rand**(C1 무작위 축소) · **C3**(C1+반사실).
핵심: C3≥C1 / C3>C-rand / C3>C2.

## 빠른 시작 (실행 순서)

```bash
# 0) 설치 + 키
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch transformers datasets openai tqdm langdetect peft
pip install -U "llamafactory[torch,deepspeed,metrics]"
export DEEPSEEK_API_KEY=...   # 생성(R1)
export OPENAI_API_KEY=...     # judge(GPT-5)

# 1) 전체 자동 (1~11단계, 산출물 있으면 건너뜀)
bash scripts/run_pipeline.sh
```

수동으로 단계별 실행 순서 (자세한 명령은 아래 **파이프라인** 절):

| 순서 | 무엇 | 실행 파일 / 명령 |
|---|---|---|
| 1 | R1 trace 생성 | `generate_traces.py` |
| 2 | C0/C1 판정·통합 | `build_arms.py unified` |
| 3 | C3 반사실: 생성→채점→merge | `counterfactual_adapter.py` → `precompute` → `build_arms.py merge-c3` |
| 3.5 | **dev 임계값 보정**(선택, 권장) | `calibrate.py --auto` → `verify_thresholds.py` |
| 4 | C2 범용 judge: 채점→merge | `judge_general.py` → `build_arms.py merge-c2` |
| 5 | C-rand 수량 통제 | `build_arms.py make-crand` |
| 6 | 학습 (arm×seed) | `train_lora.py` |
| 7 | 평가 (test) | `evaluate.py` |
| 8 | 통계 (mean±std/CI/McNemar) | `stats.py` |
| 9 | 리포트 | `make_report.py` |

> 모든 데이터는 `data/unified.jsonl` 한 파일에 누적되고, arm 은 학습 직전에 슬라이스된다.
> **test split 로는 생성·보정하지 않는다**(평가 전용).

## 절대 규칙
- **test split 에는 Teacher CoT 를 생성하지 않는다** — 최종 평가(정확도)에만. calibration/early-stop 은 dev.

## 핵심 개념 (먼저 읽기)

### 1) 5개 비교군(arm)
모든 arm 은 **같은 R1 trace 풀**에서 출발하고, 필터만 다르게 적용한 부분집합이다.
| arm | 구성 | 역할 |
|---|---|---|
| **C0** | 필터 없음(raw) | 하한(lower bound) |
| **C1** | 표준 baseline ①정답 ②가독성 ③형식/파싱 ④길이 | 일반적 정제 baseline |
| **C-rand** | C1 통과분에서 무작위 \|C3\|개 | **크기 통제군** — 이득이 "크기 축소"가 아님을 보임 |
| **C2** | C1 + 범용 LLM Judge 상위 \|C3\|개 | **대조군** — "일반 채점으론 못 잡음"을 보임 |
| **C3** | C1 + 반사실 필터(제안) | **제안 방법** |

핵심 비교: **C3 ≥ C1**(H2) · **C3 > C-rand**(H2 통제) · **C3 > C2**(H3, 같은 크기).

### 2) 두 개의 judge — 헷갈리기 쉬우니 구분
| | **C2 judge** ([judge_general.py](src/filters/judge_general.py)) | **C3 judge** ([counterfactual/judge.py](filters/counterfactual/judge.py) `_CF_RUBRIC`) |
|---|---|---|
| 목적 | 대조군: 범용 품질 채점 | 제안: 반사실 정당화 강도 |
| 무엇을 보나 | 풀이 글 자체 품질 (**답은 안 바꿈**) | 답을 **오답으로 바꿔도** 똑같이 정당화되나 |
| 기준 | coherence / justification / clarity | anti-label-cheating(`supported_letter` 먼저 판정 → target 불일치면 강제 1점) |
| 척도 | **1~5** | **1~5** |
| 모델 | GPT-5 (`OPENAI_API_KEY`) | GPT-5 (`OPENAI_API_KEY`) |
| 산출 | `signals.c2_score` → 상위 \|C3\| = **C2** | `gap = orig_score − max(cf_score)` → 작으면 탈락 = **C3** |

### 3) 신호(signals) — `data/unified.jsonl` 에 누적
- **orig_score** (1~5): 원본 CoT 가 정답(gold)을 정당화하는 강도.
- **cf_score** (1~5): cf-CoT 가 **오답**을 정당화하는 강도.
- **gap = orig_score − max(cf_score)**: 작으면(기본 <1.5) "어떤 답이든 비슷하게 정당화" = **사후합리화 의심(탈락)**, 크면 답에 종속적 = 건강한 CoT.
- **hedge_gap**: 망설임 표지("하지만/다시 생각/wait…") 비율 차이 — judge 와 독립적인 보조 신호(AND-rule).
- **c2_score** (1~5): 범용 품질 점수(C2 선별용).

### 4) 반사실 필터의 2단계 구조 (비용 분리)
- **precompute**(1단계): cf-CoT 생성 + judge 채점 등 **모든 비싼 API 호출**을 여기서 끝내고 점수를 데이터에 박는다.
- **run**(2단계): 박아둔 점수로 **산수만** 해서 통과/탈락 판정 → 임계값(gap 1.5 등)을 바꿔 여러 번 돌려도 비용 0.

---

## 디렉터리 & 모듈 역할

```
src/generation/   trace 생성   |  src/filters/  필터·judge  |  src/dataset/  arm 빌드
src/train/        학습          |  src/eval/     평가·통계    |  filters/      반사실 패키지(기여)
utils/ configs/ scripts/ data/ results/
```

**① 생성 (Teacher = DeepSeek-R1)**
| 파일 | 역할 | 출력 |
|---|---|---|
| [src/generation/generate_traces.py](src/generation/generate_traces.py) | KorMedMCQA train 에 R1 long-CoT 생성(영어 instruction + 분야별 한국어 fewshot, R1 `reasoning_content` 캡처, 체크포인트·병렬) | `train_cot_fewshot.jsonl` |
| [src/filters/counterfactual_adapter.py](src/filters/counterfactual_adapter.py) | 오답 1개를 정답이라 강제 → 그 오답을 정당화하는 cf-CoT 생성(생성 인프라 재사용, seed 로 오답 재현) | `train_cf.jsonl` |

**② 필터 (arm 판정)**
| 파일 | 역할 |
|---|---|
| [src/filters/standard.py](src/filters/standard.py) | **C1** 표준 baseline ①~④ 게이트(reasoning 기준 판정) |
| [src/filters/judge_general.py](src/filters/judge_general.py) | **C2** judge — 범용 품질 1~5 채점(GPT-5) → `signals.c2_score` |
| [filters/counterfactual/precompute.py](filters/counterfactual/precompute.py) | **C3 1단계** — 원본·cf CoT 를 `_CF_RUBRIC` 로 채점(GPT-5). 모든 API 호출 종결 |
| [filters/counterfactual/judge.py](filters/counterfactual/judge.py) | `_CF_RUBRIC` 정의 + 채점 함수(anti-label-cheating) |
| [filters/counterfactual/run.py](filters/counterfactual/run.py) | **C3 2단계** — gap/hedge_gap/min_orig + AND-rule 로 통과/탈락(산수만) |
| [filters/base.py](filters/base.py) | 필터 베이스 클래스(`run`/`run_from_file`) |

**③ arm 빌드 (데이터 통합 — 한 파일에 누적)**
| 서브커맨드 ([build_arms.py](src/dataset/build_arms.py)) | 역할 |
|---|---|
| `unified` | trace ↔ KorMedMCQA(질문·선지·gold) join + **C0/C1** 판정 → `unified.jsonl` |
| `merge-c3` | 채점 결과에 run.py 판정 적용 → `filters.C3` |
| `merge-c2` | `c2_score` 상위 \|C3\|개 → `filters.C2` (크기 매칭) |
| `make-crand` | C1 풀에서 랜덤 \|C3\|개 → `filters.C-rand` |
| `export` | 특정 arm → SFT 형식(`messages`) jsonl |

**④ 학습 · 평가 · 통계**
| 파일 | 역할 | 출력 |
|---|---|---|
| [src/train/train_lora.py](src/train/train_lora.py) | arm×seed export→dataset 등록→LLaMA-Factory LoRA 학습(질문 마스킹, seq 4096) | `output/qwen3-8b-<arm>-s<seed>/` |
| [src/eval/evaluate.py](src/eval/evaluate.py) | KorMedMCQA test 정답률(전체/과목별, resume) | `eval_*.jsonl` |
| [src/eval/calibrate.py](src/eval/calibrate.py) | **dev** 반사실 임계값 스윕 표 + `--auto`(Otsu/GMM 비지도 컷 + 이봉성 진단 + 검증강도 권고) | `calibrate_dev.md` |
| [src/eval/verify_thresholds.py](src/eval/verify_thresholds.py) | Path 3 검증 — 후보 컷 train→**dev ACC**(plan) + 분산체크·결정(analyze, 자동이동 X) | `verify_thresholds.json` |
| [src/eval/stats.py](src/eval/stats.py) | seed mean±std + 부트스트랩 95% CI + McNemar 짝지은 검정 | `stats.json` |
| [src/eval/sanity_zeroshot.py](src/eval/sanity_zeroshot.py) | base/distill zero-shot 하한 점검(학습 작동 확인) | `sanity_*.jsonl` |
| [src/eval/compare_judges.py](src/eval/compare_judges.py) | judge 모델 동등성(ref vs cand) 검증 | `judge_compare.json` |
| [scripts/make_report.py](scripts/make_report.py) | arm 크기 + 신호 분포 + 통계/가설 점검 → 마크다운 | `report.md` |
| [scripts/run_pipeline.sh](scripts/run_pipeline.sh) | 1~11 단계 end-to-end 오케스트레이션 | — |

**⑤ 유틸 · 설정**
| 파일 | 역할 |
|---|---|
| [utils/schema.py](utils/schema.py) | `CoTSample` 데이터클래스 + `FilterName` |
| [utils/data_loader.py](utils/data_loader.py) | jsonl/json ↔ `CoTSample` 로더 |
| [src/data/load_kormedmcqa.py](src/data/load_kormedmcqa.py) | KorMedMCQA 선지 join 유틸 |
| [configs/pipeline_config.yaml](configs/pipeline_config.yaml) | C1 임계값(`c1_filters`) + C3 `gap/hedge/min_orig` |
| `configs/train.yaml`, `qwen3_8b_*.yaml`, `ds_zero3.json`, `dataset_info.json` | LLaMA-Factory 학습/데이터 설정 |
| `data/`, `results/` | 데이터·체크포인트 (`.gitignore`) |

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

#### (보정) dev 로 임계값 고르기 — §3 "dev 로 임계값 보정 (test 로 맞추면 부정행위)"
`gap/hedge/min_orig` 기본값(1.5 / 0.5 / 2.0)은 [configs/pipeline_config.yaml](configs/pipeline_config.yaml)에 박혀 있다.
이를 **dev** 로 정한다 — **gap 만 데이터로 자동 결정, hedge 는 보조로 둔다**(두 축 동시 자동컷은
자유도가 늘고 이봉성 해석이 꼬여 논리가 약해짐). 방식은 **Path 3(비지도 컷 + ACC 수렴 확인)**:

```bash
# 0) dev CF 생성 → 채점 (train 과 동일 절차, --split dev)
python src/filters/counterfactual_adapter.py --total -1 --split dev --workers 16 --output data/dev_cf.jsonl
python -m filters.counterfactual.precompute --input data/dev_cf.jsonl --output data/dev_cf_judged.json --k 1 --workers 16

# 1) 비지도 컷 + 이봉성 진단 + 검증강도 권고 (학습 없음) — 이게 '결정'
python src/eval/calibrate.py --cf data/dev_cf_judged.json --auto --output results/calibrate_dev.md
#   → Otsu/GMM 컷, 분리도(Cohen's d), 이봉 여부, 검증 후보 컷 목록, "ACC 검증 얼마나 해야 하나"

# 2) 후보 컷 몇 개만 dev ACC 로 *확인* (풀스윕 X). 학습 포함이라 무거움 → 명령 생성 후 실행
python src/eval/verify_thresholds.py plan --unified data/unified.jsonl --cf data/cf_judged.json \
  --candidates 1.0 1.5 1.9 --unsup-cut 1.5 --seeds 42 43 --out-sh results/verify.sh
bash results/verify.sh        # merge-c3(--gap)→export→train→dev eval, 마지막에 analyze 자동 실행

# (analyze 단독 재실행)
python src/eval/verify_thresholds.py analyze \
  --cut 1.0 results/devacc_g1_s*.jsonl --cut 1.5 results/devacc_g1p5_s*.jsonl \
  --cut 1.9 results/devacc_g1p9_s*.jsonl --unsup-cut 1.5
# → 고른 값을 configs/pipeline_config.yaml 의 counterfactual 에 반영하고 §3-3(merge-c3) 재실행
```

포지셔닝(중요 — 과장 금지):
- **진짜 방어 논리는 "dev 에서 정하고 test 는 안 건드렸다"** 이다. 비지도 컷이 "객관적이라 더 옳다"가 아니다.
- 비지도 컷(Otsu/GMM)은 *gap 분포의 분리점*이지 *ACC 최적 컷*이 아니다(feature 분리점 ≠ task 결정경계). 그 가치는 **작은 dev 라벨에 과적합하지 않으려는** robustness.
- 그래서 **확인이지 튜닝이 아니다**: 비지도 컷을 결정으로 두고 ACC 로 *수렴만* 확인. ACC-best 로 컷을 옮기면 그건 (거친) dev 튜닝이므로 그렇게 *명시*해야 한다(analyze 가 경고).
- **분산 체크**: 후보 컷들 ACC 차이가 seed noise 보다 작으면 "비지도 컷이 다른 것만큼 좋다"가 정직한 결론(이것도 방어됨).
- **분리도가 검증강도를 조절**: 분리 강하면 골짜기 평평→ACC 둔감(가볍게), 약하면 컷 민감→ACC 확인이 값짐(seed↑). calibrate `--auto` 가 강도를 권고한다.
- test split 로는 절대 보정하지 않는다.

### 4) C2 — 범용 LLM Judge(대조군) — §6.2 / H3
```bash
# 4-1) src/filters/judge_general.py : C1 통과 trace 를 '범용 품질' 루브릭(coherence/
#      justification/clarity)으로 1~5 채점(GPT-5) → signals.c2_score. (반사실과 달리 답을 안 바꿈)
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
