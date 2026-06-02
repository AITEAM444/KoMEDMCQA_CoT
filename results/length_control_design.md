# 길이 통제 재실험 설계 — 반사실 효과 ↔ trace 길이/degeneration 교란 분리

## 1. 동기 (앞선 분석 요약)
- 학습 trace 길이가 평가 degeneration(루프·무응답)을 좌우하고, degeneration 이 정확도를 좌우한다.
  - 출력 <3000자 77.9% vs >6000자 36.6% vs None 0%.
  - 학습 토큰 길이: C2 177(최단) < C3 203 < C-rand 215 ≈ C1 216.
- degen 제외 McNemar: **C2의 C3 대비 우위(원본 p=0.038)가 소멸(p=0.18)** → C2 우위는 추론이 아니라 길이/fluency.
- 따라서 arm 정확도 차이가 "필터의 추론 충실성"인지 "길이 교란"인지 **분리 필요**.
- `corr(tokens, gap) = -0.12` → gap 은 길이와 거의 독립 ⇒ 길이 매칭해도 gap 변별이 남아 통제가 깨끗함.

## 2. 통제군 (생성 완료)
`scripts/build_length_matched.py` (토큰길이 20-분위 층화 무작위, build_seed=42):
| arm | 정의 | 토큰 mean/median (target→통제) | 비교 목적 |
|---|---|---|---|
| **C-randL** | C1 풀에서 길이분포를 **C3** 에 맞춘 무작위 \|C3\|개 | 203→204 / 150→150 | **C3 vs C-randL** = 크기·길이 동일, 차이는 gap 선별뿐 → *반사실 신호의 순수 가치* |
| **C-randS** | C1 풀에서 길이분포를 **C2** 에 맞춘 무작위 \|C3\|개 | 177→182 / 147→147 | **C2 vs C-randS** = C2 우위가 "짧음" 때문인지 검정 |

산출물: `data/sft/train_C-randL.jsonl`, `train_C-randS.jsonl` (등록: `komed_crandl`, `komed_crands`).

## 3. 가설 (사전 등록 — p-hacking 방지)
- **주가설 H_main (1순위)**: degen 제외 정확도에서 **C3 ≤ C-randL** (반사실 선별은 길이 매칭 무작위 대비 이득 없음).
  - 검정: C3 vs C-randL paired McNemar (degen 제외, 3시드 과반). 일방.
- **기전확인 H_mech**: **C2 ≈ C-randS** (C2 우위는 길이로 설명됨).
- 보조: raw / degen제외 둘 다 보고. 다중비교는 Holm 보정.

## 4. 실행 절차
```powershell
# (0) 통제군 생성 — 완료됨
.venv\Scripts\python.exe scripts\build_length_matched.py

# (1) 학습 — 새 arm 2개 × 3 seed. train_lora 는 unified 재export 하므로 우회,
#     직접 llamafactory 호출(train.yaml 의 dataset/seed override).
foreach ($ds in @('crandl','crands')) {
  foreach ($s in 42,43,44) {
    llamafactory-cli train configs/train.yaml `
      dataset=komed_$ds dataset_dir=data/sft `
      output_dir=output/qwen3-8b-$ds-s$s seed=$s data_seed=$s
  }
}

# (2) 평가 — test 3009
foreach ($arm in @('crandl','crands')) {
  foreach ($s in 42,43,44) {
    .venv\Scripts\python.exe src\eval\evaluate.py --model Qwen/Qwen3-8B `
      --lora output/qwen3-8b-$arm-s$s `
      --split test --output results/test_$arm`_s$s.jsonl
  }
}

# (3) 통계 — raw + degen제외
.venv\Scripts\python.exe src\eval\stats.py `
  --arm C3 results/test_c3_s*.jsonl --arm C-randL results/test_crandl_s*.jsonl `
  --arm C2 results/test_c2_s*.jsonl --arm C-randS results/test_crands_s*.jsonl `
  --arm C-rand results/test_c-rand_s*.jsonl `
  --mcnemar C3 C-randL --mcnemar C2 C-randS `
  --output results/stats_lenctrl.json
# degen제외 버전: scripts/stats_degen_excluded.py 의 ARMS 에 crandl/crands 추가 후 재실행
```

## 5. 결정 규칙
- C3 vs C-randL 가 **비유의 + 효과크기 |Δ|<0.5pp** → "반사실 신호는 길이 통제 시 downstream 이득 없음"(정직한 null, 논문의 H2/H3 기각 강화).
- C3 > C-randL 유의 → 반사실 신호가 길이와 독립으로 기여(제안 방법 방어 가능).
- C2 ≈ C-randS → "C2 우위 = 길이 효과" 기전 확정.

## 6. ⚠ 검정력 한계 (반드시 인지)
- **처치 용량이 작다**: C3 와 같은 크기 통제군은 최대 \|C1∖C3\| = **298/2872 = 10.4%** 샘플만 다를 수 있다(C-randL 의 C3 겹침 91%). 즉 학습데이터의 ~10%만 바뀌는 개입 → 효과 희석.
- MDE(80% power, α=.05 양측):

  | seeds | 일반 arm(σ≈0.006) | C3(σ≈0.014) |
  |---|---|---|
  | 3 | 1.37pp | 3.20pp |
  | 5 | 1.06pp | 2.48pp |
  | 8 | 0.84pp | 1.96pp |
  | 12 | 0.69pp | 1.60pp |

- 관측 효과(C-rand−C3 ≈ 0.85pp)를 잡으려면 **≥8 seed**, C3 의 큰 시드분산(σ=0.014) 고려 시 **사실상 10~12 seed** 필요.
- **권고**: 통제군은 우선 5 seed 로 돌려 점추정·방향 확인 → 경계면이면 8~12 seed 로 확장. 비용상 5 seed 면 "방향성 일관 + null 못 기각" 까지는 정직하게 보고 가능.

## 7. 더 민감한 대안(선택)
- **고용량 설계**: gap 임계값을 낮춰(예 gap<2.5) C3 탈락분을 키우면(298→더 큼) 처치 용량↑ → 검정력↑. 단 크기 통제 위해 C-randL 도 동일 크기로 재생성.
- **동일 길이에서 신호 강도 스윕**: gap 상·하위만 교체한 두 통제군으로 dose-response 확인.
