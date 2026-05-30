# Counterfactual Answer Probing (Inverse MATCHA) — ⑧ 제안 층위 필터

> 현재 코드 버전 기준 문서. 구현은
> [precompute.py](precompute.py) · [judge.py](judge.py) · [run.py](run.py),
> 하이퍼파라미터는 [configs/pipeline_config.yaml](../../configs/pipeline_config.yaml) `proposal_filters.counterfactual`.

## 1. 한 줄 요약

원본 CoT가 **정답**을 정당화하는 강도와, R1이 **틀린 답**을 정당화하느라 새로 쓴 CoT의
정당화 강도를 비교한다. 둘이 비슷하면(= 어떤 답이든 그럴듯하게 정당화됨) "답에 무관한
사후합리화(post-hoc rationalization)"로 보고 샘플을 탈락시킨다.

핵심 가설(decoupling): **좋은 추론은 답에 종속적**이다. 정답이 바뀌면 정당화도 무너져야
정상이다. 틀린 답도 정답만큼 매끄럽게 정당화된다면, 그 CoT는 추론이 아니라 사후 포장이다.

이 필터는 `think_final_divergence`(⑥) 위에 얹는 **제안 층위(proposal layer)** 필터로,
Judge Δ/δ 어블레이션 대상이 아니라 **C2 vs C3 다운스트림 Student 정확도**로만 평가된다
(README §2 파이프라인 아키텍처 참조).

## 2. 2단계 실행 구조

이 필터는 **무거운 사전계산(API 호출)** 과 **가벼운 판정**이 분리되어 있다.

```
① precompute.py  (DeepSeek-R1 재생성 + judge 채점, 느림/비쌈)
       │  metadata["counterfactual"] = {orig_score, counterfactuals:[{cf_score, cf_cot, ...}]}
       ▼
② run.py         (추가 API 호출 없이 gap 계산 → 통과/탈락 판정, 빠름)
```

### ① precompute.py — 점수 사전계산

각 샘플에 대해:

1. **wrong choice 선택** — gold를 제외한 선지에서 `k`개(default 1) 비복원 추출.
   `seed`로 순차 결정 → 병렬 실행에도 재현성 보장
   ([`_pick_wrong_indices`](precompute.py)).
2. **CF CoT 생성** — `TeacherRegenerator`가 R1(`deepseek-reasoner`)에게
   **"정답은 {wrong}다. 그 답이 왜 맞는지 4단계로 추론하라"**는 강제 프롬프트
   (`_REGEN_PROMPT_FORCED`)를 보내 가짜 CoT를 받아온다.
   - R1의 진짜 내부추론(`reasoning_content`)을 `<think>...</think>`로 감싸 사용.
     content 안에 모델이 끼워 넣은 `<think>`는 중복 방지를 위해 제거.
3. **정당화 강도 채점** — `judge.score_convincingness`로 1~5점:
   - **orig_score** = 원본 CoT를 **target=gold**로 채점
   - **cf_score** = CF CoT를 **target=wrong**로 채점
4. 결과를 `metadata["counterfactual"]`에 적재.

**판정 방식: 별도 수식이 아니라 프롬프트 기반 LLM-as-a-Judge.** judge 백엔드는
`UPSTAGE_API_KEY`가 있으면 Solar Pro 3, 없으면 DeepSeek-Chat
([`_make_judge`](precompute.py)).

#### `--regen-only` 모드 (F6 1단계)

채점(judge)을 생략하고 `cf_cot` 재생성만 수행한다. judge 키 불필요, `cf_score`/`orig_score`는
`null`로 저장. 채점은 별도 단계(GPT-5 등)에서 수행하는 워크플로우용.

#### 체크포인트

완료 샘플을 `<output>.ckpt.jsonl`에 인덱스별 append. 재실행 시 끝난 인덱스를 건너뛴다
(장시간 R1 재생성 중 Colab 끊김 등 대비). 단, **성공한 샘플만** 저장하여
(`_is_good`: 에러 없고 모든 cf가 `cf_cot` 보유) 실패분은 재실행 시 재시도된다.

## 3. 채점 루브릭 — judge.py `_CF_RUBRIC`

judge에게 **2단계 평가**를 시킨다. 채점 대상은 CoT의 `<think>` 블록 내부 추론만
(`_extract_internal_reasoning`; 없으면 전체 CoT fallback).

### 1단계 — `supported_letter` 판정 (라벨 사기 방지)

> 본문의 *임상적 내용*(기전·해부·약리·감별 배제)만 보고, 추론이 **실제로 도달하는** 선지를
> 하나 고른다. 맨 끝의 `"정답: X"` 선언 줄은 **무시**한다.

R1이 본문에서는 D를 논증해놓고 끝줄만 "정답: B"라고 적는 경우, judge가 끝줄만 보고
속지 않도록 하는 안전장치. 본문 논증이 가리키는 선지가 `supported_letter`다.

### 2단계 — target 지지 강도 1~5점

- `supported_letter ≠ target` → prose가 아무리 유창해도 **무조건 1점**
- `supported_letter == target`일 때만:

  | 점수 | 기준 |
  |---|---|
  | 5 | target에 결정적 임상 근거 + 모든 distractor 명시적 배제 |
  | 4 | 강한 정당화 + 주요 distractor 배제 |
  | 3 | 합리적 정당화 있으나 distractor 배제가 약함/일반적 |
  | 2 | 결론은 명시했으나 임상 근거 빈약(재진술 수준) |
  | 1 | target 정당화 사실상 실패 |

출력은 JSON `{"supported_letter", "matches_target", "score", "reason"}` 강제 →
`parse_geval_response`로 `score`만 추출. `n_reps`번 반복 후 정수 점수들의 **평균(mean)**이
최종 `orig_score`/`cf_score`. 파싱 실패 시 최대 3회 재시도.

## 4. 판정 로직 — run.py

`precompute` 결과를 읽어 **추가 API 호출 없이** 통과/탈락을 결정한다.

### 주 신호: gap

```
gap = orig_score - max(cf_score)
```

`gap < gap_threshold`(default **1.5**)이면 "어떤 답이든 비슷한 강도로 정당화됨" =
사후합리화 의심.

### 보조 신호: hedge_gap

CF CoT는 R1이 *틀린 답을 옹호하느라 쓴* 글이다. 추론이 답에 종속적이라면 R1은 wrong 답
정당화에 망설임 표지("하지만", "다시 생각", "확실하지" 등)를 더 많이 쓴다.

```
hedge_gap = orig_hedge_rate - min(cf_hedge_rate)   # <think> 블록 내부에서 측정
```

- **0 근처** → 양쪽 다 hedge 없음 = 자신 있게 정당화 = 사후합리화 의심
- **큰 음수** → CF가 hedge 폭증 = R1이 wrong 답 옹호에 망설임 = **robust**(rescue 대상)

hedge rate는 `think_final_divergence`의 `_hedge_rate_per_100words` / `_KO_HEDGE_PATTERNS`를
재사용하며, 측정 위치는 `<think>` 블록 내부(정형 4-step 출력은 hedge가 거의 안 잡힘).

### 판정 규칙

1. **독립 안전망 (OR, 최우선)** — `min_orig_score`(default 2.0)가 설정되어 있고
   `orig_score < min_orig_score`면, gap/hedge와 무관하게 즉시 탈락.
   원본 CoT가 gold조차 거의 정당화 못 하는 샘플 제거. `null`이면 비활성.

2. **`hedge_gap_threshold`가 `None`** → **gap 단독 판정**
   `gap < gap_threshold`면 탈락.

3. **`hedge_gap_threshold`가 설정됨 (default 0.5)** → **AND 룰 (보수적)**
   `gap < gap_threshold` **AND** `hedge_gap >= hedge_gap_threshold`일 때만 탈락.
   gap 단독은 judge가 3.0 plateau에 몰려 over-reject 위험이 있으므로, hedge가
   사후합리화를 확증할 때만 탈락. `hedge_gap << 0`(robust)이면 gap이 작아도 rescue.

4. **precompute 누락 / cf_score 전부 None** → `on_missing` 정책(`"pass"` | `"fail"`, default `pass`).

판정 후 `cf_max`, `gap`, `hedge`(상세 신호)를 `metadata["counterfactual"]`에 업데이트한다.

## 5. 하이퍼파라미터 (pipeline_config.yaml)

```yaml
proposal_filters:
  counterfactual:
    enabled: true
    gap_threshold: 1.5         # gap < 1.5 → judge 단독 의심 신호
    min_orig_score: 2.0        # orig_score < 2.0 → 독립 안전망(OR) 탈락 (1~5 스케일, null=비활성)
    hedge_gap_threshold: 0.5   # AND-rule: gap 의심 AND hedge_gap>=0.5 일 때만 탈락
    on_missing: "pass"         # precompute 누락 시: "pass" | "fail"
```

탈락률 참고(n=258 실측): gap 단독 56.2% → AND-rule(0.5) ~23% (33%p 완화).
`hedge_gap >= +0.5`는 분포의 상위 25.6%로, 사후합리화 신호가 명확히 양수일 때만 탈락 처리.

## 6. 사용법

```powershell
# 1단계 — 사전계산 (R1 재생성 + judge 채점). 워커 8~16 권장, rate-limit 초과 시 낮춤
python -m filters.counterfactual.precompute `
  --input  data/filtered/02_after_structure.json `
  --output data/filtered/counterfactual_output_precomputed.json `
  --k 1 --workers 8 --judge-reps 1

# (선택) regen-only — 채점은 별도 단계에서. judge 키 불필요
python -m filters.counterfactual.precompute --input ... --output ... --regen-only

# 2단계 — 판정 필터 (추가 API 호출 없음)
python -m filters.counterfactual.run `
  --input  data/filtered/counterfactual_output_precomputed.json `
  --output data/filtered/08_after_counterfactual.json
```

### 필요한 환경변수

| 변수 | 용도 |
|---|---|
| `DEEPSEEK_API_KEY` (또는 `MATCHA_TEACHER_API_KEY`) | CF CoT 재생성 (R1) |
| `UPSTAGE_API_KEY` | judge 백엔드 = Solar Pro 3 (있으면 우선) |
| (없으면 DEEPSEEK_API_KEY) | judge 백엔드 = DeepSeek-Chat fallback |

`TeacherRegenerator` 추가 오버라이드: `MATCHA_TEACHER_BASE_URL`(default `https://api.deepseek.com`),
`MATCHA_TEACHER_MODEL`(default `deepseek-reasoner`).

## 7. metadata 스키마

```jsonc
metadata["counterfactual"] = {
  "k": 1,
  "orig_target_letter": "C",
  "orig_score": 4.0,                  // 원본 CoT의 gold 정당화 강도 (regen_only면 null)
  "orig_judge_runs": [4],
  "counterfactuals": [
    {
      "target_idx": 1, "target_letter": "B",
      "cf_cot": "<think>...</think>\n[Step 1]...정답: B",
      "cf_score": 3.0,               // wrong 답 정당화 강도 (regen_only면 null)
      "judge_runs": [3],
      "regen_model": "deepseek-reasoner",
      "error": null
    }
  ],
  // ↓ run.py가 판정 시 추가
  "cf_max": 3.0,
  "gap": 1.0,
  "hedge": {
    "measured_on": "think_block",
    "orig_hedge_rate": 0.0, "cf_hedge_min": 0.0, "cf_hedge_max": 0.0,
    "cf_hedge_mean": 0.0, "hedge_gap": 0.0,
    "per_cf": [ { "target_letter": "B", "cf_hedge_rate": 0.0, "hedge_gap": 0.0 } ]
  }
}
```
