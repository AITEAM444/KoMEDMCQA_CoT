"""C2 — 범용 LLM Judge 필터 (구현 예정 / TODO).

스펙 §4 / 연구계획 §6.2. DC-CoT(2505.18759) 범용 품질 루브릭(coherence/correctness/clarity)으로
C1 통과 trace 를 점수화하고 임계값 미만 제거 → C2 데이터셋.
- judge 백엔드 교체 가능(config), n_reps 평균·재시도.
- 반사실 필터의 judge 와는 개념상 별개(범용 품질 vs 반사실 정당화 강도). API 인프라는 재사용 가능.
"""
raise NotImplementedError("C2 범용 Judge 필터 — 구현 예정 (스펙 §4)")
