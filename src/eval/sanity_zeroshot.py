"""Sanity zero-shot — base / R1-Distill (구현 예정 / TODO).

스펙 §9 / 연구계획 §6.4. 성능 경쟁이 아니라 SFT 작동·벤치 비포화 점검용 하한.
- base Qwen3-8B (zero-shot/CoT-prompt): C1 이 이보다 분명히 높아야 distillation 작동.
- DeepSeek-R1-Distill-Qwen-8B (zero-shot): 절대 위치 참조점.
(현재 evaluate.py 로 --model 만 바꿔 측정 가능 — 전용 래퍼는 추후.)
"""
raise NotImplementedError("sanity zero-shot 래퍼 — 구현 예정 (스펙 §9)")
