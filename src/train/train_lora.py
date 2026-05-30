"""Student LoRA 학습 — Qwen3-8B, 질문 마스킹, ≥3 시드 (구현 예정 / TODO).

스펙 §8 / 연구계획 §5. 현재는 LLaMA-Factory 설정(configs/train.yaml)으로 학습.
이 스크립트는 arm·시드 루프를 묶어 자동화하기 위한 자리(예정).
- loss: trace+answer 토큰만 (질문 label -100). seq 4096. arm 당 ≥3 시드.
"""
raise NotImplementedError("arm×시드 학습 자동화 — 구현 예정 (스펙 §8). 현재는 configs/train.yaml + llamafactory-cli")
