#!/usr/bin/env bash
set -uo pipefail

cd /workspace/KoMEDMCQA_CoT
mkdir -p results logs

MODEL="Qwen/Qwen3-8B"
SEEDS="42 43 44"
HF_REPO="AIteam4/KoMEDMCQA-LoRA"
REPO_DIR="eval/C0"

for s in $SEEDS; do
  OUT="results/eval_C0_s${s}.jsonl"
  LOG="logs/eval_C0_s${s}.log"

  echo "=========================================="
  echo "[run] C0 seed=$s 시작 $(date '+%F %T')"
  echo "=========================================="

  python -u src/eval/evaluate.py \
    --model "$MODEL" \
    --lora "./c0_adapters/qwen3-8b-c0-s${s}" \
    --split test \
    --output "$OUT" 2>&1 | tee -a "$LOG"

  echo "[push] $OUT -> $HF_REPO/$REPO_DIR/"
  hf upload "$HF_REPO" "$OUT" "$REPO_DIR/$(basename "$OUT")" \
    --repo-type model \
    --commit-message "eval C0 seed=$s $(date '+%F %T')"

  echo "[run] C0 seed=$s 완료 + push $(date '+%F %T')"
done

echo "[done] 전체 완료 -> https://huggingface.co/$HF_REPO/tree/main/$REPO_DIR"
