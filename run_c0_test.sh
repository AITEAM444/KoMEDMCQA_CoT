#!/usr/bin/env bash
cd /workspace/KoMEDMCQA_CoT
git config user.name "J-Hyunwoo"
git config user.email "jhwoo582@gmail.com"

wait_for_gpu () {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 16000 ]; then break; fi
    echo "[wait] GPU free ${FREE}MiB < 16000 - 60s"
    sleep 60
  done
}

push_seed () {
  s=$1
  f="results/eval_C0_s$s.jsonl"
  if [ ! -s "$f" ]; then echo "[push skip] $f none"; return; fi
  git add -f "$f"
  git commit -m "C0 test result s$s" || true
  git pull --rebase origin main || true
  git push "https://${GH_TOKEN}@github.com/AITEAM444/KoMEDMCQA_CoT.git" main && echo "[push ok] s$s" || echo "[push fail] s$s"
}

for s in 42 43 44; do
  echo "===== C0 seed=$s start ====="
  wait_for_gpu
  python src/eval/evaluate.py --model Qwen/Qwen3-8B --lora ./c0_adapters/qwen3-8b-c0-s$s --split test --output results/eval_C0_s$s.jsonl || echo "[eval fail] s$s"
  push_seed $s
done
echo "=== ALL DONE ==="
