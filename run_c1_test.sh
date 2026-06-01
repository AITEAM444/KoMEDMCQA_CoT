#!/usr/bin/env bash
cd /workspace/KoMEDMCQA_CoT
git config user.name "J-Hyunwoo"
git config user.email "jhwoo582@gmail.com"

push_seed () {
  s=$1
  f="results/eval_C1_s$s.jsonl"
  if [ ! -s "$f" ]; then echo "[push skip] $f none"; return; fi
  git add -f "$f"
  git commit -m "C1 test result s$s" || true
  git pull --rebase origin main || true
  git push "https://${GH_TOKEN}@github.com/AITEAM444/KoMEDMCQA_CoT.git" main && echo "[push ok] s$s" || echo "[push fail] s$s"
}

for s in 42 43 44; do
  echo "===== C1 seed=$s start ====="
  python src/eval/evaluate.py --model Qwen/Qwen3-8B --lora ./c1_adapters/qwen3-8b-c1-s$s --split test --output results/eval_C1_s$s.jsonl || echo "[eval fail] s$s"
  push_seed $s
done
echo "=== ALL DONE ==="
