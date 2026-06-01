#!/usr/bin/env bash
cd /workspace/KoMEDMCQA_CoT
for s in 42 43 44; do
  python src/eval/evaluate.py --model Qwen/Qwen3-8B \
    --lora output/final/qwen3-8b-c3-s$s --split test \
    --output results/test_c3_s$s.jsonl
done
git config user.name dahye44
git config user.email honeyna05@gmail.com
git add -f results/test_c3_s*.jsonl
git commit -m "C3 test 결과(gap1.5 hedge-off, s42/43/44)"
git pull --rebase origin main
git push https://${GH_TOKEN}@github.com/AITEAM444/KoMEDMCQA_CoT.git main
echo "=== 완료: push까지 끝남 ==="
