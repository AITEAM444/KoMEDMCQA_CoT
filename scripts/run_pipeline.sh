#!/usr/bin/env bash
# KoMED end-to-end 파이프라인 — 연구계획 §전체
#   생성 → C1(표준) → C3(반사실: 생성+채점+merge) → C2(범용judge+merge) → C-rand
#   → arm build → 학습(arm×seed) → 평가 → 통계 → 리포트
#
# 전제 환경변수:
#   DEEPSEEK_API_KEY   R1 trace / 반사실 CF 생성 (필수)
#   OPENAI_API_KEY     C2/C3 judge = GPT-5 (모델은 OPENAI_JUDGE_MODEL 로 override, default gpt-5)
# 실행:  bash scripts/run_pipeline.sh
# 단계별로 비싸므로 각 산출물이 있으면 건너뛰도록 짜였다(체크포인트는 각 스크립트 내장).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
RESULTS=${RESULTS:-results}
WORKERS=${WORKERS:-16}
SEEDS=${SEEDS:-"42 43 44"}
ARMS=${ARMS:-"C0 C1 C2 C3 C-rand"}
mkdir -p "$DATA" "$RESULTS"

TRACES="$DATA/train_cot_fewshot.jsonl"
CF_RAW="$DATA/train_cf.jsonl"
CF_JUDGED="$DATA/cf_judged.json"
UNIFIED="$DATA/unified.jsonl"

echo "== 1) R1 trace 생성 (train, fewshot) =="
[ -f "$TRACES" ] || python src/generation/generate_traces.py \
    --model deepseek-r1 --total -1 --split train --prompt-mode fewshot \
    --workers "$WORKERS" --output "$TRACES"

echo "== 2) unified 빌드 (C0/C1 판정) =="
python src/dataset/build_arms.py unified --input "$TRACES" --output "$UNIFIED"

echo "== 3) 반사실 CF 생성 (C3 신호) =="
[ -f "$CF_RAW" ] || python src/filters/counterfactual_adapter.py \
    --total -1 --split train --workers "$WORKERS" --output "$CF_RAW"

echo "== 4) 반사실 채점(judge) — precompute =="
# CF_RAW(생성본)을 채점해 metadata.counterfactual 적재. (이미 cf_cot 있으면 채점만)
[ -f "$CF_JUDGED" ] || python -m filters.counterfactual.precompute \
    --input "$CF_RAW" --output "$CF_JUDGED" --workers "$WORKERS"

echo "== 5) C3 merge (C1 ∩ 반사실 통과) =="
# (선택) 임계값(gap/hedge/min_orig)은 configs/pipeline_config.yaml 기본값 사용.
#   dev 로 재보정하려면 먼저: src/eval/calibrate.py 스윕 표 확인 → config 수정 후 이 단계 실행 (README 참고)
python src/dataset/build_arms.py merge-c3 --unified "$UNIFIED" --cf "$CF_JUDGED" --output "$UNIFIED"

echo "== 6) C2 범용 judge 채점 + merge (상위 |C3|) =="
python src/filters/judge_general.py --unified "$UNIFIED" --output "$UNIFIED" --workers "$WORKERS"
python src/dataset/build_arms.py merge-c2 --unified "$UNIFIED" --output "$UNIFIED"

echo "== 7) C-rand (수량 통제군, |C3| 크기) =="
python src/dataset/build_arms.py make-crand --unified "$UNIFIED" --output "$UNIFIED" --seed 42

echo "== 8) 학습 (arm × seed) =="
python src/train/train_lora.py --unified "$UNIFIED" --arms $ARMS --seeds $SEEDS

echo "== 9) 평가 (각 학습본 test) =="
for arm in $ARMS; do
  for s in $SEEDS; do
    lora="output/qwen3-8b-$(echo "$arm" | tr 'A-Z' 'a-z')-s${s}"
    [ -d "$lora" ] || { echo "  (건너뜀: $lora 없음)"; continue; }
    python src/eval/evaluate.py --model Qwen/Qwen3-8B --lora "$lora" \
        --split test --output "$RESULTS/eval_${arm}_s${s}.jsonl"
  done
done

echo "== 10) 통계 (mean±std + CI + McNemar) =="
STAT_ARGS=()
for arm in $ARMS; do
  files=$(for s in $SEEDS; do f="$RESULTS/eval_${arm}_s${s}.jsonl"; [ -f "$f" ] && echo "$f"; done)
  [ -n "$files" ] && STAT_ARGS+=(--arm "$arm" $files)
done
python src/eval/stats.py "${STAT_ARGS[@]}" \
    --mcnemar C3 C1 --mcnemar C3 C-rand --mcnemar C3 C2 \
    --output "$RESULTS/stats.json"

echo "== 11) 리포트 =="
python scripts/make_report.py --unified "$UNIFIED" --stats "$RESULTS/stats.json" \
    --output "$RESULTS/report.md"

echo "✅ 파이프라인 완료 → $RESULTS/report.md"
