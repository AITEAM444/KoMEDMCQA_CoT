#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
CONFIG=${CONFIG:-configs/train.yaml}
DATASET_DIR=${DATASET_DIR:-data/sft}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/lowdata}
RESULTS=${RESULTS:-results/lowdata}
mkdir -p "$RESULTS"

export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}
RUN_EVAL=${RUN_EVAL:-0}
GPU_MEM=${GPU_MEM:-0.82}
VLLM_DTYPE=${VLLM_DTYPE:-float16}
VLLM_USE_V1=${VLLM_USE_V1:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
EVAL_CHUNK=${EVAL_CHUNK:-256}

run_train_eval() {
    local arm_label="$1"
    local arm_name="$2"
    local n="$3"
    local seed="$4"
    local dataset="komed_low_${arm_name}_n${n}_r42"
    local output_dir="$OUTPUT_ROOT/qwen3-8b-low-${arm_name}-n${n}-r42-s${seed}"
    local result_file="$RESULTS/test_low_${arm_name}_n${n}_r42_s${seed}.jsonl"

    if [ -d "$output_dir" ] && [ "$(find "$output_dir" -maxdepth 1 -type f | wc -l)" -gt 0 ]; then
        echo "[skip train] 이미 학습 아웃풋 존재: $output_dir"
    else
        echo "[train] $arm_label n=$n sample_seed=42 train_seed=$seed"
        llamafactory-cli train "$CONFIG" dataset="$dataset" dataset_dir="$DATASET_DIR" output_dir="$output_dir" seed="$seed" data_seed=42
    fi

    if [ "$RUN_EVAL" = "1" ]; then
        echo "[eval] $arm_label n=$n sample_seed=42 train_seed=$seed"
        python src/eval/evaluate_vllm.py \
            --model "$MODEL" \
            --lora "$output_dir" \
            --split test \
            --output "$result_file" \
            --gpu-mem "$GPU_MEM" \
            --dtype "$VLLM_DTYPE" \
            --vllm-use-v1 "$VLLM_USE_V1" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --chunk "$EVAL_CHUNK"
    else
        echo "[skip eval] RUN_EVAL=0: $result_file"
    fi
}

for n in 300 500 1000; do
    for entry in "C3:c3" "C2:c2" "C-rand:crand"; do
        arm_label="${entry%%:*}"
        arm_name="${entry#*:}"
        for seed in 42 43 44; do
            run_train_eval "$arm_label" "$arm_name" "$n" "$seed"
        done
    done
done

if [ "$RUN_EVAL" = "1" ]; then
    echo '[stats] n=300'
    python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n300_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n300_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n300_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n300.json

    echo '[stats] n=500'
    python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n500_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n500_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n500_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n500.json

    echo '[stats] n=1000'
    python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n1000_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n1000_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n1000_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n1000.json
else
    echo '[done] training only. Set RUN_EVAL=1 to run vLLM eval + stats later.'
fi
