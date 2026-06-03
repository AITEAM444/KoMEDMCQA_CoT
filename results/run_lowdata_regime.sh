#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
CONFIG=${CONFIG:-configs/train.yaml}
DATASET_DIR=${DATASET_DIR:-data/sft}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/lowdata}
RESULTS=${RESULTS:-results/lowdata}
mkdir -p "$RESULTS"

export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}
TRAIN_SAVE_ARGS=${TRAIN_SAVE_ARGS:-"save_strategy=\"no\" save_only_model=true save_safetensors=true save_total_limit=1"}
if [ -z "${TRAIN_PRECISION_ARGS:-}" ]; then
    if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
        TRAIN_PRECISION_ARGS="bf16=true fp16=false"
    else
        TRAIN_PRECISION_ARGS="bf16=false fp16=true"
    fi
fi
echo "[config] TRAIN_PRECISION_ARGS=$TRAIN_PRECISION_ARGS"
echo "[config] TRAIN_SAVE_ARGS=$TRAIN_SAVE_ARGS"
RUN_EVAL=${RUN_EVAL:-0}
EVAL_BACKEND=${EVAL_BACKEND:-vllm}
EVAL_FALLBACK=${EVAL_FALLBACK:-1}
LOWDATA_NS=${LOWDATA_NS:-"300 500 1000"}
LOWDATA_ARMS=${LOWDATA_ARMS:-"C3:c3 C2:c2 C-rand:crand"}
LOWDATA_SEEDS=${LOWDATA_SEEDS:-"42 43 44"}
GPU_MEM=${GPU_MEM:-0.82}
VLLM_DTYPE=${VLLM_DTYPE:-float16}
VLLM_USE_V1=${VLLM_USE_V1:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
EVAL_CHUNK=${EVAL_CHUNK:-256}
echo "[config] LOWDATA_NS=$LOWDATA_NS"
echo "[config] LOWDATA_ARMS=$LOWDATA_ARMS"
echo "[config] LOWDATA_SEEDS=$LOWDATA_SEEDS"
echo "[config] EVAL_BACKEND=$EVAL_BACKEND EVAL_FALLBACK=$EVAL_FALLBACK"

run_eval() {
    local output_dir="$1"
    local result_file="$2"

    if [ "$EVAL_BACKEND" = "hf" ]; then
        python src/eval/evaluate.py --model "$MODEL" --lora "$output_dir" --split test --output "$result_file"
        return
    fi

    if python src/eval/evaluate_vllm.py \
        --model "$MODEL" \
        --lora "$output_dir" \
        --split test \
        --output "$result_file" \
        --gpu-mem "$GPU_MEM" \
        --dtype "$VLLM_DTYPE" \
        --vllm-use-v1 "$VLLM_USE_V1" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --chunk "$EVAL_CHUNK"; then
        return
    fi

    if [ "$EVAL_FALLBACK" = "1" ]; then
        echo "[eval fallback] vLLM failed; running HF transformers evaluate.py for $result_file"
        python src/eval/evaluate.py --model "$MODEL" --lora "$output_dir" --split test --output "$result_file"
    else
        return 1
    fi
}

run_train_eval() {
    local arm_label="$1"
    local arm_name="$2"
    local n="$3"
    local seed="$4"
    local dataset="komed_low_${arm_name}_n${n}_r42"
    local output_dir="$OUTPUT_ROOT/qwen3-8b-low-${arm_name}-n${n}-r42-s${seed}"
    local result_file="$RESULTS/test_low_${arm_name}_n${n}_r42_s${seed}.jsonl"
    local done_file="$output_dir/adapter_model.safetensors"

    if [ -f "$done_file" ]; then
        echo "[skip train] completed adapter exists: $done_file"
    else
        if [ -d "$output_dir" ]; then
            echo "[retry train] incomplete output dir exists, removing: $output_dir"
            rm -rf "$output_dir"
        fi
        echo "[train] $arm_label n=$n sample_seed=42 train_seed=$seed"
        llamafactory-cli train "$CONFIG" dataset="$dataset" dataset_dir="$DATASET_DIR" output_dir="$output_dir" seed="$seed" data_seed=42 $TRAIN_PRECISION_ARGS $TRAIN_SAVE_ARGS
    fi

    if [ "$RUN_EVAL" = "1" ]; then
        echo "[eval] $arm_label n=$n sample_seed=42 train_seed=$seed"
        run_eval "$output_dir" "$result_file"
    else
        echo "[skip eval] RUN_EVAL=0: $result_file"
    fi
}

for n in $LOWDATA_NS; do
    for entry in $LOWDATA_ARMS; do
        arm_label="${entry%%:*}"
        arm_name="${entry#*:}"
        for seed in $LOWDATA_SEEDS; do
            run_train_eval "$arm_label" "$arm_name" "$n" "$seed"
        done
    done
done

if [ "$RUN_EVAL" = "1" ]; then
    for n in $LOWDATA_NS; do
        echo "[stats] n=$n"
        python src/eval/stats.py \
            --arm C3 results/lowdata/test_low_c3_n${n}_r*_s*.jsonl \
            --arm C-rand results/lowdata/test_low_crand_n${n}_r*_s*.jsonl \
            --arm C2 results/lowdata/test_low_c2_n${n}_r*_s*.jsonl \
            --mcnemar C3 C-rand --mcnemar C3 C2 \
            --output results/lowdata/stats_low_n${n}.json
    done
else
    echo '[done] training only. Set RUN_EVAL=1 to run vLLM eval + stats later.'
fi
