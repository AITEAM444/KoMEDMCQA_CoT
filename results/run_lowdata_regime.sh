#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
CONFIG=${CONFIG:-configs/train.yaml}
DATASET_DIR=${DATASET_DIR:-data/sft}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/lowdata}
RESULTS=${RESULTS:-results/lowdata}
mkdir -p "$RESULTS"

echo '[train] C3 n=300 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s42" seed=42 data_seed=42
echo '[eval] C3 n=300 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s42" --split test --output "$RESULTS/test_low_c3_n300_r42_s42.jsonl"

echo '[train] C3 n=300 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s43" seed=43 data_seed=43
echo '[eval] C3 n=300 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s43" --split test --output "$RESULTS/test_low_c3_n300_r42_s43.jsonl"

echo '[train] C3 n=300 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s44" seed=44 data_seed=44
echo '[eval] C3 n=300 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n300-r42-s44" --split test --output "$RESULTS/test_low_c3_n300_r42_s44.jsonl"

echo '[train] C2 n=300 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s42" seed=42 data_seed=42
echo '[eval] C2 n=300 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s42" --split test --output "$RESULTS/test_low_c2_n300_r42_s42.jsonl"

echo '[train] C2 n=300 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s43" seed=43 data_seed=43
echo '[eval] C2 n=300 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s43" --split test --output "$RESULTS/test_low_c2_n300_r42_s43.jsonl"

echo '[train] C2 n=300 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s44" seed=44 data_seed=44
echo '[eval] C2 n=300 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n300-r42-s44" --split test --output "$RESULTS/test_low_c2_n300_r42_s44.jsonl"

echo '[train] C-rand n=300 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s42" seed=42 data_seed=42
echo '[eval] C-rand n=300 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s42" --split test --output "$RESULTS/test_low_crand_n300_r42_s42.jsonl"

echo '[train] C-rand n=300 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s43" seed=43 data_seed=43
echo '[eval] C-rand n=300 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s43" --split test --output "$RESULTS/test_low_crand_n300_r42_s43.jsonl"

echo '[train] C-rand n=300 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n300_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s44" seed=44 data_seed=44
echo '[eval] C-rand n=300 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n300-r42-s44" --split test --output "$RESULTS/test_low_crand_n300_r42_s44.jsonl"

echo '[train] C3 n=500 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s42" seed=42 data_seed=42
echo '[eval] C3 n=500 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s42" --split test --output "$RESULTS/test_low_c3_n500_r42_s42.jsonl"

echo '[train] C3 n=500 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s43" seed=43 data_seed=43
echo '[eval] C3 n=500 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s43" --split test --output "$RESULTS/test_low_c3_n500_r42_s43.jsonl"

echo '[train] C3 n=500 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s44" seed=44 data_seed=44
echo '[eval] C3 n=500 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n500-r42-s44" --split test --output "$RESULTS/test_low_c3_n500_r42_s44.jsonl"

echo '[train] C2 n=500 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s42" seed=42 data_seed=42
echo '[eval] C2 n=500 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s42" --split test --output "$RESULTS/test_low_c2_n500_r42_s42.jsonl"

echo '[train] C2 n=500 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s43" seed=43 data_seed=43
echo '[eval] C2 n=500 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s43" --split test --output "$RESULTS/test_low_c2_n500_r42_s43.jsonl"

echo '[train] C2 n=500 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s44" seed=44 data_seed=44
echo '[eval] C2 n=500 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n500-r42-s44" --split test --output "$RESULTS/test_low_c2_n500_r42_s44.jsonl"

echo '[train] C-rand n=500 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s42" seed=42 data_seed=42
echo '[eval] C-rand n=500 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s42" --split test --output "$RESULTS/test_low_crand_n500_r42_s42.jsonl"

echo '[train] C-rand n=500 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s43" seed=43 data_seed=43
echo '[eval] C-rand n=500 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s43" --split test --output "$RESULTS/test_low_crand_n500_r42_s43.jsonl"

echo '[train] C-rand n=500 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n500_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s44" seed=44 data_seed=44
echo '[eval] C-rand n=500 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n500-r42-s44" --split test --output "$RESULTS/test_low_crand_n500_r42_s44.jsonl"

echo '[train] C3 n=1000 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s42" seed=42 data_seed=42
echo '[eval] C3 n=1000 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s42" --split test --output "$RESULTS/test_low_c3_n1000_r42_s42.jsonl"

echo '[train] C3 n=1000 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s43" seed=43 data_seed=43
echo '[eval] C3 n=1000 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s43" --split test --output "$RESULTS/test_low_c3_n1000_r42_s43.jsonl"

echo '[train] C3 n=1000 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c3_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s44" seed=44 data_seed=44
echo '[eval] C3 n=1000 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c3-n1000-r42-s44" --split test --output "$RESULTS/test_low_c3_n1000_r42_s44.jsonl"

echo '[train] C2 n=1000 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s42" seed=42 data_seed=42
echo '[eval] C2 n=1000 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s42" --split test --output "$RESULTS/test_low_c2_n1000_r42_s42.jsonl"

echo '[train] C2 n=1000 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s43" seed=43 data_seed=43
echo '[eval] C2 n=1000 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s43" --split test --output "$RESULTS/test_low_c2_n1000_r42_s43.jsonl"

echo '[train] C2 n=1000 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_c2_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s44" seed=44 data_seed=44
echo '[eval] C2 n=1000 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-c2-n1000-r42-s44" --split test --output "$RESULTS/test_low_c2_n1000_r42_s44.jsonl"

echo '[train] C-rand n=1000 sample_seed=42 train_seed=42'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s42" seed=42 data_seed=42
echo '[eval] C-rand n=1000 sample_seed=42 train_seed=42'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s42" --split test --output "$RESULTS/test_low_crand_n1000_r42_s42.jsonl"

echo '[train] C-rand n=1000 sample_seed=42 train_seed=43'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s43" seed=43 data_seed=43
echo '[eval] C-rand n=1000 sample_seed=42 train_seed=43'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s43" --split test --output "$RESULTS/test_low_crand_n1000_r42_s43.jsonl"

echo '[train] C-rand n=1000 sample_seed=42 train_seed=44'
llamafactory-cli train "$CONFIG" dataset=komed_low_crand_n1000_r42 dataset_dir="$DATASET_DIR" output_dir="$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s44" seed=44 data_seed=44
echo '[eval] C-rand n=1000 sample_seed=42 train_seed=44'
python src/eval/evaluate.py --model "$MODEL" --lora "$OUTPUT_ROOT/qwen3-8b-low-crand-n1000-r42-s44" --split test --output "$RESULTS/test_low_crand_n1000_r42_s44.jsonl"

echo '[stats] n=300'
python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n300_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n300_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n300_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n300.json

echo '[stats] n=500'
python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n500_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n500_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n500_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n500.json

echo '[stats] n=1000'
python src/eval/stats.py --arm C3 results/lowdata/test_low_c3_n1000_r*_s*.jsonl --arm C-rand results/lowdata/test_low_crand_n1000_r*_s*.jsonl --arm C2 results/lowdata/test_low_c2_n1000_r*_s*.jsonl --mcnemar C3 C-rand --mcnemar C3 C2 --output results/lowdata/stats_low_n1000.json
