#!/bin/bash
# MPIB Evaluation Pipeline Script
# Running LLM-as-a-Judge to evaluate safety of model outputs.

# --- 1. CONFIGURATION ---

# Path to the input file containing model responses (format: .jsonl)
# Must contain fields: "user_query", "response"
INPUT_FILE="../examples/sample_input.jsonl"

# Directory to save evaluation results
OUTPUT_DIR="../results"
mkdir -p "$OUTPUT_DIR"

# Python Script Paths
JUDGE_SCRIPT="../src/benchmark_judges.py"
ANALYZE_SCRIPT="../src/analyze_judge_benchmark.py"

# Hardware / Model Settings
# TP_SIZE: Tensor Parallelism (Number of GPUs per Judge instance)
# Recommended: 1 for 8B/32B models, 4-8 for 70B+ models
TP_SIZE=4
GPU_MEM_UTIL=0.90

# Define Judge Models to use
# Format: "alias:huggingface_model_id"
# We recommend using Qwen-2.5-72B-Instruct as it mimics human experts best for MPIB.
JUDGES=(
    "qwen_72b_judge:Qwen/Qwen2.5-72B-Instruct"
)

# --- 2. EXECUTION ---

echo "=================================================="
echo "      MPIB Safety Evaluation Pipeline"
echo "=================================================="
echo "Input File : $INPUT_FILE"
echo "Output Dir : $OUTPUT_DIR"
echo "TP Size    : $TP_SIZE"
echo "=================================================="

for entry in "${JUDGES[@]}"; do
    IFS=':' read -r alias model_id <<< "$entry"
    
    OUTPUT_FILE="$OUTPUT_DIR/${alias}_results.jsonl"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[!] Result file already exists for $alias. Skipping..."
        continue
    fi

    echo ""
    echo "[STEP 1] Running Judge: $alias ($model_id)..."
    
    python3 "$JUDGE_SCRIPT" \
        --model "$model_id" \
        --alias "$alias" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --tensor_parallel_size $TP_SIZE \
        --batch_size 16 \
        --gpu_memory_utilization $GPU_MEM_UTIL

    if [ $? -ne 0 ]; then
        echo "[ERROR] Judge failed. Exiting."
        exit 1
    fi
done

echo ""
echo "[STEP 2] Analyzing Metrics (ASR / FPR)..."
python3 "$ANALYZE_SCRIPT" --results_dir "$OUTPUT_DIR"

echo ""
echo "All steps completed successfully."
