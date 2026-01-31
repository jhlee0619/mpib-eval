#!/bin/bash
# MPIB Evaluation Kit - Run Script

# 1. Configuration
# Input file containing (query, response) pairs to judge
INPUT_FILE="../examples/sample_input.jsonl"

# Output directory
OUTPUT_DIR="../results"
mkdir -p "$OUTPUT_DIR"

# Paths to Python scripts
JUDGE_SCRIPT="../src/benchmark_judges.py"
ANALYZE_SCRIPT="../src/analyze_judge_benchmark.py"

# Hugging Face Token (Required for gated models like Llama-3)
# export HF_TOKEN="your_token_here"

# GPU Settings
# Adjust based on your environment
TP_SIZE=1  # Tensor Parallel size for the Judge model
GPU_MEM_UTIL=0.85

# Define Judge Models
# Format: "alias:huggingface_model_id"
JUDGE_MODELS=(
    "qwen_72b:Qwen/Qwen2.5-72B-Instruct"
    # "llama_70b:meta-llama/Llama-3.1-70B-Instruct"
)

echo "=========================================="
echo "MPIB Evaluation Pipeline"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 2. Run Evaluation
for entry in "${JUDGE_MODELS[@]}"; do
    IFS=':' read -r alias model_id <<< "$entry"
    
    echo "Running Judge: $alias ($model_id)..."
    OUTPUT_FILE="$OUTPUT_DIR/${alias}_results.jsonl"
    
    python3 "$JUDGE_SCRIPT" \
        --model "$model_id" \
        --alias "$alias" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --tensor_parallel_size $TP_SIZE \
        --batch_size 16 \
        --gpu_memory_utilization $GPU_MEM_UTIL
        
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation complete: $OUTPUT_FILE"
    else
        echo "✗ Evaluation failed for $alias"
        exit 1
    fi
done

# 3. Analyze Results
echo ""
echo "Analyzing results..."
python3 "$ANALYZE_SCRIPT" --results_dir "$OUTPUT_DIR"

echo "Done."
