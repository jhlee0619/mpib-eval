# MPIB Evaluation Toolkit

This repository contains the official evaluation toolkit for **MPIB (Medical Prompt Injection Benchmark)**. It provides the **LLM-as-a-Judge** pipeline used to evaluate the safety and helpfulness of medical LLMs against the MPIB dataset.

## 📂 Repository Structure

- `src/`: Core Python scripts for the judge and analysis.
- `scripts/`: Shell scripts to run the evaluation pipeline.
- `examples/`: Sample input files to demonstrate the expected format.

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

Requirements: `vllm`, `transformers`, `numpy`, `scipy`.

### 2. Prepare Input Data

Prepare your model's responses in a JSONL file. The judge expects the following fields:

```json
{
  "sample_id": "unique_id",
  "user_query": "The medical query/attack text",
  "response": "The model's response to be evaluated",
  "labels": {
    "severity": 0,  // Optional: Ground truth severity (0-4) for metric calculation
    "harm_types": [] // Optional: Ground truth harm types
  }
}
```

See `examples/sample_input.jsonl` for a concrete example.

### 3. Run Evaluation

Use the provided script to run the evaluation. You may need to adjust `TP_SIZE` (Tensor Parallelism) in the script based on your GPU configuration.

```bash
cd scripts
bash run_eval.sh
```

By default, it uses **Qwen-2.5-72B-Instruct** as the judge, which was found to have the highest correlation with human experts in our paper.

## 📊 Metrics

The analysis script reports:
- **Recall@4**: Sensitivity to high-severity risks.
- **SCE (Severity Classification Error)**: Mean Absolute Error against ground truth (if provided).
- **JSON Validity**: Rate of valid JSON outputs from the judge.

## 🔗 Links

- **Dataset**: [Hugging Face Dataset Link](https://huggingface.co/datasets/...)
- **Paper**: [ArXiv Link](https://arxiv.org/abs/...)

## 📜 License

[MIT / CC-BY-NC-4.0]
