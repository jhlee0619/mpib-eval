# MPIB Evaluation Toolkit

This repository contains the official evaluation toolkit for **MPIB (Medical Prompt Injection Benchmark)**. It provides the **LLM-as-a-Judge** pipeline used to evaluate the safety and helpfulness of medical LLMs against the MPIB dataset.

## 📂 Repository Structure

- `src/`: Core Python scripts (Judge, Analysis, Reconstruction).
- `scripts/`: Shell scripts to run the evaluation pipeline.
- `examples/`: Sample input files.

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

Requirements: `vllm`, `transformers`, `numpy`, `scipy`.

### 2. Prepare Input Data

Prepare your model's responses in a JSONL file. See `examples/sample_input.jsonl` for the format.

### 3. (Optional) Reconstruct Payloads for V2

If you have Gated Access to the MPIB dataset, you can reconstruct the exact V2 attack payloads using the registry file.

```bash
python src/reconstruct_payload.py \
    --input_file path/to/mpib_test.jsonl \
    --output_file path/to/mpib_test_restored.jsonl \
    --payload_db path/to/payload_registry_v1.x.json
```

- **Without `--payload_db`**: Restores structural placeholders (safe for debug).
- **With `--payload_db`**: Restores exact malicious payloads (for actual testing).

### 4. Run Evaluation

Use the provided script to run the evaluation.

```bash
cd scripts
bash run_eval.sh
```

## 📊 Metrics

The evaluation pipeline reports safety and utility metrics based on the taxonomy:
- **CHER Analysis**: Classification of responses into **C**ompliant (Safe), **H**armful, **E**vasive, or **R**efusal.
- **ASR (Attack Success Rate)**: Percentage of Harmful responses under attack scenarios.
- **FPR (False Positive Rate)**: Percentage of unnecessary Refusals on benign queries.

## 🔗 Links

- **Dataset**: [Hugging Face Dataset](https://huggingface.co/datasets/jhlee0619/mpib)
- **Paper**: [ArXiv](https://arxiv.org/abs/25xx.xxxxx) (Coming Soon)

## 📜 License

This evaluation toolkit (code) is released under the **MIT License**.
The MPIB dataset itself is released under **CC-BY-NC-4.0**.
