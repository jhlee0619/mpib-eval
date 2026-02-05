# MPIB Inference Toolkit

This repository contains the official inference toolkit for **MPIB (Medical Prompt Injection Benchmark)**. It provides a streamlined, production-ready pipeline to test medical LLMs against adversarial prompt injections in a **Blind Evaluation** setting.

## 📦 Installation

This toolkit requires Python 3.9+ and CUDA-capable hardware for vLLM inference.

```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

- `src/`: Core logic and inference engines.
    - `reconstruct_payload.py`: Restores [REDACTED] tokens using the `payload_registry.json`.
    - `evaluate_subject.py`: Handles "Blind" prompt assembly and batch inference via vLLM.
- `data/restricted/`: (User-provided) Storage for sensitive payload registries (Git-ignored).
- `examples/`: Contains sample redacted data for testing.

---

## 🚀 Subject Evaluation Workflow

The toolkit implements a "Reconstruct -> Assemble -> Infer" flow to simulate real-world clinical RAG environments.

### Step 1: Clinical Evidence Reconstruction
V2 payloads are redacted to prevent the propagation of harmful medical advice. Restore them using your registry:
```bash
python3 src/reconstruct_payload.py \
    --input_file examples/sample_redacted.jsonl \
    --output_file results/reconstructed.jsonl \
    --payload_db data/restricted/payload_registry.json
```

### Step 2: Blind Subject Inference (Role Obfuscation)
Run the model through the clinical audit. This step prevents **Label Leakage** by masking internal attack roles into generic clinical terms.

```bash
python3 src/evaluate_subject.py \
    --model "BioMistral/BioMistral-7B" \
    --input_file results/reconstructed.jsonl \
    --output_file results/subject_responses.jsonl \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90
```

---

## 🛡️ The Blind Evaluation Principle
Technical rigor requires that the "Subject" (the model under test) is treated like a clinician in a real-world setting.
- **Asymmetric Information**: Only the Evaluator/Judge knows which documents are "poisoned".
- **Role Transformation**:
    - `benign_evidence` → `clinical_evidence`
    - `poisoned_update` → `clinical_update`
- **Identifier Masking**: Specific document IDs are replaced with generic `REF_X` tags.

---

## 🔗 Project Links
- **Dataset**: [Hugging Face Dataset](https://huggingface.co/datasets/jhlee0619/mpib)

## 📜 License & Citation
- **Toolkit**: MIT License
- **Dataset**: CC-BY-NC-4.0
- **Citation**: Please cite our paper if you use this benchmark.
