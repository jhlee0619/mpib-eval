# MPIB Inference Toolkit

Official inference toolkit for **MPIB (Medical Prompt Injection Benchmark)**.

Release version: `v1.1`

Canonical dataset version: `v1.1`

## Installation
```bash
pip install -r requirements.txt
```

## Structure
- `src/`: Core logic (`evaluate_subject.py`, `reconstruct_payload.py`).
- `examples/`: Sample redacted data.
- `data/restricted/`: (User-provided) Payload registries.

## Usage
1. Reconstruct:
   ```bash
   python3 src/reconstruct_payload.py --input_file examples/sample_redacted.jsonl --output_file results/reconstructed.jsonl
   ```
2. Evaluate:
   ```bash
   python3 src/evaluate_subject.py --model "model-id" --input_file results/reconstructed.jsonl --output_file results/responses.jsonl
   ```

## Citation
```bibtex
@misc{lee2026mpibbenchmarkmedicalprompt,
      title={MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs},
      author={Junhyeok Lee and Han Jang and Kyu Sung Choi},
      year={2026},
      eprint={2602.06268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06268},
}
```
