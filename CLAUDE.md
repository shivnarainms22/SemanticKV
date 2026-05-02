# SemanticKV

Learned semantic importance for KV cache eviction in long-context LLMs.

## Tech Stack
- Python 3.10+, PyTorch 2.2+, Transformers 4.40+
- Primary model: Llama 3 8B Instruct
- Runs on: RunPod A100 80GB (~$1.50/hr)

## How to Run

```bash
# One-time setup on RunPod
bash runpod_setup.sh

# Phase 1 — counterfactual importance + heuristic comparison (~44 GPU hrs)
python scripts/phase1_baseline.py

# Phase 2 — feature extraction + classifier training
python scripts/phase2_generate_labels.py
python scripts/phase2_train_classifier.py
python scripts/phase2_feature_ablation.py

# Phase 3 — full benchmark (accuracy + latency)
python scripts/phase3_benchmark.py
```

## Sanity-check runs (cheap, fast)
Before full runs, override N_PROMPTS/n_samples at the top of each script:
- Phase 1: set N_PROMPTS=3, SAMPLE_FRACTION=0.1
- Phase 3: set N_SAMPLES=5

## Project conventions
- All model inference: fp16 on CUDA, eager attention (not flash)
- Results saved incrementally to data/processed/ — RunPod instances can die
- Blueprint: SemanticKV_Research_Blueprint.md
- Feature count: 10 (6 attention + 3 positional + 1 generation_prefix)
- Classifier input_dim: 10 (matches feature count)
