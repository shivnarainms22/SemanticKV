# SemanticKV: Learned Semantic Importance for KV Cache Eviction
## Full Research & Implementation Blueprint

---

## What We Are Building and Why

### The Problem

Large language models store intermediate computation as **Key-Value (KV) cache entries** — one KV pair per token per layer. For long contexts, this becomes the dominant memory bottleneck. A single Llama 3 70B request with 8K tokens consumes ~21GB of KV cache memory alone.

Existing solutions evict tokens from the KV cache to save memory. But every current method uses **attention scores as a proxy for importance** — the assumption being that tokens with high attention weights are the ones the model needs. This is a weak proxy. It measures correlation of attention during prefill, not causal necessity for correct generation.

### The Gap We Are Targeting

The core failure mode of attention-score heuristics:

- A token can receive high attention during prefill because of positional patterns, not semantic content
- A token can be low-attention during prefill but become critical when the model starts generating a specific type of output
- Attention patterns shift across layers — a token important in layer 5 may be irrelevant by layer 25
- Task type matters: summarization needs different tokens than question answering over the same context

No existing method learns *what the model actually needs for correct generation*. They all approximate it with a static heuristic measured at prefill time.

### What We Are Proposing

**SemanticKV**: A lightweight learned eviction policy that predicts token-level KV importance based on features that are more causally predictive than raw attention scores.

The key insight: we can generate **ground truth importance labels** by running counterfactual ablations — systematically removing KV entries and measuring how much the output changes. This gives us a supervised learning signal that no prior work has exploited.

### The Research Claim

> A lightweight classifier trained on cross-layer attention consistency, token entropy, and generation-prefix signals can predict KV entry importance more accurately than attention-score heuristics, yielding higher task performance at identical compression ratios.

### Why This Is Novel

| Method | Eviction Signal | Task-Aware | Learned | Cross-Layer |
|---|---|---|---|---|
| H2O | Cumulative attention received | No | No | No |
| SnapKV | Observation-window attention | No | No | No |
| PyramidKV | Layer-wise budget (SnapKV per layer) | No | No | Partial |
| ScissorHands | Pivot token consistency | No | No | No |
| CSKV (2024) | Singular value decomposition | No | No | No |
| KVSharer (2024) | Layer similarity sharing | No | No | Partial |
| **SemanticKV (ours)** | **Counterfactual KL divergence** | **Yes** | **Yes** | **Yes** |

> **Note on novelty:** CSKV and KVSharer (both 2024) use learned compression but target *representation quality*, not *generation-importance prediction*, and neither uses counterfactual labels. Our supervised signal — output KL divergence under ablation — is orthogonal to these approaches and is the first work to frame KV eviction as a supervised learning problem with causal ground truth.

### Why This Is Tractable on RunPod

- Primary model: Llama 3 8B Instruct (all phases)
- Generalization check: Mistral 7B v0.3 (Phase 3 only, ~10% of benchmark cost)
- A100 80GB on RunPod (~$1.50/hr) is sufficient for all phases
- Total estimated compute cost: $150-200 (revised from original estimate; see budget section)
- Timeline: 18-20 weeks to arXiv preprint (revised for multi-model + significance testing)

---

## Repository Structure

```
semantickv/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── prompts/
│   │   ├── longbench_samples.jsonl       # LongBench evaluation prompts
│   │   ├── niah_samples.jsonl            # Needle-in-a-Haystack prompts
│   │   └── diverse_prompts.jsonl         # Training data for classifier
│   └── processed/
│       ├── importance_labels/            # Counterfactual importance scores
│       └── features/                     # Extracted feature vectors
│
├── semantickv/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── loader.py                     # Model + tokenizer loading
│   │   └── hooks.py                      # Attention capture hooks
│   │
│   ├── ablation/
│   │   ├── __init__.py
│   │   ├── counterfactual.py             # Core counterfactual eviction test
│   │   └── scoring.py                    # Output divergence metrics
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── attention_entropy.py          # Per-token attention entropy
│   │   ├── cross_layer.py                # Cross-layer consistency score
│   │   ├── position.py                   # Positional and structural features
│   │   ├── generation_prefix.py          # Probe-generation task-aware features
│   │   └── extractor.py                  # Combined feature extraction pipeline
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py                      # Importance classifier (MLP)
│   │   ├── train.py                      # Training loop
│   │   └── eval.py                       # Classifier evaluation
│   │
│   ├── eviction/
│   │   ├── __init__.py
│   │   ├── base.py                       # Abstract eviction policy interface
│   │   ├── h2o.py                        # H2O baseline
│   │   ├── snapkv.py                     # SnapKV baseline
│   │   ├── pyramidkv.py                  # PyramidKV baseline
│   │   ├── scissorhands.py               # ScissorHands baseline
│   │   └── semantickv.py                 # Our learned policy
│   │
│   └── eval/
│       ├── __init__.py
│       ├── longbench.py                  # LongBench evaluation harness
│       ├── niah.py                       # Needle-in-a-Haystack evaluation
│       └── metrics.py                    # ROUGE, accuracy, KL divergence
│
├── scripts/
│   ├── phase1_baseline.py                # Phase 1 runner
│   ├── phase2_generate_labels.py         # Phase 2 label generation
│   ├── phase2_train_classifier.py        # Phase 2 classifier training
│   ├── phase2_feature_ablation.py        # Feature importance ablation study
│   ├── phase3_benchmark.py               # Phase 3 full benchmark (accuracy + latency)
│   └── visualize_results.py             # Result plotting
│
├── notebooks/
│   ├── 01_explore_attention_patterns.ipynb
│   ├── 02_counterfactual_analysis.ipynb
│   └── 03_classifier_analysis.ipynb
│
└── experiments/
    └── configs/
        ├── phase1.yaml
        ├── phase2.yaml
        └── phase3.yaml
```

---

## Environment Setup

### requirements.txt

```txt
torch>=2.2.0
transformers>=4.40.0
accelerate>=0.28.0
datasets>=2.18.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
rouge-score>=0.1.2
nltk>=3.8.1
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
pyyaml>=6.0.1
wandb>=0.16.0
einops>=0.7.0
```

### RunPod Setup Script

```bash
#!/bin/bash
# runpod_setup.sh — run once after spinning up instance

pip install --upgrade pip
pip install -r requirements.txt

# Download Llama 3 8B Instruct (requires HF token)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    token='YOUR_HF_TOKEN',
    local_dir='./models/llama3-8b-instruct'
)
"

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

---

## Phase 1: Diagnostic — Quantify Heuristic Failure (Weeks 1-3)

### Goal

Prove empirically that attention-score heuristics evict tokens that actually matter for correct generation. This is the empirical foundation of the entire paper. Without this, the research claim is a hypothesis. With this, it is a demonstrated gap.

### What We Are Measuring

For each token position in a prompt's KV cache, we want to know: **if we removed this token's KV entry, how much would the output change?**

This is the **counterfactual importance score** — the ground truth we will later train a classifier to predict.

We then compare this against what H2O and SnapKV would have predicted. The gap between heuristic predictions and ground truth is the research contribution of Phase 1.

### Core Implementation

#### `semantickv/model/loader.py`

```python
"""
Model loading with attention output capture enabled.
We need raw attention weights for feature extraction,
so we must disable Flash Attention and enable output_attentions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Llama 3 8B with attention weights exposed.
    
    NOTE: We disable Flash Attention here intentionally.
    Flash Attention does not return attention weight matrices,
    which we need for feature extraction. The memory cost is
    acceptable for 8B at float16 on A100 80GB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",   # NOT flash_attention_2
        output_attentions=False,        # We use hooks instead (more efficient)
    )
    model.eval()
    return model, tokenizer
```

#### `semantickv/model/hooks.py`

```python
"""
Forward hooks to capture attention weights and hidden states
during a single forward pass without modifying model internals.
"""

import torch
from typing import Dict, List, Optional
from contextlib import contextmanager


class AttentionCapture:
    """
    Captures attention weight matrices from all layers
    during a forward pass via PyTorch forward hooks.
    
    Usage:
        capture = AttentionCapture(model)
        with capture.capture():
            outputs = model(**inputs)
        attn_weights = capture.get_weights()  # [num_layers, heads, seq, seq]
    """

    def __init__(self, model):
        self.model = model
        self._weights: List[torch.Tensor] = []
        self._hooks = []

    def _hook_fn(self, module, input, output):
        """
        Hook registered on each attention layer.
        output[1] contains attention weights when output_attentions=True
        in the forward call. We capture and move to CPU immediately
        to avoid accumulating on GPU across 32 layers.
        """
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            # Shape: [batch, heads, seq_len, seq_len]
            self._weights.append(output[1].detach().cpu())

    @contextmanager
    def capture(self):
        """Context manager that installs hooks, yields, then removes them."""
        try:
            for layer in self.model.model.layers:
                h = layer.self_attn.register_forward_hook(self._hook_fn)
                self._hooks.append(h)
            yield self
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

    def get_weights(self) -> Optional[torch.Tensor]:
        """
        Returns stacked attention weights.
        Shape: [num_layers, batch, heads, seq_len, seq_len]
        """
        if not self._weights:
            return None
        return torch.stack(self._weights, dim=0)

    def clear(self):
        self._weights.clear()
```

#### `semantickv/ablation/counterfactual.py`

```python
"""
Counterfactual eviction test — the core of Phase 1.

For each token in a prompt's KV cache, we measure how much the
model output changes when that token's KV entry is removed.
This gives us a ground truth importance score for every token.

High output divergence when a token is removed = high importance.
Low output divergence = that token's KV entry is expendable.

This is computationally expensive (O(seq_len) forward passes per prompt)
so we use a stratified sampling strategy and early stopping.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


def run_baseline_forward(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
) -> torch.Tensor:
    """
    Run greedy generation via manual loop, capture logits at each step.
    Returns: [max_new_tokens, vocab_size]
    """
    scores = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]          # [batch, vocab]
        scores.append(next_logits.squeeze(0))
        next_token = next_logits.argmax(dim=-1, keepdim=True)

        for _ in range(max_new_tokens - 1):
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            next_logits = out.logits[:, -1, :]
            scores.append(next_logits.squeeze(0))
            past_kv = out.past_key_values
            next_token = next_logits.argmax(dim=-1, keepdim=True)

    return torch.stack(scores, dim=0)               # [max_new_tokens, vocab]


def run_ablated_forward(
    model,
    input_ids: torch.Tensor,
    ablate_position: int,
    max_new_tokens: int = 50,
) -> torch.Tensor:
    """
    Run generation with position `ablate_position` removed from the KV cache.

    Strategy:
      1. Pass attention_mask=0 at ablate_position during prefill so the
         softmax denominator excludes it from the start.
      2. Zero v at that position in the resulting cache as belt-and-suspenders.
      3. During each generation step, extend the mask to keep ablate_position
         excluded from all future attention.

    This is the correct approach: masking k via attention_mask fixes the
    softmax normalization; zeroing v ensures zero value contribution even
    if the mask is not applied perfectly by all attention kernel paths.

    Returns: [max_new_tokens, vocab_size]
    """
    seq_len = input_ids.shape[1]
    scores = []

    with torch.no_grad():
        # Prefill mask: [batch, seq_len], 1=attend, 0=exclude
        prefill_mask = torch.ones(1, seq_len, dtype=torch.long, device=input_ids.device)
        prefill_mask[:, ablate_position] = 0

        prefill_out = model(input_ids, attention_mask=prefill_mask, use_cache=True)
        past_kv = prefill_out.past_key_values

        # Zero v at ablated position in every layer (belt-and-suspenders)
        ablated_kv = []
        for k, v in past_kv:
            v = v.clone()
            v[:, :, ablate_position, :] = 0.0
            ablated_kv.append((k, v))
        past_kv = tuple(ablated_kv)

        # First generation step reuses prefill logits — no extra forward pass
        next_logits = prefill_out.logits[:, -1, :]
        scores.append(next_logits.squeeze(0))
        next_token = next_logits.argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens - 1):
            # Mask covers: original seq_len tokens + `step` already-generated tokens + 1 current
            current_past_len = seq_len + step
            gen_mask = torch.ones(1, current_past_len + 1, dtype=torch.long, device=input_ids.device)
            gen_mask[:, ablate_position] = 0

            out = model(
                next_token,
                past_key_values=past_kv,
                attention_mask=gen_mask,
                use_cache=True,
            )
            next_logits = out.logits[:, -1, :]
            scores.append(next_logits.squeeze(0))
            past_kv = out.past_key_values
            next_token = next_logits.argmax(dim=-1, keepdim=True)

    return torch.stack(scores, dim=0)   # [max_new_tokens, vocab]


def compute_output_divergence(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
) -> float:
    """
    Compute KL divergence between baseline and ablated output distributions.
    
    We average KL divergence across all generation steps.
    Higher divergence = ablated token had more impact on generation.
    
    KL(P || Q) where P=baseline, Q=ablated
    """
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)

    # KL divergence per step: [steps]
    kl_per_step = F.kl_div(
        ablated_probs.log(),
        baseline_probs,
        reduction='none'
    ).sum(dim=-1)

    return kl_per_step.mean().item()


def compute_importance_scores(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    sample_fraction: float = 1.0,
    skip_first_n: int = 5,        # Skip BOS and early system tokens
    skip_last_n: int = 10,        # Skip most recent tokens (always high importance)
) -> Dict:
    """
    Main function: compute counterfactual importance for all tokens in a prompt.
    
    Args:
        prompt: Input text
        sample_fraction: Fraction of positions to ablate (1.0 = all, use 0.3 for speed)
        skip_first_n: Skip first N tokens (system prompt, BOS — always important)
        skip_last_n: Skip last N tokens (recent context — always important by recency)
    
    Returns dict with:
        - tokens: list of token strings
        - importance_scores: array of counterfactual importance per token
        - baseline_logits: baseline generation logits
        - metadata: prompt stats
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get baseline
    baseline_logits = run_baseline_forward(model, input_ids, max_new_tokens)

    # Determine which positions to ablate
    positions_to_test = list(range(skip_first_n, seq_len - skip_last_n))
    
    if sample_fraction < 1.0:
        n_sample = max(1, int(len(positions_to_test) * sample_fraction))
        positions_to_test = np.random.choice(
            positions_to_test, n_sample, replace=False
        ).tolist()

    # Run ablations
    importance_scores = np.zeros(seq_len)
    importance_scores[:skip_first_n] = 1.0        # Assume important
    importance_scores[seq_len - skip_last_n:] = 1.0  # Assume important

    for pos in tqdm(positions_to_test, desc=f"Ablating positions (seq_len={seq_len})"):
        ablated_logits = run_ablated_forward(model, input_ids, pos, max_new_tokens)
        divergence = compute_output_divergence(baseline_logits, ablated_logits)
        importance_scores[pos] = divergence

    return {
        "tokens": tokens,
        "importance_scores": importance_scores,
        "baseline_logits": baseline_logits.cpu().numpy(),
        "seq_len": seq_len,
        "prompt": prompt,
    }
```

#### `semantickv/ablation/scoring.py`

```python
"""
Utilities to compare heuristic eviction predictions against
ground truth counterfactual importance scores.

This is where we quantify the "research gap" — the mismatch between
what H2O/SnapKV would evict and what we know actually matters.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from typing import Dict, Tuple


def binarize_importance(
    importance_scores: np.ndarray,
    keep_fraction: float = 0.3,
) -> np.ndarray:
    """
    Convert continuous importance scores to binary labels.
    Top keep_fraction tokens are labeled 1 (important), rest are 0.
    
    This mirrors the real eviction setting where we keep a budget
    of K tokens and discard the rest.
    """
    threshold = np.percentile(importance_scores, (1 - keep_fraction) * 100)
    return (importance_scores >= threshold).astype(int)


def compute_h2o_scores(
    attention_weights: np.ndarray,
    recent_window: int = 20,
) -> np.ndarray:
    """
    H2O eviction score: sum of attention received + recency bonus.
    attention_weights: [heads, seq_len, seq_len] averaged over layers
    Returns: importance score per token position
    """
    # Average over heads, sum attention received (column sum)
    avg_attn = attention_weights.mean(axis=0)  # [seq_len, seq_len]
    heavy_scores = avg_attn.sum(axis=0)        # [seq_len] — attention received

    # Add recency: last `recent_window` tokens always kept
    recency_bonus = np.zeros(len(heavy_scores))
    recency_bonus[-recent_window:] = heavy_scores.max() * 10

    return heavy_scores + recency_bonus


def compute_snapkv_scores(
    attention_weights: np.ndarray,
    observation_window: int = 32,
) -> np.ndarray:
    """
    SnapKV score: attention from the last observation_window tokens
    averaged across heads and last few layers.
    """
    # Use last few layers (SnapKV uses last layer by default)
    last_layer_attn = attention_weights[-1]    # [heads, seq_len, seq_len]
    
    # Attention from observation window queries to all key positions
    obs_queries = last_layer_attn[:, -observation_window:, :]  # [heads, obs, seq]
    
    # Pool: mean over heads and observation queries
    scores = obs_queries.mean(axis=0).mean(axis=0)  # [seq_len]
    return scores


def evaluate_heuristic(
    ground_truth_scores: np.ndarray,
    heuristic_scores: np.ndarray,
    keep_fraction: float = 0.3,
    name: str = "heuristic",
) -> Dict:
    """
    Evaluate how well a heuristic predicts true token importance.
    
    Metrics:
    - AUC-ROC: overall ranking quality
    - Average Precision: precision-recall quality
    - Top-K Recall: of the truly important tokens, how many does the
      heuristic correctly identify as important?
    """
    binary_gt = binarize_importance(ground_truth_scores, keep_fraction)
    
    # Normalize heuristic scores to [0,1] for probability-like interpretation
    h_norm = (heuristic_scores - heuristic_scores.min())
    if h_norm.max() > 0:
        h_norm = h_norm / h_norm.max()

    auc = roc_auc_score(binary_gt, h_norm)
    ap = average_precision_score(binary_gt, h_norm)

    # Top-K recall: what fraction of truly important tokens does
    # the heuristic rank in its top-K?
    k = int(keep_fraction * len(ground_truth_scores))
    heuristic_top_k = set(np.argsort(heuristic_scores)[-k:])
    true_top_k = set(np.argsort(ground_truth_scores)[-k:])
    top_k_recall = len(heuristic_top_k & true_top_k) / len(true_top_k)

    return {
        "name": name,
        "auc_roc": auc,
        "average_precision": ap,
        "top_k_recall": top_k_recall,
        "keep_fraction": keep_fraction,
    }
```

#### `scripts/phase1_baseline.py`

```python
"""
Phase 1 Runner: Quantify heuristic failure on a sample of LongBench prompts.

What this script produces:
1. Counterfactual importance scores for 50 prompts
2. H2O and SnapKV heuristic predictions for same prompts
3. Comparison metrics (AUC, top-K recall)
4. The key figure: scatter plot of heuristic score vs counterfactual score
   showing low correlation — this IS the paper's opening figure.

Runtime estimate: ~4-6 hours on A100 80GB
Cost estimate: ~$9 on RunPod
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from semantickv.model.loader import load_model
from semantickv.model.hooks import AttentionCapture
from semantickv.ablation.counterfactual import compute_importance_scores
from semantickv.ablation.scoring import (
    compute_h2o_scores,
    compute_snapkv_scores,
    evaluate_heuristic,
)


MODEL_PATH = "./models/llama3-8b-instruct"
OUTPUT_DIR = Path("./data/processed/phase1_results")
N_PROMPTS = 50          # Start with 50; expand to 200 for final paper
KEEP_FRACTION = 0.3     # Simulate 70% eviction rate
MAX_NEW_TOKENS = 50
SAMPLE_FRACTION = 0.5   # Ablate 50% of positions (speed/cost tradeoff)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_evaluation_prompts(n: int = 50):
    """Load diverse prompts from LongBench (2WikiMQA, Qasper, MultiNews)."""
    # Using HuggingFace datasets
    from datasets import load_dataset
    
    prompts = []
    
    # 2WikiMultiHopQA — multi-hop reasoning, needs specific evidence tokens
    ds = load_dataset("THUDM/LongBench", "2wikimqa_e", split="test")
    for item in list(ds)[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nQuestion: " + item["input"],
            "task": "multi_hop_qa",
            "answer": item["answers"][0] if item["answers"] else "",
        })
    
    # Qasper — document QA, needs targeted evidence
    ds = load_dataset("THUDM/LongBench", "qasper_e", split="test")
    for item in list(ds)[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nQuestion: " + item["input"],
            "task": "document_qa",
            "answer": item["answers"][0] if item["answers"] else "",
        })

    # MultiNews — summarization, needs broad coverage
    ds = load_dataset("THUDM/LongBench", "multi_news_e", split="test")
    for item in list(ds)[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nSummarize the above:",
            "task": "summarization",
            "answer": item["answers"][0] if item["answers"] else "",
        })

    return prompts[:n]


def main():
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    capture = AttentionCapture(model)

    print(f"Loading {N_PROMPTS} evaluation prompts...")
    prompts = load_evaluation_prompts(N_PROMPTS)

    results = []

    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        print(f"\n--- Prompt {i+1}/{N_PROMPTS} (task: {prompt_data['task']}) ---")
        
        # Tokenize to check length
        tokens = tokenizer(prompt_data["text"], return_tensors="pt")
        seq_len = tokens["input_ids"].shape[1]
        
        # Skip very short or very long sequences
        if seq_len < 100 or seq_len > 3000:
            print(f"  Skipping: seq_len={seq_len}")
            continue

        # Step 1: Compute ground truth counterfactual importance
        print(f"  Computing counterfactual importance (seq_len={seq_len})...")
        importance_data = compute_importance_scores(
            model, tokenizer, prompt_data["text"],
            max_new_tokens=MAX_NEW_TOKENS,
            sample_fraction=SAMPLE_FRACTION,
        )

        # Step 2: Capture attention weights for heuristic computation
        print("  Capturing attention weights for heuristics...")
        inputs = tokenizer(prompt_data["text"], return_tensors="pt").to(model.device)
        with capture.capture():
            with torch.no_grad():
                _ = model(
                    **inputs,
                    output_attentions=True,
                )
        attn_weights = capture.get_weights()  # [layers, batch, heads, seq, seq]
        capture.clear()

        if attn_weights is None:
            print("  Warning: No attention weights captured, skipping.")
            continue

        # Shape: [layers, heads, seq, seq]
        attn_np = attn_weights[:, 0, :, :, :].numpy()

        # Step 3: Compute heuristic scores
        h2o_scores = compute_h2o_scores(attn_np.mean(axis=0))
        snapkv_scores = compute_snapkv_scores(attn_np)
        gt_scores = importance_data["importance_scores"]

        # Step 4: Evaluate heuristics
        h2o_eval = evaluate_heuristic(gt_scores, h2o_scores, KEEP_FRACTION, "H2O")
        snapkv_eval = evaluate_heuristic(gt_scores, snapkv_scores, KEEP_FRACTION, "SnapKV")

        result = {
            "prompt_id": i,
            "task": prompt_data["task"],
            "seq_len": seq_len,
            "gt_scores": gt_scores.tolist(),
            "h2o_scores": h2o_scores.tolist(),
            "snapkv_scores": snapkv_scores.tolist(),
            "h2o_eval": h2o_eval,
            "snapkv_eval": snapkv_eval,
            "tokens": importance_data["tokens"],
        }
        results.append(result)

        # Save incrementally (RunPod instances can die)
        with open(OUTPUT_DIR / f"result_{i:04d}.json", "w") as f:
            json.dump(result, f)

        print(f"  H2O   AUC={h2o_eval['auc_roc']:.3f}  Top-K Recall={h2o_eval['top_k_recall']:.3f}")
        print(f"  SnapKV AUC={snapkv_eval['auc_roc']:.3f}  Top-K Recall={snapkv_eval['top_k_recall']:.3f}")

    # Aggregate results
    print("\n=== PHASE 1 SUMMARY ===")
    for method in ["H2O", "SnapKV"]:
        key = f"{method.lower()}_eval"
        aucs = [r[key]["auc_roc"] for r in results]
        recalls = [r[key]["top_k_recall"] for r in results]
        print(f"{method}: Mean AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"Mean Top-K Recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}")

    # Save summary
    summary = {
        "n_prompts": len(results),
        "keep_fraction": KEEP_FRACTION,
        "h2o_mean_auc": np.mean([r["h2o_eval"]["auc_roc"] for r in results]),
        "snapkv_mean_auc": np.mean([r["snapkv_eval"]["auc_roc"] for r in results]),
        "h2o_mean_recall": np.mean([r["h2o_eval"]["top_k_recall"] for r in results]),
        "snapkv_mean_recall": np.mean([r["snapkv_eval"]["top_k_recall"] for r in results]),
    }
    with open(OUTPUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("\nIf H2O/SnapKV AUC < 0.75 or Top-K Recall < 0.65, the gap is significant enough to publish.")


if __name__ == "__main__":
    main()
```

---

## Phase 2: Feature Engineering & Classifier (Weeks 4-10)

### Goal

Train a lightweight MLP that takes per-token features as input and outputs a predicted importance score. It must be fast enough to run inline during inference.

### Feature Design

These are the features that should be more predictive than raw attention scores:

| Feature | Intuition | Computation Cost |
|---|---|---|
| Mean attention received (all layers) | Baseline H2O signal | Low |
| Attention entropy per token | High entropy = diffuse, possibly low importance | Low |
| Cross-layer attention consistency | Tokens consistently attended across layers = structurally important | Medium |
| Layer-wise attention rank | Is this token in the top-K attended tokens per layer? | Low |
| Token type (punctuation, entity, etc.) | Structural tokens often less important | Zero (from tokenizer) |
| Relative position | Recency bias is real but should be learned, not hardcoded | Zero |
| Generation prefix signal | What the model is currently generating hints at what past tokens it needs | Medium |

#### `semantickv/features/generation_prefix.py`

```python
"""
Generation prefix signal: capture attention FROM early generated tokens BACK
to the prompt.  This is the feature that makes SemanticKV genuinely task-aware.

Intuition: the first few tokens the model generates reveal what type of answer
it is constructing.  Tokens in the prompt that receive focused attention during
early generation are provably needed for the current output — this is a stronger
signal than static prefill attention, which only reflects structural patterns.

At training time: run probe generation after counterfactual labels are computed.
At inference time: run probe generation before the eviction decision.
"""

import torch
import numpy as np
from contextlib import contextmanager
from typing import Optional


def compute_generation_prefix_features(
    model,
    input_ids: torch.Tensor,
    n_probe_tokens: int = 5,
) -> np.ndarray:
    """
    Run n_probe_tokens steps of greedy generation and capture per-step attention
    from the new token back to all prompt positions.

    Args:
        model: loaded model with eager attention (output_attentions supported)
        input_ids: [1, prompt_seq_len] prompt token IDs
        n_probe_tokens: number of generation steps to probe (5 adds ~5ms overhead)

    Returns:
        [prompt_seq_len] array — mean attention received from probe tokens,
        averaged over all probe steps, layers, and heads.
    """
    seq_len = input_ids.shape[1]
    probe_attn_accumulator = np.zeros(seq_len)
    total_contributions = 0

    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(n_probe_tokens):
            probe_out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )
            # probe_out.attentions: tuple of [batch, heads, 1, past_len] per layer
            # past_len = seq_len + step (original prompt + already-generated tokens)
            # We only want attention to the original prompt positions [0:seq_len]
            for layer_attn in probe_out.attentions:
                if layer_attn is None:
                    continue
                # [batch, heads, 1, past_len] → [heads, past_len]
                attn = layer_attn[0, :, 0, :].cpu().numpy()
                # Sum over heads, accumulate only prompt-position columns
                probe_attn_accumulator += attn[:, :seq_len].sum(axis=0)
                total_contributions += attn.shape[0]  # n_heads

            past_kv = probe_out.past_key_values
            next_token = probe_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if total_contributions > 0:
        probe_attn_accumulator /= total_contributions
    return probe_attn_accumulator  # [seq_len]
```

#### `semantickv/features/extractor.py`

```python
"""
Feature extraction pipeline.
Given attention weights [layers, heads, seq, seq] and token metadata,
produce a feature matrix [seq_len, n_features] for each prompt.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from scipy.stats import entropy


def compute_attention_features(
    attn_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute per-token attention-based features.
    
    Args:
        attn_weights: [n_layers, n_heads, seq_len, seq_len]
    
    Returns:
        features: [seq_len, n_attention_features]
    """
    n_layers, n_heads, seq_len, _ = attn_weights.shape

    features = []

    # Feature 1: Mean attention received across all layers and heads
    # Shape: [seq_len] — how much attention does each token receive on average
    mean_attn_received = attn_weights.mean(axis=(0, 1)).sum(axis=0)
    features.append(mean_attn_received)

    # Feature 2: Column entropy per token per layer, then averaged.
    # We want to know how "focused" the attention TO this token is across
    # the queries that attend to it.  Low column entropy = a few specific
    # queries attend heavily here = genuine key token.  High column entropy
    # = attention is diffuse = likely less causally important.
    # (Row entropy — what this token attends to — is irrelevant for KV
    # importance because it describes the query side, not the key side.)
    entropies = []
    for layer in range(n_layers):
        layer_attn = attn_weights[layer].mean(axis=0)  # [seq, seq]
        # Entropy of each column (which queries attend to this key position)
        col_entropies = np.array([
            entropy(layer_attn[:, j] + 1e-10) for j in range(seq_len)
        ])
        entropies.append(col_entropies)
    mean_entropy = np.stack(entropies).mean(axis=0)  # [seq_len]
    features.append(mean_entropy)

    # Feature 3: Cross-layer attention consistency
    # For each token, how consistently is it attended to across layers?
    # High std = inconsistent = potentially less reliably important
    per_layer_attn = attn_weights.mean(axis=1).sum(axis=1)  # [layers, seq_len]
    cross_layer_std = per_layer_attn.std(axis=0)   # [seq_len]
    cross_layer_mean = per_layer_attn.mean(axis=0) # [seq_len]
    consistency = cross_layer_mean / (cross_layer_std + 1e-8)  # signal-to-noise
    features.append(consistency)

    # Feature 4: Layer-wise rank (is this token in top-K per layer?)
    # Average rank across layers, normalized
    K = max(1, seq_len // 5)  # top 20%
    top_k_counts = np.zeros(seq_len)
    for layer in range(n_layers):
        layer_scores = attn_weights[layer].mean(axis=0).sum(axis=0)  # [seq_len]
        top_k_idx = np.argsort(layer_scores)[-K:]
        top_k_counts[top_k_idx] += 1
    top_k_freq = top_k_counts / n_layers  # fraction of layers where token is top-K
    features.append(top_k_freq)

    # Feature 5: Early vs late layer attention split
    # Some tokens are important early (syntactic) vs late (semantic)
    mid = n_layers // 2
    early_attn = attn_weights[:mid].mean(axis=(0, 1)).sum(axis=0)
    late_attn = attn_weights[mid:].mean(axis=(0, 1)).sum(axis=0)
    early_late_ratio = early_attn / (late_attn + 1e-8)
    features.append(early_late_ratio)
    features.append(late_attn)  # Late-layer attention is often most relevant

    return np.stack(features, axis=1)  # [seq_len, n_features]


def get_boundary_token_ids(tokenizer) -> set:
    """
    Compute sentence-boundary token IDs from the tokenizer at runtime.
    Never hardcode token IDs — they differ between Llama 2 and Llama 3.
    Call once and pass the result to extract_all_features / compute_positional_features.
    """
    boundary_chars = ['.', '\n', '?', '!', ',', ';', '.\n']
    ids = set()
    for char in boundary_chars:
        encoded = tokenizer.encode(char, add_special_tokens=False)
        ids.update(encoded)
    return ids


def compute_positional_features(
    seq_len: int,
    token_ids: Optional[np.ndarray] = None,
    boundary_token_ids: Optional[set] = None,
) -> np.ndarray:
    """
    Positional and structural features that don't require attention weights.

    boundary_token_ids should come from get_boundary_token_ids(tokenizer) so
    the IDs are correct for whichever model/tokenizer is in use.
    """
    features = []

    # Normalized position (0=beginning, 1=end)
    positions = np.arange(seq_len) / max(seq_len - 1, 1)
    features.append(positions)

    # Distance from end (recency signal — learned, not hardcoded)
    features.append(1.0 - positions)

    # Sentence boundary indicator
    if token_ids is not None and boundary_token_ids is not None:
        is_boundary = np.array([
            1.0 if int(t) in boundary_token_ids else 0.0
            for t in token_ids
        ])
    else:
        is_boundary = np.zeros(seq_len)
    features.append(is_boundary)

    return np.stack(features, axis=1)  # [seq_len, n_pos_features]


def extract_all_features(
    attn_weights: np.ndarray,
    seq_len: int,
    token_ids: Optional[np.ndarray] = None,
    boundary_token_ids: Optional[set] = None,
    generation_prefix_attn: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full feature extraction pipeline.

    generation_prefix_attn: [seq_len] array from compute_generation_prefix_features().
        Pass None to omit (falls back to zeros, giving the model a zero feature signal).
        Total features = 6 (attention) + 3 (positional) + 1 (generation prefix) = 10.

    Returns: [seq_len, 10]
    """
    attn_features = compute_attention_features(attn_weights)    # [seq_len, 6]
    pos_features = compute_positional_features(seq_len, token_ids, boundary_token_ids)  # [seq_len, 3]

    if generation_prefix_attn is not None:
        gen_feature = generation_prefix_attn.reshape(-1, 1)     # [seq_len, 1]
    else:
        gen_feature = np.zeros((seq_len, 1))

    return np.concatenate([attn_features, pos_features, gen_feature], axis=1)  # [seq_len, 10]
```

#### `semantickv/classifier/model.py`

```python
"""
Lightweight importance classifier.

Design constraints:
- Must run in <1ms per token on GPU (inference latency budget)
- Input: per-token feature vector
- Output: importance score in [0, 1]
- Must generalize across prompt types and lengths
"""

import torch
import torch.nn as nn
from typing import Optional


class ImportanceClassifier(nn.Module):
    """
    3-layer MLP importance classifier with layer normalization.
    
    Input: [batch_size, n_features] — one row per token
    Output: [batch_size, 1] — importance score in [0, 1]
    
    Design choices:
    - LayerNorm instead of BatchNorm: stable across variable batch sizes
    - Residual connection: helps gradient flow, improves calibration
    - Sigmoid output: interpretable as importance probability
    - Dropout: regularization, important since training data is noisy
      (counterfactual labels have measurement noise)
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.layer3 = nn.Linear(hidden_dim, 1)
        # No Sigmoid here — BCEWithLogitsLoss in train.py applies it internally.
        # Call torch.sigmoid() explicitly when you need probabilities at inference.

        # Residual projection if dims don't match
        self.residual_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        residual = self.residual_proj(x)

        h = self.drop1(self.act1(self.norm1(self.layer1(x))))
        h = h + residual

        h = self.drop2(self.act2(self.norm2(self.layer2(h))))

        return self.layer3(h).squeeze(-1)  # [batch] — raw logits


class ImportanceDataset(torch.utils.data.Dataset):
    """
    Dataset of (feature_vector, importance_label) pairs.
    One sample = one token from one prompt.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

#### `semantickv/classifier/train.py`

```python
"""
Training loop for the importance classifier.

Key design decisions:
- Loss: BCEWithLogitsLoss on binarized importance labels (model outputs raw logits)
- Also try: ranking loss (pairwise) — may be better than classification
  since we care more about relative ordering than absolute threshold
- Evaluation: AUC-ROC on held-out prompts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import wandb

from semantickv.classifier.model import ImportanceClassifier, ImportanceDataset


def load_training_data(
    phase1_results_dir: Path,
    keep_fraction: float = 0.3,
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Load Phase 1 results and return prompt-level train/val splits.

    Split is done at the prompt level before flattening to tokens, so no
    token from a val prompt can appear in the train set.

    Returns: (train_features, train_labels, val_features, val_labels)
    """
    result_files = sorted(phase1_results_dir.glob("result_*.json"))
    print(f"Loading {len(result_files)} result files...")

    per_prompt_features = []
    per_prompt_labels = []

    for f in result_files:
        with open(f) as fp:
            result = json.load(fp)

        feature_path = f.parent.parent / "features" / f.name
        if not feature_path.exists():
            continue

        with open(feature_path) as fp:
            feat_data = json.load(fp)

        features = np.array(feat_data["features"])   # [seq_len, n_features]
        gt_scores = np.array(result["gt_scores"])    # [seq_len]

        threshold = np.percentile(gt_scores, (1 - keep_fraction) * 100)
        labels = (gt_scores >= threshold).astype(np.float32)

        per_prompt_features.append(features)
        per_prompt_labels.append(labels)

    # Split at the prompt level to prevent data leakage
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(per_prompt_features))
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    def _concat(idx_set):
        feats = np.concatenate([per_prompt_features[i] for i in sorted(idx_set)], axis=0)
        labs = np.concatenate([per_prompt_labels[i] for i in sorted(idx_set)], axis=0)
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(labs, dtype=torch.float32),
        )

    train_features, train_labels = _concat(train_idx)
    val_features, val_labels = _concat(val_idx)

    print(f"Train tokens: {len(train_labels)}  Val tokens: {len(val_labels)}")
    print(f"Train positive rate: {train_labels.mean():.3f}  "
          f"Val positive rate: {val_labels.mean():.3f}")

    return train_features, train_labels, val_features, val_labels

# Usage in scripts/phase2_train_classifier.py:
#   train_f, train_l, val_f, val_l = load_training_data(results_dir)
#   model, auc = train_classifier(train_f, train_l, val_f, val_l, output_dir)


def train_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    output_dir: Path,
    n_epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
    use_wandb: bool = False,
):
    """Full training loop with early stopping based on val AUC."""

    if use_wandb:
        wandb.init(project="semantickv", name="classifier_training")

    train_ds = ImportanceDataset(train_features, train_labels)
    val_ds = ImportanceDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False)

    input_dim = train_features.shape[1]
    model = ImportanceClassifier(input_dim=input_dim).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Weighted BCE to handle class imbalance
    pos_weight = torch.tensor([(1 - train_labels.mean()) / train_labels.mean()]).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_epoch = 0
    patience = 10

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_losses = []
        for feat_batch, label_batch in train_loader:
            feat_batch = feat_batch.cuda()
            label_batch = label_batch.cuda()
            
            optimizer.zero_grad()
            pred = model(feat_batch)
            loss = criterion(pred, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feat_batch, label_batch in val_loader:
                pred = model(feat_batch.cuda()).cpu().numpy()
                val_preds.extend(pred)
                val_labels.extend(label_batch.numpy())
        
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1:3d}: Loss={np.mean(train_losses):.4f}  Val AUC={val_auc:.4f}")
        
        if use_wandb:
            wandb.log({"train_loss": np.mean(train_losses), "val_auc": val_auc})
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_classifier.pt")
        
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest Val AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
    return model, best_val_auc
```

#### `scripts/phase2_feature_ablation.py`

```python
"""
Feature ablation study: train a classifier with each feature group removed,
report AUC drop.  This produces the feature importance table for the paper.

Feature groups:
  0: mean_attn_received
  1: column_entropy
  2: cross_layer_consistency
  3: top_k_layer_frequency
  4: early_late_ratio
  5: late_layer_attn
  6: position (normalized + reverse)
  7: boundary_indicator
  8: generation_prefix_attn  ← the task-aware feature

Run after phase2_train_classifier.py has produced the full-feature baseline.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from semantickv.classifier.model import ImportanceClassifier, ImportanceDataset
from semantickv.classifier.train import load_training_data, train_classifier
from torch.utils.data import DataLoader

RESULTS_DIR = Path("./data/processed/phase1_results")
OUTPUT_DIR = Path("./experiments/feature_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "mean_attn_received",
    "column_entropy",
    "cross_layer_consistency",
    "top_k_layer_frequency",
    "early_late_ratio",
    "late_layer_attn",
    "position_normalized",
    "position_reversed",
    "boundary_indicator",
    "generation_prefix_attn",
]

N_FEATURES = len(FEATURE_NAMES)


def train_and_eval_without(feature_idx, train_f, train_l, val_f, val_l, output_dir):
    """Train classifier with feature `feature_idx` zeroed out; return val AUC."""
    # Zero out the feature column
    train_f_ablated = train_f.clone()
    val_f_ablated = val_f.clone()
    train_f_ablated[:, feature_idx] = 0.0
    val_f_ablated[:, feature_idx] = 0.0

    model, auc = train_classifier(
        train_f_ablated, train_l, val_f_ablated, val_l,
        output_dir / f"ablate_{feature_idx}",
        n_epochs=30,    # Fewer epochs for ablation runs
        use_wandb=False,
    )
    return auc


def main():
    print("Loading training data...")
    train_f, train_l, val_f, val_l = load_training_data(RESULTS_DIR)

    print("Training full-feature baseline...")
    _, baseline_auc = train_classifier(
        train_f, train_l, val_f, val_l,
        OUTPUT_DIR / "baseline",
        n_epochs=50,
        use_wandb=False,
    )
    print(f"Baseline AUC: {baseline_auc:.4f}")

    results = {"baseline_auc": baseline_auc, "ablations": {}}

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        print(f"Ablating feature: {feat_name} (idx {feat_idx})...")
        ablated_auc = train_and_eval_without(
            feat_idx, train_f, train_l, val_f, val_l, OUTPUT_DIR
        )
        drop = baseline_auc - ablated_auc
        results["ablations"][feat_name] = {
            "auc_without": ablated_auc,
            "auc_drop": drop,
        }
        print(f"  AUC without: {ablated_auc:.4f}  Drop: {drop:+.4f}")

    # Sort by importance (AUC drop descending)
    sorted_features = sorted(
        results["ablations"].items(),
        key=lambda x: x[1]["auc_drop"],
        reverse=True,
    )
    print("\n=== Feature Importance (by AUC drop) ===")
    for name, stats in sorted_features:
        print(f"  {name:35s}  drop={stats['auc_drop']:+.4f}")

    with open(OUTPUT_DIR / "feature_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

---

## Phase 3: Online Eviction Policy & Benchmark (Weeks 11-14)

### Goal

Replace SnapKV's heuristic with our trained classifier and benchmark across standard long-context tasks. Demonstrate that SemanticKV maintains higher task accuracy at the same compression ratio.

#### `semantickv/eviction/pyramidkv.py`

```python
"""
PyramidKV baseline (Cai et al., 2024).
Allocates more KV budget to higher layers (semantic) and less to lower layers
(syntactic), following a linear pyramid schedule.
"""

import numpy as np


class PyramidKVEviction:
    """
    Layer-wise budget allocation following a linear pyramid.
    Lower layers (syntactic) get min_budget; top layer gets full budget.
    Within each layer, token selection uses observation-window attention (SnapKV style).
    """

    def __init__(self, min_budget_fraction: float = 0.1, observation_window: int = 32):
        self.min_budget_fraction = min_budget_fraction
        self.observation_window = observation_window

    def get_keep_mask(
        self,
        attn_weights: np.ndarray,
        seq_len: int,
        keep_fraction: float = 0.3,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns boolean mask via pyramid-weighted voting across layers.
        attn_weights: [n_layers, n_heads, seq_len, seq_len]
        """
        n_layers = attn_weights.shape[0]

        # Per-layer budgets: linear from min at layer 0 to keep_fraction at layer n-1
        min_k = max(1, int(seq_len * self.min_budget_fraction))
        max_k = max(min_k, int(seq_len * keep_fraction))
        layer_budgets = np.linspace(min_k, max_k, n_layers, dtype=int)

        # Per-layer importance scores using observation window attention
        vote_counts = np.zeros(seq_len)
        for layer_idx in range(n_layers):
            layer_attn = attn_weights[layer_idx].mean(axis=0)  # [seq, seq]
            obs_scores = layer_attn[-self.observation_window:, :].mean(axis=0)
            k = layer_budgets[layer_idx]
            top_k_idx = np.argsort(obs_scores)[-k:]
            vote_counts[top_k_idx] += 1

        # Final mask: top-K positions by vote count
        final_k = max(1, int(seq_len * keep_fraction))
        top_idx = np.argsort(vote_counts)[-final_k:]
        mask = np.zeros(seq_len, dtype=bool)
        mask[top_idx] = True
        return mask
```

#### `semantickv/eviction/scissorhands.py`

```python
"""
ScissorHands baseline (Liu et al., 2023).
Identifies "pivot" tokens — those that receive consistently high attention —
and keeps them plus recent context.
"""

import numpy as np


class ScissorHandsEviction:
    """
    Eviction policy based on pivot token identification.
    Pivot tokens: positions with attention score above a percentile threshold
    in more than `consistency_threshold` fraction of attention heads/layers.
    """

    def __init__(self, consistency_threshold: float = 0.5, recent_window: int = 20):
        self.consistency_threshold = consistency_threshold
        self.recent_window = recent_window

    def get_keep_mask(
        self,
        attn_weights: np.ndarray,
        seq_len: int,
        keep_fraction: float = 0.3,
        **kwargs,
    ) -> np.ndarray:
        """
        attn_weights: [n_layers, n_heads, seq_len, seq_len]
        """
        n_layers, n_heads, _, _ = attn_weights.shape
        k = max(self.recent_window, int(seq_len * keep_fraction))

        # For each layer×head, compute attention received per token
        pivot_votes = np.zeros(seq_len)
        threshold_pct = (1 - keep_fraction) * 100

        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attn_weights[layer, head]  # [seq, seq]
                received = attn.sum(axis=0)       # [seq]
                threshold = np.percentile(received, threshold_pct)
                pivot_votes[received >= threshold] += 1

        # Pivot = consistently high across heads/layers
        total_votes = n_layers * n_heads
        is_pivot = pivot_votes / total_votes >= self.consistency_threshold

        # Combine: pivots + recent window
        mask = is_pivot.copy()
        mask[-self.recent_window:] = True

        # If we're over budget, trim to top-K by vote count
        if mask.sum() > k:
            top_idx = np.argsort(pivot_votes)[-k:]
            mask = np.zeros(seq_len, dtype=bool)
            mask[top_idx] = True
            mask[-self.recent_window:] = True

        return mask
```

#### `semantickv/eviction/semantickv.py`

```python
"""
SemanticKV eviction policy — drop-in replacement for H2O/SnapKV.

The policy runs the trained classifier on extracted features and
selects the top-K tokens to retain in the KV cache.

Latency budget: classifier inference should add <5% overhead.
The MLP with 64 hidden units on a seq_len=2048 prompt takes
~0.2ms on A100 — well within budget.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from semantickv.classifier.model import ImportanceClassifier
from semantickv.features.extractor import extract_all_features


class SemanticKVEviction:
    """
    Learned eviction policy using SemanticKV importance classifier.
    
    Usage:
        policy = SemanticKVEviction.from_checkpoint("path/to/best_classifier.pt")
        keep_mask = policy.get_keep_mask(attn_weights, seq_len, keep_fraction=0.3)
        # keep_mask: [seq_len] boolean tensor — True = keep this token's KV
    """

    def __init__(
        self,
        classifier: ImportanceClassifier,
        input_dim: int,
        boundary_token_ids: Optional[set] = None,
    ):
        self.classifier = classifier
        self.classifier.eval()
        self.input_dim = input_dim
        self.boundary_token_ids = boundary_token_ids
        self._model_ref = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        input_dim: int = 10,
        boundary_token_ids: Optional[set] = None,
    ):
        model = ImportanceClassifier(input_dim=input_dim)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.cuda()
        return cls(model, input_dim, boundary_token_ids)

    @torch.no_grad()
    def get_importance_scores(
        self,
        attn_weights: np.ndarray,
        seq_len: int,
        token_ids: Optional[np.ndarray] = None,
        generation_prefix_attn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict importance score for each token position.

        Args:
            attn_weights: [n_layers, n_heads, seq_len, seq_len]
            seq_len: sequence length
            token_ids: optional token IDs for positional/boundary features
            generation_prefix_attn: [seq_len] from compute_generation_prefix_features().
                Pass None when running without probe generation (ablation study use).

        Returns: [seq_len] importance scores in [0, 1]
        """
        features = extract_all_features(
            attn_weights, seq_len, token_ids,
            self.boundary_token_ids, generation_prefix_attn
        )
        features_t = torch.tensor(features, dtype=torch.float32).cuda()
        scores = torch.sigmoid(self.classifier(features_t)).cpu().numpy()
        return scores

    def set_model(self, model):
        self._model_ref = model

    def get_keep_mask(
        self,
        attn_weights: np.ndarray,
        seq_len: int,
        keep_fraction: float = 0.3,
        token_ids: Optional[np.ndarray] = None,
        always_keep_recent: int = 20,
        n_probe_tokens: int = 0,
        input_ids=None,
    ) -> np.ndarray:
        """
        Returns boolean mask: True = keep this token in KV cache.
        
        always_keep_recent: always keep the last N tokens regardless of score.
        This is a sensible inductive bias — recent context is almost always needed.
        n_probe_tokens: if > 0 and input_ids is not None, run probe generation
            to compute generation_prefix_attn before scoring.
        """
        gen_prefix = None
        if n_probe_tokens > 0 and input_ids is not None:
            from semantickv.features.generation_prefix import compute_generation_prefix_features
            gen_prefix = compute_generation_prefix_features(
                self._model_ref, input_ids, n_probe_tokens=n_probe_tokens
            )
        scores = self.get_importance_scores(attn_weights, seq_len, token_ids, gen_prefix)
        
        # Force-keep recent tokens
        scores[-always_keep_recent:] = scores.max() + 1.0
        
        # Select top-K by score
        k = max(always_keep_recent, int(seq_len * keep_fraction))
        k = min(k, seq_len)
        top_k_idx = np.argsort(scores)[-k:]
        
        mask = np.zeros(seq_len, dtype=bool)
        mask[top_k_idx] = True
        return mask
```

#### `scripts/phase3_benchmark.py`

```python
"""
Phase 3 Benchmark: SemanticKV vs H2O vs SnapKV vs Full KV Cache baseline.

Metrics:
- LongBench accuracy (F1 for QA tasks, ROUGE-L for summarization)
- Needle-in-a-Haystack recall
- KV cache memory reduction ratio
- Inference throughput (tokens/sec)

This script produces the main results table for the paper.
"""

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from rouge_score import rouge_scorer

from semantickv.model.loader import load_model
from semantickv.model.hooks import AttentionCapture
from semantickv.eviction.semantickv import SemanticKVEviction
from semantickv.eviction.h2o import H2OEviction
from semantickv.eviction.snapkv import SnapKVEviction


CLASSIFIER_PATH = "./experiments/checkpoints/best_classifier.pt"
RESULTS_DIR = Path("./experiments/phase3_results")
KEEP_FRACTIONS = [0.5, 0.3, 0.2]  # 50%, 30%, 20% KV retention
TASKS = ["2wikimqa_e", "qasper_e", "multi_news_e", "narrativeqa"]


def apply_kv_eviction(past_key_values, keep_mask):
    """
    Physically remove evicted token positions from past_key_values.

    keep_mask: [seq_len] boolean array — True = keep this position.
    Returns a new past_key_values tuple with only the kept positions.

    For RoPE models (Llama 3) this is safe: positional information is already
    baked into the key tensors at prefill time, so reindexing is not needed.
    """
    import torch
    keep_indices = torch.where(torch.tensor(keep_mask, dtype=torch.bool))[0]
    evicted = []
    for k, v in past_key_values:
        # k, v: [batch, heads, seq_len, head_dim]
        evicted.append((k[:, :, keep_indices, :], v[:, :, keep_indices, :]))
    return tuple(evicted)


def generate_with_eviction(model, prefill_logits, past_key_values, max_new_tokens=50):
    """
    Greedy generation from a (possibly evicted) KV cache.

    Accepts prefill_logits directly instead of re-running the last prompt token,
    which would create a duplicate entry in past_key_values for that position.

    Args:
        prefill_logits: [batch, vocab] — last-position logits from prefill
        past_key_values: evicted KV cache (already trimmed)
    Returns: [1, max_new_tokens] generated token IDs
    """
    import torch
    generated = []
    with torch.no_grad():
        # Sample first token from prefill logits — no extra forward pass
        next_token = prefill_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        past_kv = past_key_values

        for _ in range(max_new_tokens - 1):
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            next_logits = out.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            past_kv = out.past_key_values

    return torch.cat(generated, dim=1)  # [1, max_new_tokens]


# QA tasks use token-level F1; summarization uses ROUGE-L
_QA_TASKS = {"2wikimqa_e", "qasper_e", "narrativeqa", "hotpotqa"}


def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score for extractive QA."""
    from collections import Counter
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _bootstrap_ci(scores, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval for the mean."""
    scores = np.array(scores)
    means = np.array([
        np.mean(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    return float(np.mean(scores)), float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def evaluate_on_task(model, tokenizer, capture, policy, task_name, keep_fraction, n_samples=200):
    """
    Run evaluation on a single LongBench task with given eviction policy.

    Returns dict with mean score and 95% bootstrap CI.
    Uses token-level F1 for QA tasks and ROUGE-L for summarization.
    """
    import torch
    ds = load_dataset("THUDM/LongBench", task_name, split="test")
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    use_f1 = task_name in _QA_TASKS

    scores = []

    for item in list(ds)[:n_samples]:
        prompt = item["context"] + "\n\nQuestion: " + item.get("input", "")
        references = item["answers"] if item["answers"] else [""]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if policy is None:
                out = model.generate(inputs["input_ids"], max_new_tokens=50, do_sample=False)
                generated_ids = out[0][seq_len:]
            else:
                with capture.capture():
                    prefill_out = model(**inputs, output_attentions=True, use_cache=True)
                attn = capture.get_weights()   # [layers, batch, heads, seq, seq]
                capture.clear()
                past_kv = prefill_out.past_key_values
                prefill_logits = prefill_out.logits[:, -1, :]  # [batch, vocab]

                if attn is not None:
                    attn_np = attn[:, 0, :, :, :].numpy()
                    keep_mask = policy.get_keep_mask(
                        attn_np,
                        seq_len,
                        keep_fraction=keep_fraction,
                        token_ids=inputs["input_ids"][0].cpu().numpy(),
                    )
                    past_kv = apply_kv_eviction(past_kv, keep_mask)

                generated_ids = generate_with_eviction(
                    model, prefill_logits, past_kv, max_new_tokens=50
                )[0]

        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if use_f1:
            # Take max F1 over all reference answers (SQuAD convention)
            score = max(_token_f1(prediction, ref) for ref in references)
        else:
            score = rouge.score(references[0], prediction)["rougeL"].fmeasure

        scores.append(score)

    mean, ci_lo, ci_hi = _bootstrap_ci(scores)
    return {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(scores), "metric": "F1" if use_f1 else "ROUGE-L"}


N_SAMPLES = 200          # Enough for meaningful bootstrap CIs
KEEP_FRACTIONS = [0.5, 0.3, 0.2]
TASKS = ["2wikimqa_e", "qasper_e", "multi_news_e", "narrativeqa"]


def profile_inference_latency(model, tokenizer, policy, capture, keep_fraction=0.3, n_trials=25):
    """
    Measure end-to-end per-token latency using CUDA events.
    Compares full-KV baseline vs. eviction policy on a synthetic 2048-token prompt.
    Returns latency stats in milliseconds.
    """
    import torch
    dummy_ids = torch.randint(1000, 30000, (1, 2048), device=model.device)

    def timed_run(use_policy):
        torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                if not use_policy:
                    out = model(dummy_ids, use_cache=True)
                    past_kv = out.past_key_values
                    generate_with_eviction(model, out.logits[:, -1, :], past_kv, max_new_tokens=50)
                else:
                    with capture.capture():
                        prefill_out = model(dummy_ids, output_attentions=True, use_cache=True)
                    attn = capture.get_weights()
                    capture.clear()
                    past_kv = prefill_out.past_key_values
                    if attn is not None:
                        attn_np = attn[:, 0].numpy()
                        keep_mask = policy.get_keep_mask(attn_np, dummy_ids.shape[1], keep_fraction)
                        past_kv = apply_kv_eviction(past_kv, keep_mask)
                    generate_with_eviction(model, prefill_out.logits[:, -1, :], past_kv, max_new_tokens=50)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return np.array(times)

    baseline_ms = timed_run(use_policy=False)
    policy_ms = timed_run(use_policy=True)

    return {
        "baseline_mean_ms": float(baseline_ms.mean()),
        "baseline_std_ms": float(baseline_ms.std()),
        "policy_mean_ms": float(policy_ms.mean()),
        "policy_std_ms": float(policy_ms.std()),
        "overhead_pct": float((policy_ms.mean() - baseline_ms.mean()) / baseline_ms.mean() * 100),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model("./models/llama3-8b-instruct")
    capture = AttentionCapture(model)

    from semantickv.features.extractor import get_boundary_token_ids
    boundary_ids = get_boundary_token_ids(tokenizer)

    print("Loading eviction policies...")
    semantickv = SemanticKVEviction.from_checkpoint(
        CLASSIFIER_PATH, boundary_token_ids=boundary_ids
    )
    semantickv.set_model(model)
    from semantickv.eviction.pyramidkv import PyramidKVEviction
    from semantickv.eviction.scissorhands import ScissorHandsEviction

    policies = {
        "Full KV (baseline)": None,
        "H2O": H2OEviction(),
        "SnapKV": SnapKVEviction(),
        "PyramidKV": PyramidKVEviction(),
        "ScissorHands": ScissorHandsEviction(),
        "SemanticKV (ours)": semantickv,
    }

    results_table = {}

    for task in TASKS:
        results_table[task] = {}
        for policy_name, policy in policies.items():
            if policy is None:
                result = evaluate_on_task(model, tokenizer, capture, None, task, 1.0, N_SAMPLES)
                results_table[task]["Full KV"] = result
                print(f"  {task} | Full KV: {result['mean']:.4f} [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}] ({result['metric']})")
            else:
                for kf in KEEP_FRACTIONS:
                    result = evaluate_on_task(model, tokenizer, capture, policy, task, kf, N_SAMPLES)
                    key = f"{policy_name} @{int(kf*100)}%"
                    results_table[task][key] = result
                    print(f"  {task} | {key}: {result['mean']:.4f} [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")

    with open(RESULTS_DIR / "main_results.json", "w") as f:
        json.dump(results_table, f, indent=2)

    # Latency profiling (SemanticKV vs Full KV at 30% retention)
    print("\n=== Latency Profiling (seq_len=2048, 50 new tokens) ===")
    latency = profile_inference_latency(model, tokenizer, semantickv, capture, keep_fraction=0.3)
    print(f"Full KV:    {latency['baseline_mean_ms']:.1f} ± {latency['baseline_std_ms']:.1f} ms")
    print(f"SemanticKV: {latency['policy_mean_ms']:.1f} ± {latency['policy_std_ms']:.1f} ms")
    print(f"Overhead:   {latency['overhead_pct']:+.1f}%")
    with open(RESULTS_DIR / "latency_profile.json", "w") as f:
        json.dump(latency, f, indent=2)

    print("\nAll results saved. Main table and latency profile ready for paper.")


if __name__ == "__main__":
    main()
```

---

## Phase 4: Paper & Submission (Weeks 15-16)

### Target Venues

| Venue | Deadline (est.) | Fit |
|---|---|---|
| MLSys 2027 | ~October 2026 | Primary — systems + ML co-design |
| ICLR 2027 | ~October 2026 | Strong if theoretical framing is sharp |
| ACL Findings 2027 | ~February 2027 | Fallback if NLP application angle is emphasized |
| arXiv preprint | Before October 2026 | **Critical — this is what PhD apps see** |

### Paper Outline

```
Title: SemanticKV: Counterfactual Importance Supervision for Task-Aware KV Cache Eviction

Abstract (4 sentences):
  1. Problem: KV cache memory is the primary bottleneck in long-context LLM serving
  2. Gap: Existing eviction methods rely on attention-score heuristics that poorly predict
     actual generation importance — we quantify this gap empirically
  3. Method: Counterfactual ablation labels + lightweight MLP classifier trained on
     cross-layer consistency, column entropy, and generation-prefix signals
  4. Result: SemanticKV achieves X F1 / ROUGE-L points above SnapKV at 70% compression
     on LongBench, with <5% latency overhead, generalizing across Llama 3 and Mistral 7B

1. Introduction
   - Hook: KV memory at scale — one request, 21GB
   - The heuristic trap: high prefill attention ≠ generation necessity
   - Our insight: causal importance is directly measurable via counterfactual ablation
   - Contributions (4 bullet points):
       (i) first counterfactual importance dataset for KV eviction
       (ii) generation-prefix signal: a task-aware feature no prior work uses
       (iii) SemanticKV: a lightweight learned policy with real latency measurements
       (iv) systematic feature ablation showing cross-layer consistency and
            generation-prefix drive most of the improvement over heuristics

2. Related Work
   - KV cache eviction heuristics: H2O, SnapKV, PyramidKV, ScissorHands
   - Learned KV compression: CSKV, KVSharer — distinguish from our approach
   - Counterfactual methods in NLP (influence functions, rationale extraction)
   - Why this is different: supervised signal from output KL divergence, not rank/SVD

3. Counterfactual Importance Estimation
   - Formal definition: importance(i) = E[KL(P_baseline || P_ablate_i)]
   - Ablation methodology (attention mask + v-zeroing for correct softmax normalization)
   - Analysis: H2O/SnapKV AUC vs ground truth — the opening figure
   - Figure 1: scatter plot, heuristic rank vs counterfactual rank, colored by task type
   - Key finding: AUC gap is larger for multi-hop QA than summarization (task-dependence)

4. Feature Engineering
   - 10-feature taxonomy with computation cost table
   - The generation-prefix signal: why it's strictly more informative than prefill attention
   - Figure 2: example showing low-prefill-attention tokens that get high probe-generation attention
   - Table: feature ablation study — which features drive improvement

5. SemanticKV
   - Classifier architecture (3-layer MLP with LayerNorm, 10→64→64→1)
   - Training: BCEWithLogitsLoss, prompt-level train/val split, cosine LR
   - Integration: prefill → probe generation (5 tokens) → classifier → evict → generate
   - Latency budget analysis: probe adds ~5ms, classifier adds ~0.2ms on A100

6. Experiments
   - Models: Llama 3 8B Instruct, Mistral 7B v0.3 (generalization)
   - Tasks: LongBench (2WikiMQA, Qasper, MultiNews, NarrativeQA), NIAH
   - Metrics: F1 for QA, ROUGE-L for summarization, 95% bootstrap CI, n=200/task
   - Compression ratios: 20%, 30%, 50% KV retention
   - Baselines: Full KV, H2O, SnapKV, PyramidKV, ScissorHands
   - Main results table (accuracy + latency overhead column)
   - Ablation: w/o generation prefix, w/o cross-layer features, w/o learned (→ heuristic)

7. Analysis
   - Task-type breakdown: where does SemanticKV help most and why
   - Failure modes: when does it not help? (very short contexts, repetitive prompts)
   - Does generalization hold on Mistral? (zero-shot transfer without retraining)
   - Compute-quality tradeoff curve: n_probe_tokens ∈ {0, 3, 5, 10}

8. Conclusion
   - Counterfactual supervision is a general principle applicable beyond KV eviction
   - Limitations: trained on Llama 3 8B, probe generation adds overhead for very short contexts
   - Future work: multi-query attention, speculative decoding integration
```

---

## Compute Budget & Timeline

### Detailed Cost Estimate (RunPod A100 80GB @ ~$1.50/hr)

| Phase | Task | Est. Hours | Est. Cost |
|---|---|---|---|
| Phase 1 | Counterfactual ablations (50 prompts, 50% sampling) | 44 | $66 |
| Phase 1 | Attention capture + heuristic evaluation | 3 | $4.50 |
| Phase 2 | Generation prefix feature extraction (200 prompts) | 12 | $18 |
| Phase 2 | Classifier training + feature ablation study | 6 | $9 |
| Phase 3 | Full benchmark (4 tasks × 6 policies, n=200, Llama 3 8B) | 24 | $36 |
| Phase 3 | Generalization benchmark (Mistral 7B v0.3, 2 tasks) | 8 | $12 |
| Phase 3 | NIAH evaluation | 5 | $7.50 |
| Buffer | Reruns, hyperparameter search, debugging | 15 | $22.50 |
| **Total** | | **~117 hrs** | **~$175** |

> **Why the original $87 estimate was wrong:** 50 prompts × ~450 ablated positions × ~7s/pass ≈ 44 GPU hours for Phase 1 alone. Always estimate ablation costs as O(sample_fraction × seq_len × max_new_tokens × n_prompts / throughput).

### Week-by-Week Timeline

```
Week 1:    Environment setup (RunPod, Llama 3 8B, verify GPU); run ablations on 5 prompts as sanity check
Week 2-3:  Complete Phase 1 ablations (50 prompts); generate opening figure (heuristic vs counterfactual scatter)
Week 4:    Feature extraction pipeline: attention features + positional features; verify shapes
Week 5:    Implement generation prefix feature; profile overhead; integrate into extractor
Week 6-7:  Expand counterfactual dataset to 200 prompts (needed for training)
Week 8:    Train full-feature classifier; validate prompt-level train/val split
Week 9:    Feature ablation study (10 ablation runs); generate feature importance table
Week 10:   Classifier evaluation vs heuristics; tune n_probe_tokens tradeoff
Week 11:   Implement PyramidKV and ScissorHands baselines; verify correctness
Week 12:   Run LongBench benchmark (Llama 3 8B, 6 policies, n=200, 4 tasks)
Week 13:   NIAH evaluation; latency profiling; memory measurement
Week 14:   Generalization: run 2-task benchmark on Mistral 7B (no retraining)
Week 15:   Result analysis; generate all paper figures
Week 16-17: Paper draft
Week 18:   Internal review, revisions
Week 19:   arXiv submission; conference submission prep
```

---

## What Success Looks Like

### Minimum Publishable Result

SemanticKV achieves **≥3 ROUGE-L points higher** than SnapKV at 30% KV retention on LongBench multi-hop QA. This would be a clear, defensible contribution.

### Strong Result

SemanticKV matches full KV cache performance at 50% retention while SnapKV requires 70% retention to achieve the same score. This would make the paper compelling for MLSys or ICLR.

### PhD Application Angle

Even the minimum result is enough. The narrative is:

> "I identified that attention-score heuristics systematically fail to predict true token importance, developed a counterfactual labeling methodology to generate ground truth importance signals, and trained a lightweight learned policy that demonstrably outperforms heuristics. This is documented in a preprint at arXiv:XXXX."

That sentence, backed by a real preprint with real numbers, is what separates a competitive Stanford PhD application from a strong one.

---

## Key Papers to Read Before Starting

1. **H2O** (Zhang et al., 2023) — the heavy hitter + recent attention baseline
2. **SnapKV** (Li et al., 2024) — observation window approach
3. **PyramidKV** (Cai et al., 2024) — layer-wise budget allocation
4. **ScissorHands** (Liu et al., 2023) — pivot token eviction
5. **TurboQuant** (Zandieh et al., 2025) — KV compression (complementary, not competing)
6. **LongBench** (Bai et al., 2023) — the evaluation benchmark we use
7. **Orca** (Yu et al., 2022) — KV scheduling in serving systems (systems context)

---

*SemanticKV Research Blueprint v1.0 — for 2027 PhD Applications*
