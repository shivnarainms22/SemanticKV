"""
Counterfactual eviction test — the core of Phase 1.

For each token in a prompt's KV cache, we measure how much the model output
changes when that token's KV entry is removed. This gives us a ground truth
importance score for every token.

High output divergence when a token is removed = high importance.
Low output divergence = that token's KV entry is expendable.

Computationally expensive (O(seq_len) forward passes per prompt), so we use
stratified sampling via sample_fraction.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
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


def _zero_value_tensor(layer, position: int) -> bool:
    """Try to zero value tensor at `position` in a cache layer object. Returns True if zeroed."""
    for attr in ('values', 'value_states', 'value_cache', 'v_cache', '_v_cache'):
        v = getattr(layer, attr, None)
        if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[2] > position:
            v_new = v.clone()
            v_new[:, :, position, :] = 0.0
            try:
                setattr(layer, attr, v_new)
                return True
            except (AttributeError, TypeError):
                pass
    return False


def _ablate_kv(past_kv, ablate_position: int):
    """
    Zero v at ablate_position in past_key_values.

    Handles three cache formats:
      - Legacy tuple-of-tuples (pre-transformers-4.x)
      - transformers 4.x/5.0-5.6 DynamicCache (.key_cache / .value_cache lists)
      - transformers 5.7+ DynamicCache (.layers list of per-layer cache objects)

    NOTE: run_ablated_forward already sets attention_mask=0 at ablate_position,
    which zeroes the attention weight via softmax. The v-zeroing here is
    belt-and-suspenders; if we cannot reach the value tensors we return the
    cache unchanged and rely on the mask alone.
    """
    if isinstance(past_kv, tuple):
        # Legacy tuple-of-tuples; use index access to handle layers with >2 elements.
        ablated = []
        for layer_kv in past_kv:
            layer = list(layer_kv)
            layer[1] = layer[1].clone()
            layer[1][:, :, ablate_position, :] = 0.0
            ablated.append(tuple(layer))
        return tuple(ablated)

    # Modern Cache object — mutate in-place and return the same object.

    # Pattern A: transformers 4.x / 5.0–5.6 DynamicCache
    if hasattr(past_kv, 'value_cache') and isinstance(past_kv.value_cache, list):
        for i in range(len(past_kv.value_cache)):
            v = past_kv.value_cache[i]
            if isinstance(v, torch.Tensor) and v.shape[2] > ablate_position:
                past_kv.value_cache[i] = v.clone()
                past_kv.value_cache[i][:, :, ablate_position, :] = 0.0
        return past_kv

    # Pattern B: transformers 5.7+ DynamicCache with per-layer objects
    if hasattr(past_kv, 'layers') and past_kv.layers:
        for layer in past_kv.layers:
            _zero_value_tensor(layer, ablate_position)
        return past_kv

    # Unknown structure — return unchanged; attention_mask already handles exclusion.
    return past_kv


def run_ablated_forward(
    model,
    input_ids: torch.Tensor,
    ablate_position: int,
    max_new_tokens: int = 50,
) -> torch.Tensor:
    """
    Run generation with token at `ablate_position` removed from the input.

    This is the correct counterfactual for KV cache eviction: the evicted token
    is absent from both key lookup (attention weights) and value aggregation.
    We remove it from the prompt entirely and run normal greedy generation,
    which avoids all cache-surgery complexity and NaN issues from masked softmax.

    NOTE: tokens after ablate_position are shifted left by one, so their
    RoPE positions change by 1. This is a minor effect vs. the information
    content difference being measured.

    Returns: [max_new_tokens, vocab_size]
    """
    ablated_ids = torch.cat([
        input_ids[:, :ablate_position],
        input_ids[:, ablate_position + 1:],
    ], dim=1)
    return run_baseline_forward(model, ablated_ids, max_new_tokens)


def compute_output_divergence(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
) -> float:
    """
    KL(P_baseline || P_ablated) averaged across all generation steps.
    Higher divergence = ablated token had more causal impact on generation.
    """
    eps = 1e-10
    baseline_probs = F.softmax(baseline_logits, dim=-1).clamp(min=eps)
    ablated_probs  = F.softmax(ablated_logits,  dim=-1).clamp(min=eps)

    kl_per_step = F.kl_div(
        ablated_probs.log(),
        baseline_probs,
        reduction='none'
    ).clamp(min=0).sum(dim=-1)   # clamp kills rare fp negatives near 0

    return kl_per_step.mean().item()


def compute_importance_scores(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    sample_fraction: float = 1.0,
    skip_first_n: int = 5,
    skip_last_n: int = 10,
) -> Dict:
    """
    Compute counterfactual importance for all (sampled) tokens in a prompt.

    Args:
        prompt: Input text
        sample_fraction: Fraction of positions to ablate (use 0.5 for speed)
        skip_first_n: Skip BOS + early system tokens (always treated as important)
        skip_last_n: Skip most recent tokens (always important by recency)

    Returns dict with:
        tokens, importance_scores [seq_len], baseline_logits, seq_len, prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    baseline_logits = run_baseline_forward(model, input_ids, max_new_tokens)

    positions_to_test = list(range(skip_first_n, seq_len - skip_last_n))

    if not positions_to_test:
        # Prompt too short for the skip margins — score everything as important.
        importance_scores = np.ones(seq_len)
        return {
            "tokens": tokens,
            "importance_scores": importance_scores,
            "baseline_logits": baseline_logits.cpu().numpy(),
            "seq_len": seq_len,
            "prompt": prompt,
        }

    if sample_fraction < 1.0:
        n_sample = max(1, int(len(positions_to_test) * sample_fraction))
        positions_to_test = np.random.choice(
            positions_to_test, n_sample, replace=False
        ).tolist()

    importance_scores = np.zeros(seq_len)
    importance_scores[:skip_first_n] = 1.0
    importance_scores[seq_len - skip_last_n:] = 1.0

    for pos in tqdm(positions_to_test, desc=f"Ablating (seq_len={seq_len})"):
        ablated_logits = run_ablated_forward(model, input_ids, pos, max_new_tokens)
        importance_scores[pos] = compute_output_divergence(baseline_logits, ablated_logits)

    return {
        "tokens": tokens,
        "importance_scores": importance_scores,
        "baseline_logits": baseline_logits.cpu().numpy(),
        "seq_len": seq_len,
        "prompt": prompt,
    }
