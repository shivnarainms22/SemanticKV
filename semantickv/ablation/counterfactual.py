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

    NOTE: cast to fp32 before softmax/log. fp16's smallest positive subnormal
    is ~6e-8, so any clamp(min=1e-10) on fp16 probs is a no-op — small softmax
    outputs underflow to 0, log(0) = -inf, and NaN propagates through KL.
    """
    baseline_logits = baseline_logits.float()
    ablated_logits  = ablated_logits.float()

    eps = 1e-10
    baseline_probs = F.softmax(baseline_logits, dim=-1).clamp(min=eps)
    ablated_probs  = F.softmax(ablated_logits,  dim=-1).clamp(min=eps)

    # Manual KL to avoid F.kl_div argument-order confusion:
    # KL(P || Q) = sum P * (log P - log Q)
    kl_per_step = (baseline_probs * (baseline_probs.log() - ablated_probs.log())).sum(dim=-1)

    return kl_per_step.clamp(min=0).mean().item()


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

    # NaN = unsampled (unknown). Distinct from 0 (sampled, low impact).
    # evaluate_heuristic masks out NaNs so AUC isn't biased by unlabeled positions.
    importance_scores = np.full(seq_len, np.nan)
    importance_scores[:skip_first_n] = 1.0
    importance_scores[seq_len - skip_last_n:] = 1.0

    for step, pos in enumerate(tqdm(positions_to_test, desc=f"Ablating (seq_len={seq_len})")):
        ablated_logits = run_ablated_forward(model, input_ids, pos, max_new_tokens)
        importance_scores[pos] = compute_output_divergence(baseline_logits, ablated_logits)
        del ablated_logits
        # Each ablated forward allocates a fresh past_key_values cache (~GBs at long
        # seq_len). Allocator reservations grow without bound otherwise.
        if (step + 1) % 500 == 0:
            torch.cuda.empty_cache()

    return {
        "tokens": tokens,
        "importance_scores": importance_scores,
        "baseline_logits": baseline_logits.cpu().numpy(),
        "seq_len": seq_len,
        "prompt": prompt,
    }
