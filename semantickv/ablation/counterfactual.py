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


def _ablate_kv(past_kv, ablate_position: int):
    """
    Zero v at ablate_position in past_key_values.
    Handles both transformers 5.x DynamicCache and legacy tuple-of-tuples.
    """
    if hasattr(past_kv, 'key_cache'):
        # transformers >= 5.x: DynamicCache with .key_cache / .value_cache lists
        for i in range(len(past_kv.key_cache)):
            past_kv.value_cache[i] = past_kv.value_cache[i].clone()
            past_kv.value_cache[i][:, :, ablate_position, :] = 0.0
        return past_kv
    else:
        # Legacy tuple-of-tuples; first two elements per layer are (key, value).
        # Use index access instead of unpacking to handle layers with >2 elements.
        ablated = []
        for layer_kv in past_kv:
            layer = list(layer_kv)
            layer[1] = layer[1].clone()
            layer[1][:, :, ablate_position, :] = 0.0
            ablated.append(tuple(layer))
        return tuple(ablated)


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

    Returns: [max_new_tokens, vocab_size]
    """
    seq_len = input_ids.shape[1]
    scores = []

    with torch.no_grad():
        # Prefill mask: [batch, seq_len], 1=attend, 0=exclude
        prefill_mask = torch.ones(1, seq_len, dtype=torch.long, device=input_ids.device)
        prefill_mask[:, ablate_position] = 0

        prefill_out = model(input_ids, attention_mask=prefill_mask, use_cache=True)
        past_kv = _ablate_kv(prefill_out.past_key_values, ablate_position)

        # First generation step reuses prefill logits — no extra forward pass
        next_logits = prefill_out.logits[:, -1, :]
        scores.append(next_logits.squeeze(0))
        next_token = next_logits.argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens - 1):
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
    KL(P_baseline || P_ablated) averaged across all generation steps.
    Higher divergence = ablated token had more causal impact on generation.
    """
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)

    kl_per_step = F.kl_div(
        ablated_probs.log(),
        baseline_probs,
        reduction='none'
    ).sum(dim=-1)   # [steps]

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
