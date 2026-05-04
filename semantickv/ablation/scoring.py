"""
Utilities to compare heuristic eviction predictions against ground truth
counterfactual importance scores.

This is where we quantify the research gap — the mismatch between what
H2O/SnapKV would evict and what we know actually matters.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict


def binarize_importance(
    importance_scores: np.ndarray,
    keep_fraction: float = 0.3,
) -> np.ndarray:
    """
    Top keep_fraction tokens → label 1 (important), rest → 0.
    Mirrors the real eviction setting where we keep a fixed budget of K tokens.
    """
    threshold = np.percentile(importance_scores, (1 - keep_fraction) * 100)
    return (importance_scores >= threshold).astype(int)


def compute_h2o_scores(
    attention_weights: np.ndarray,
    recent_window: int = 20,
) -> np.ndarray:
    """
    H2O score: cumulative attention received + recency bonus.
    attention_weights: [heads, seq_len, seq_len] averaged over layers.
    """
    avg_attn = attention_weights.mean(axis=0)       # [seq_len, seq_len]
    heavy_scores = avg_attn.sum(axis=0)             # [seq_len] — column sum

    recency_bonus = np.zeros(len(heavy_scores))
    recency_bonus[-recent_window:] = heavy_scores.max() * 10

    return heavy_scores + recency_bonus


def compute_snapkv_scores(
    attention_weights: np.ndarray,
    observation_window: int = 32,
) -> np.ndarray:
    """
    SnapKV score: mean attention from the last observation_window query tokens
    to all key positions, using the last layer.
    attention_weights: [n_layers, n_heads, seq_len, seq_len]
    """
    last_layer_attn = attention_weights[-1]                         # [heads, seq, seq]
    obs_queries = last_layer_attn[:, -observation_window:, :]       # [heads, obs, seq]
    return obs_queries.mean(axis=0).mean(axis=0)                    # [seq_len]


def evaluate_heuristic(
    ground_truth_scores: np.ndarray,
    heuristic_scores: np.ndarray,
    keep_fraction: float = 0.3,
    name: str = "heuristic",
) -> Dict:
    """
    Evaluate how well a heuristic predicts true token importance.

    Only positions with a real ground-truth label are scored. Unsampled
    positions are NaN by convention (see compute_importance_scores) and
    masked out — including them as zeros would bias AUC.

    Metrics:
    - AUC-ROC: overall ranking quality (target: < 0.75 to show the gap)
    - Average Precision: precision-recall quality
    - Top-K Recall: of the truly important tokens, what fraction does the
      heuristic correctly identify in its top-K?
    """
    valid = ~np.isnan(ground_truth_scores)
    gt = ground_truth_scores[valid]
    h = np.nan_to_num(heuristic_scores, nan=0.0, posinf=0.0, neginf=0.0)[valid]

    binary_gt = binarize_importance(gt, keep_fraction)

    h_norm = h - h.min()
    if h_norm.max() > 0:
        h_norm = h_norm / h_norm.max()

    if len(np.unique(binary_gt)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = roc_auc_score(binary_gt, h_norm)
        ap = average_precision_score(binary_gt, h_norm)

    k = max(1, int(keep_fraction * len(gt)))
    heuristic_top_k = set(np.argsort(h)[-k:])
    true_top_k = set(np.argsort(gt)[-k:])
    top_k_recall = len(heuristic_top_k & true_top_k) / max(len(true_top_k), 1)

    return {
        "name": name,
        "auc_roc": auc,
        "average_precision": ap,
        "top_k_recall": top_k_recall,
        "keep_fraction": keep_fraction,
        "n_evaluated": int(valid.sum()),
    }
