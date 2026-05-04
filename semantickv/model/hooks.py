"""
Forward hooks to capture attention statistics during a single forward pass.

Instead of storing full [layers, heads, seq, seq] attention matrices
(38 GB for seq_len=4K on 32-layer model — causes OOM), we accumulate
H2O and SnapKV statistics incrementally per layer and immediately free
each layer's GPU attention tensor via hook output replacement.
"""

import numpy as np
import torch
from contextlib import contextmanager
from typing import Optional


class AttentionCapture:
    """
    Captures H2O and SnapKV attention statistics during a single forward pass.

    The hook fires once per attention layer. It:
      1. Accumulates column sums (for H2O) across layers.
      2. Overwrites the stored observation-window rows with the current layer
         (so only the last layer's rows are kept — this is what SnapKV uses).
      3. Replaces attn_weights in the layer output with None so PyTorch frees
         the GPU tensor before the next layer runs.

    Peak extra VRAM: one layer's [heads, seq, seq] at a time (~1.2 GB for
    seq_len=4K on Llama 3 8B) rather than all 32 layers simultaneously.

    Usage:
        capture = AttentionCapture(model)
        with capture.capture():
            _ = model(**inputs, output_attentions=True)
        h2o_scores   = capture.get_h2o_scores()    # [seq_len]
        snapkv_scores = capture.get_snapkv_scores() # [seq_len]
    """

    def __init__(self, model, obs_window: int = 32, recent_window: int = 20):
        self.model = model
        self.obs_window = obs_window
        self.recent_window = recent_window
        self._hooks = []
        self._h2o_sum: Optional[np.ndarray] = None
        self._h2o_count: int = 0
        self._snapkv_last: Optional[np.ndarray] = None  # [heads, obs, seq]

    def _hook_fn(self, module, input, output):
        if not (isinstance(output, tuple) and len(output) > 1 and output[1] is not None):
            return output

        attn_w = output[1]  # [batch, heads, seq, seq]

        with torch.no_grad():
            # Sum on GPU in fp16 (intermediate values within fp16 range), then
            # cast the final [seq] vector to fp32 before numpy so accumulation
            # across 32 layers can't overflow fp16's 65504 limit.

            # H2O: head-averaged column sum for this layer  →  [seq] in fp32
            col = attn_w[0].mean(0).sum(0).float().cpu().numpy()
            if self._h2o_sum is None:
                self._h2o_sum = col.copy()
            else:
                self._h2o_sum += col
            self._h2o_count += 1

            # SnapKV: keep last obs_window query rows of this layer  →  [heads, obs, seq]
            obs = min(self.obs_window, attn_w.shape[2])
            self._snapkv_last = attn_w[0, :, -obs:, :].float().cpu().numpy()

        # Replace with None so PyTorch frees the GPU tensor immediately
        return (output[0], None) + output[2:]

    @contextmanager
    def capture(self):
        """Context manager: resets state, installs hooks, removes them on exit."""
        self._h2o_sum = None
        self._h2o_count = 0
        self._snapkv_last = None
        try:
            for layer in self.model.model.layers:
                h = layer.self_attn.register_forward_hook(self._hook_fn)
                self._hooks.append(h)
            yield self
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

    def get_h2o_scores(self) -> Optional[np.ndarray]:
        """H2O score: mean column-sum across layers + recency bonus. Shape: [seq_len]."""
        if self._h2o_sum is None:
            return None
        heavy = self._h2o_sum / max(self._h2o_count, 1)
        recency_bonus = np.zeros_like(heavy)
        recency_bonus[-self.recent_window:] = heavy.max() * 10
        return heavy + recency_bonus

    def get_snapkv_scores(self) -> Optional[np.ndarray]:
        """SnapKV score: mean of last-layer obs-window attention. Shape: [seq_len]."""
        if self._snapkv_last is None:
            return None
        return self._snapkv_last.mean(axis=0).mean(axis=0)

    def clear(self):
        self._h2o_sum = None
        self._h2o_count = 0
        self._snapkv_last = None
