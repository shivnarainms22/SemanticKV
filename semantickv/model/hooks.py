"""
Forward hooks to capture attention weights during a single forward pass
without modifying model internals.
"""

import torch
from typing import List, Optional
from contextlib import contextmanager


class AttentionCapture:
    """
    Captures attention weight matrices from all layers during a forward pass
    via PyTorch forward hooks.

    Usage:
        capture = AttentionCapture(model)
        with capture.capture():
            outputs = model(**inputs, output_attentions=True)
        attn_weights = capture.get_weights()  # [num_layers, batch, heads, seq, seq]
    """

    def __init__(self, model):
        self.model = model
        self._weights: List[torch.Tensor] = []
        self._hooks = []

    def _hook_fn(self, module, input, output):
        """
        Registered on each self_attn layer.
        output[1] is the attention weight matrix when output_attentions=True.
        Move to CPU immediately to avoid accumulating on GPU across 32 layers.
        """
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            # Shape: [batch, heads, seq_len, seq_len]
            self._weights.append(output[1].detach().cpu())

    @contextmanager
    def capture(self):
        """Context manager: installs hooks before yield, removes them after."""
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
        Returns None if no weights were captured.
        """
        if not self._weights:
            return None
        return torch.stack(self._weights, dim=0)

    def clear(self):
        self._weights.clear()
