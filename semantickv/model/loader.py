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
    Load a CausalLM with attention weights exposed.

    NOTE: We use attn_implementation="eager" intentionally.
    Flash Attention does not return attention weight matrices,
    which we need for feature extraction. The memory cost is
    acceptable for 8B at float16 on A100 80GB.
    Works with any AutoModelForCausalLM (Llama 3, Mistral, etc.).
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
