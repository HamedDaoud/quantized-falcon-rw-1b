"""Loads Falcon-RW 1B with NF4 4-bit quantization when CUDA is available,
falling back to bfloat16 on MPS (Apple Silicon) or float32 on CPU."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.config import MODEL_ID, NF4_QUANT_CONFIG

_CACHE: dict[bool, tuple] = {}


def _resolve_quantization(use_quantization: bool | None) -> bool:
    if use_quantization is None:
        return torch.cuda.is_available()
    if use_quantization and not torch.cuda.is_available():
        raise RuntimeError(
            "NF4 quantization via bitsandbytes requires CUDA. "
            "Run on a CUDA-enabled GPU, or call load_model(use_quantization=False)."
        )
    return bool(use_quantization)


def load_model(use_quantization: bool | None = None):
    """Load (model, tokenizer) for Falcon-RW 1B.

    use_quantization=None (default): auto — quantize iff CUDA is available.
    use_quantization=True: force NF4 4-bit (requires CUDA).
    use_quantization=False: load full-precision weights (MPS/CPU fallback path).
    """
    quantize = _resolve_quantization(use_quantization)

    if quantize in _CACHE:
        return _CACHE[quantize]

    if quantize:
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                **NF4_QUANT_CONFIG,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
    elif torch.backends.mps.is_available():
        kwargs = {"torch_dtype": torch.float16, "device_map": {"": "mps"}}
    else:
        # CPU path: float16 saves ~50% memory so the model fits comfortably in
        # Docker Desktop's default RAM allocation (~4 GB). Inference is slow on
        # CPU regardless of dtype.
        kwargs = {"torch_dtype": torch.float16, "device_map": {"": "cpu"}}

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _CACHE[quantize] = (model, tokenizer)
    return model, tokenizer
