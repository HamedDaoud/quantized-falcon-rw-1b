"""Text generation entrypoint shared by the FastAPI service and the Gradio UI."""

from __future__ import annotations

import torch

from app.config import INFERENCE_DEFAULTS


def generate_text(model, tokenizer, prompt: str, seed: int = 42, **overrides) -> str:
    torch.manual_seed(seed)

    settings = {**INFERENCE_DEFAULTS, **overrides}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **settings,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
