import torch
from app.config import DEVICE, INFERENCE_SETTINGS

def generate_text(model, tokenizer, prompt, seed=42):
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=INFERENCE_SETTINGS["max_new_tokens"],
        do_sample=True,
        temperature=INFERENCE_SETTINGS["temperature"],
        top_p=INFERENCE_SETTINGS["top_p"],
        pad_token_id=tokenizer.pad_token_id  # avoids warning
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)