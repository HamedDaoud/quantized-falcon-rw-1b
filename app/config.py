import torch

# Model
MODEL_ID = "tiiuae/falcon-rw-1b"

# Quantization config
QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
    "llm_int8_threshold": 6.0,
    "llm_int8_has_fp16_weight": False
}

# Inference defaults
INFERENCE_SETTINGS = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
}

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"