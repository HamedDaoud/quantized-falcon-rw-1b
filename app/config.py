"""Static configuration for the Falcon-RW 1B quantized text generation app."""

MODEL_ID = "tiiuae/falcon-rw-1b"

NF4_QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

INFERENCE_DEFAULTS = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}
