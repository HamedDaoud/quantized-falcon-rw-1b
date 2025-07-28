import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from app.config import MODEL_ID, QUANT_CONFIG

import streamlit as st

@st.cache_resource(show_spinner="Loading Falcon-RW 1B model...")
def load_model():
    quant_config = BitsAndBytesConfig(**QUANT_CONFIG)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer
