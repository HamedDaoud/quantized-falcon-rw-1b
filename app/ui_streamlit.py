import streamlit as st
from app.model_loader import load_model
from app.inference import generate_text

st.title("🧠 Falcon-RW 1B – Quantized Text Generator")

# Load once and cache
model, tokenizer = load_model()

prompt = st.text_area("Enter your prompt", "Explain quantum computing")

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_text(model, tokenizer, prompt)
        st.success(output)