"""Gradio interface for the quantized Falcon-RW 1B model.

Run with: python -m app.ui_gradio
"""

from __future__ import annotations

import gradio as gr

from app.inference import generate_text
from app.model_loader import load_model

print("Loading Falcon-RW 1B...")
MODEL, TOKENIZER = load_model()
print("Model loaded.")


def run_generation(prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int) -> str:
    if not prompt or not prompt.strip():
        return "Please enter a prompt."
    return generate_text(
        MODEL,
        TOKENIZER,
        prompt,
        seed=int(seed),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )


EXAMPLES = [
    ["Once upon a time, there was a robot named Atlas who lived in a small workshop on the edge of the city.", 150, 0.9, 0.95, 7],
    ["In machine learning, overfitting refers to the phenomenon where", 120, 0.6, 0.9, 42],
    ["def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ", 100, 0.4, 0.9, 0],
    ["Dear Hiring Manager,\n\nI am writing to apply for the Machine Learning Engineer position at", 180, 0.7, 0.9, 12],
]

with gr.Blocks(theme=gr.themes.Soft(), title="Falcon-RW 1B - NF4 Quantized") as demo:
    gr.Markdown(
        """
        # Falcon-RW 1B - 4-bit NF4 Quantized

        Text generation with a 4-bit NormalFloat (NF4) quantized Falcon-RW 1B model.
        Combines NF4 quantization with double quantization via `bitsandbytes` to reduce
        the memory footprint by approximately 75% versus float32 while preserving
        generation quality.

        > **About the model.** Falcon-RW-1B is a 1.3B-parameter *base* language model — it
        > completes text rather than following instructions. For best results, give it the
        > *start* of a passage and let it continue (try the Examples below). This project
        > demonstrates the quantization and deployment stack; output quality reflects the
        > base model's capabilities, not a quantization artifact.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt (give it the start of a passage to continue)",
                placeholder="Once upon a time, there was a robot named Atlas who...",
                lines=5,
            )
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")
            output = gr.Textbox(label="Generated text", lines=10)

        with gr.Column(scale=1):
            gr.Markdown("### Generation parameters")
            max_new_tokens = gr.Slider(16, 500, value=150, step=16, label="Max new tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="Top-p")
            seed = gr.Number(value=42, label="Seed", precision=0)

    gr.Examples(examples=EXAMPLES, inputs=[prompt, max_new_tokens, temperature, top_p, seed])

    generate_btn.click(
        run_generation,
        inputs=[prompt, max_new_tokens, temperature, top_p, seed],
        outputs=output,
    )
    clear_btn.click(lambda: ("", ""), outputs=[prompt, output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
