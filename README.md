# Falcon-RW 1B — 4-bit NF4 Quantized Text Generation

[![Open in HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Open%20in-HF%20Spaces-blue)](https://huggingface.co/spaces/hameddaoud/quantized-falcon-rw-1b)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Live demo:** [hameddaoud/quantized-falcon-rw-1b on Hugging Face Spaces](https://huggingface.co/spaces/hameddaoud/quantized-falcon-rw-1b) — slow on free CPU tier (~1 tok/sec); duplicate to GPU hardware for NF4 quantization to activate.

_Live demo recording coming soon — fresh screen capture of the Gradio UI to replace the prior Streamlit screenshot._

A text generation application that loads [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b) under **4-bit NormalFloat (NF4) quantization** with **double quantization** via `bitsandbytes`, exposed through both a Gradio UI and a FastAPI service. The 4-bit weights are paired with a `bfloat16` compute dtype, reducing the model's memory footprint by approximately **75% versus float32** while preserving generation quality.

> **About the model.** Falcon-RW-1B is a 1.3B-parameter *base* language model — it completes text rather than following instructions, and at this size it's well below the capability of modern instruction-tuned chat models. **This project's contribution is the quantization technique, deployment surface, and device-fallback engineering — not state-of-the-art generation quality.** Prompts that work best are the start of a passage rather than questions or commands (see Examples in the Gradio UI).

## What this project demonstrates

- **Quantization-aware deployment** — loading a transformer LLM in NF4 with double quantization and `bfloat16` compute, with a clean fallback path for non-CUDA hardware (Apple Silicon MPS / CPU).
- **Two-surface serving** — the same `model_loader` + `inference` core powers both a REST API (`/generate`) and an interactive Gradio web app, with no duplicated model-loading logic.
- **Container-ready** — a single `Dockerfile` builds an image that can run either surface based on an environment variable.

## Quantization at a glance

| Method | Bits / param | Compute dtype | Memory vs FP32 |
|---|---|---|---|
| Float32 baseline | 32 | float32 | 1.00× |
| BFloat16 | 16 | bfloat16 | 0.50× |
| **NF4 + double quantization** *(this project)* | ~4.13 | bfloat16 | **~0.13×** |

NF4 (4-bit NormalFloat) maps weights to 16 quantization levels chosen from the quantiles of a standard normal distribution, which is information-theoretically optimal for normally distributed weights. Double quantization additionally compresses the quantization constants, saving roughly **0.37 bits per parameter** on average. As reported in the QLoRA paper, NF4 with double quantization recovers full-precision performance on standard academic benchmarks while operating with a fraction of the memory.

> Published evidence: "4-bit QLoRA with NF4 data type matches 16-bit full finetuning and 16-bit LoRA finetuning performance on academic benchmarks. NF4 is more effective than FP4 and double quantization does not degrade performance." — Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314). See also the [Hugging Face bitsandbytes 4-bit blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

## Architecture

```
            ┌──────────────────────────┐
  prompt -->│  Gradio UI (port 7861)   │──┐
            └──────────────────────────┘  │
                                          ▼
                            ┌─────────────────────────┐
                            │ app/inference.py        │
                            │   generate_text(...)    │
                            └────────────┬────────────┘
                                         │
                            ┌────────────▼────────────┐
                            │ app/model_loader.py     │
                            │   load_model() (cached) │
                            │   ┌─ CUDA  -> NF4       │
                            │   ├─ MPS   -> float16   │
                            │   └─ CPU   -> float32   │
                            └─────────────────────────┘
                                         ▲
            ┌──────────────────────────┐ │
  HTTP --> │  FastAPI (port 8000)      │─┘
           │  POST /generate           │
           └──────────────────────────┘
```

## Quick start

### Local — Gradio UI

```bash
pip install -r requirements.txt
python -m app.ui_gradio
# open http://localhost:7861
```

### Local — FastAPI service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# health check
curl http://localhost:8000/health
# generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantization in one paragraph.", "max_new_tokens": 150}'
```

### Docker

```bash
# Gradio (default) — mount the Hugging Face cache to avoid re-downloading the 2.5 GB model
docker build -t falcon-nf4 .
docker run --rm -p 7861:7861 -v ~/.cache/huggingface:/cache/huggingface falcon-nf4

# FastAPI mode
docker run --rm -e APP_MODE=api -p 8000:8000 -v ~/.cache/huggingface:/cache/huggingface falcon-nf4
```

For CUDA acceleration inside Docker, add `--gpus all` and switch to a CUDA-enabled base image
(e.g. `nvidia/cuda:12.1.1-runtime-ubuntu22.04`).

> **Memory note for CPU-only containers (e.g. Docker Desktop on Mac).** Falcon-RW-1B's load
> path spikes well above the resident model size; we recommend giving Docker at least
> **6 GB of RAM** (`Docker Desktop → Settings → Resources → Memory`). The container picks
> `float16` on CPU, which keeps the model under ~2.6 GB but the loader and runtime headroom
> push peak memory closer to 4 GB. Generation on CPU is slow (~1–2 tokens/sec); this
> container is best suited as a *deployment-artifact demonstration*. For interactive use,
> run on a CUDA-enabled host so NF4 quantization activates.

## Hardware notes

| Environment | NF4 quantization | Fallback path |
|---|---|---|
| Linux + NVIDIA GPU (CUDA) | ✅ Active | — |
| Apple Silicon (MPS) | ❌ `bitsandbytes` requires CUDA | float16 on MPS |
| CPU-only (incl. Docker on Mac) | ❌ | float16 on CPU |

The `load_model()` helper detects CUDA automatically. Pass `use_quantization=True` to force NF4 (errors if CUDA unavailable) or `use_quantization=False` to load full-precision weights explicitly.

## Project structure

```
quantized-falcon-rw-1b/
├── app/
│   ├── __init__.py
│   ├── config.py          # Model ID + NF4 + generation defaults
│   ├── model_loader.py    # Cached loader with CUDA/MPS/CPU paths
│   ├── inference.py       # Pure generate_text() entry point
│   ├── main.py            # FastAPI service (/health, /generate)
│   └── ui_gradio.py       # Gradio Blocks interface
├── notebooks/
│   └── quantization.ipynb # Exploratory work before refactoring
├── assets/
│   └── demo.png
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech stack

| Layer | Tools |
|---|---|
| Model | `tiiuae/falcon-rw-1b` (Hugging Face) |
| Quantization | `bitsandbytes` (NF4 + double quantization) |
| Compute | PyTorch (bfloat16 / float16 / float32 depending on device) |
| Serving | FastAPI + Uvicorn |
| UI | Gradio (Blocks, Soft theme) |
| Packaging | Docker |

## References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). arXiv:2305.14314.
- Hugging Face. [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

## License

MIT — see [LICENSE](LICENSE).
