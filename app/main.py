"""FastAPI service exposing the quantized Falcon-RW 1B model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.inference import generate_text
from app.model_loader import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Falcon-RW 1B Quantized API",
    description="4-bit NF4 quantized Falcon-RW 1B text generation service.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(150, ge=1, le=500)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    seed: int = Field(42, ge=0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model, tokenizer = load_model()
    text = generate_text(
        model,
        tokenizer,
        req.prompt,
        seed=req.seed,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(prompt=req.prompt, generated_text=text)
