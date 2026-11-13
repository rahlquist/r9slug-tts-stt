"""OpenAI-compatible Parakeet STT service (resident, pre-warmed).

Loads the model once at startup, warms it, then serves /v1/audio/transcriptions.
Used for the Stage-3 style concurrency test against the Kokoro TTS service.
"""
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import onnx_asr
from fastapi import FastAPI, File, Form, UploadFile

MODEL = os.environ.get("ASR_MODEL", "nemo-parakeet-tdt-0.6b-v3")
MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "/models/parakeet")
PROVIDERS = [p for p in os.environ.get("ASR_PROVIDERS", "CPUExecutionProvider").split(",") if p]

_state = {"ready": False, "model": None, "load_ms": None, "warm_ms": None, "requests": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    _state["model"] = onnx_asr.load_model(MODEL, MODEL_PATH, quantization="int8", providers=PROVIDERS)
    _state["load_ms"] = (time.perf_counter() - t0) * 1000

    # Warm-up: 2 s of speech-like audio (a swept tone) so kernel selection is done.
    sr = 16000
    n = 2 * sr
    t = np.linspace(0, 2, n, endpoint=False)
    warm = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    t1 = time.perf_counter()
    _state["model"].recognize(warm, sample_rate=sr)
    _state["warm_ms"] = (time.perf_counter() - t1) * 1000
    _state["ready"] = True
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if _state["ready"] else "loading",
        "model": MODEL,
        "providers": PROVIDERS,
        "load_ms": round(_state["load_ms"], 1) if _state["load_ms"] else None,
        "warm_ms": round(_state["warm_ms"], 1) if _state["warm_ms"] else None,
        "requests": _state["requests"],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = Form("whisper-1")):
    data = await file.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t0 = time.perf_counter()
    text = _state["model"].recognize(audio, sample_rate=16000)
    dt = time.perf_counter() - t0
    _state["requests"] += 1
    return {"text": text, "elapsed_ms": round(dt * 1000, 1)}
