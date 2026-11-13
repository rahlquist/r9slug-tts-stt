#!/usr/bin/env python3
"""Parakeet TDT bench runner — GPU (CUDA/ROCm) variant via onnx-asr.

Identical protocol to bench_parakeet.py; only the execution provider differs.
Provider order is configurable through ASR_PROVIDERS (comma separated).
"""
import json, os, time
from pathlib import Path

import onnx_asr

MANIFEST = json.loads(Path(os.environ.get("ASR_MANIFEST", "/data/manifest.json")).read_text())
OUT = Path(os.environ.get("BENCH_OUT", "/results/parakeet_tdt-0.6b-v3_int8-gpu.json"))
PROVIDERS = [p for p in os.environ.get("ASR_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",") if p]

# Guard: never let a silent CPU fallback be reported as a GPU result.
import onnxruntime as _ort
_avail = _ort.get_available_providers()
_missing = [p for p in PROVIDERS if p != "CPUExecutionProvider" and p not in _avail]
if _missing:
    raise SystemExit(f"requested providers unavailable: {_missing}; available={_avail}")

def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024
    return None

t0 = time.perf_counter()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",
                            os.environ.get("ASR_MODEL_PATH", "/models/parakeet"),
                            quantization="int8", providers=PROVIDERS)
load_s = time.perf_counter() - t0

cpu0 = os.times()
results = []
for c in MANIFEST:
    t = time.perf_counter()
    text = model.recognize(c["wav"])
    results.append({"id": c["id"], "audio_s": c["duration_s"],
                    "wall_s": round(time.perf_counter() - t, 4), "text": text})
cpu = os.times()

payload = {
    "system": f"onnx-asr parakeet-tdt-0.6b-v3 int8 GPU ({','.join(PROVIDERS)})",
    "model": "NVIDIA Parakeet TDT 0.6B v3 (int8 ONNX)",
    "arch": "tdt-0.6b-v3-int8", "mode": "offline",
    "providers": PROVIDERS,
    "load_s": round(load_s, 3),
    "peak_rss_mb": round(rss_mb(), 1),
    "cpu_s": round((cpu.user - cpu0.user) + (cpu.system - cpu0.system), 1),
    "clips": results,
}
OUT.write_text(json.dumps(payload, indent=1))
print(f"done parakeet GPU: {len(results)} clips, peak RSS {payload['peak_rss_mb']} MB, providers={PROVIDERS}")
