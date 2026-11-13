#!/usr/bin/env python3
"""Slug-protocol per-request probe — GPU variant.

Same 0.5 s / 11 s audio-pair method as bench_probe.py; only the execution
provider differs (ASR_PROVIDERS, default CUDA+CPU).
usage: bench_probe_gpu.py parakeet [moonshine_arch]
"""
import json, os, sys, time
from pathlib import Path

ENGINE = sys.argv[1]
ARCH = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(os.environ.get("BENCH_OUT", f"/results/probe_{ENGINE}_{ARCH or 'gpu'}.json"))
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

def cpu_sec():
    t = os.times()
    return t.user + t.system

if ENGINE == "parakeet":
    import onnx_asr
    t0 = time.perf_counter()
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3",
                                os.environ.get("ASR_MODEL_PATH", "/models/parakeet"),
                                quantization="int8", providers=PROVIDERS)
    load_s = time.perf_counter() - t0
    def transcribe(path):
        return model.recognize(path)
else:
    raise SystemExit("moonshine has no GPU path (bundled CPU onnxruntime); use bench_probe.py")

def probe(path, n=5):
    times = []
    c0 = cpu_sec()
    transcribe(path)                      # warmup, uncounted
    w0 = cpu_sec()
    for _ in range(n):
        t = time.perf_counter()
        transcribe(path)
        times.append(time.perf_counter() - t)
    cpu_total = cpu_sec() - w0
    s = sorted(times)
    return {"median_s": round(s[n // 2], 4), "min_s": round(s[0], 4), "max_s": round(s[-1], 4),
            "cpu_s_per_req": round(cpu_total / n, 4)}

PROBE_DIR = os.environ.get("ASR_PROBE_DIR", "/data/probe")
data = {"engine": f"{ENGINE} {ARCH or 'tdt-0.6b-v3-int8'} GPU ({','.join(PROVIDERS)})",
        "load_s": round(load_s, 3), "peak_rss_mb": round(rss_mb(), 1),
        "audio_11s": probe(f"{PROBE_DIR}/speech11.wav"),
        "audio_05s": probe(f"{PROBE_DIR}/short05.wav")}
data["fixed_overhead_est_s"] = round(data["audio_05s"]["median_s"], 4)
data["marginal_per_audio_s"] = round((data["audio_11s"]["median_s"] - data["audio_05s"]["median_s"]) / 10.63, 5)
OUT.write_text(json.dumps(data, indent=1))
print(json.dumps(data))
