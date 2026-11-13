#!/usr/bin/env python3
"""Slug-protocol per-request probe: steady-state wall latency + cpu-s/request at
two audio lengths (0.5 s and 11 s), matching the 2026-08 slug NPU-vs-iGPU method.
usage: bench_probe.py <engine> [moonshine_arch]"""
import json, os, sys, time
from pathlib import Path

ENGINE = sys.argv[1]
ARCH = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(f"/results/probe_{ENGINE}_{ARCH or 'x'}.json")

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
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "/models/parakeet", quantization="int8")
    load_s = time.perf_counter() - t0
    def transcribe(path):
        return model.recognize(path)
else:
    from moonshine_voice import Transcriber, get_model_for_language, load_wav_file, ModelArch
    mpath, march = get_model_for_language("en", ModelArch[ARCH], cache_root=Path("/models/ms"))
    t0 = time.perf_counter()
    tr = Transcriber(model_path=mpath, model_arch=march)
    load_s = time.perf_counter() - t0
    def transcribe(path):
        audio, sr = load_wav_file(path)
        t = tr.transcribe_without_streaming(list(audio), sr)
        return t.text if hasattr(t, "text") else str(t)

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

data = {"engine": f"{ENGINE} {ARCH or 'tdt-0.6b-v3-int8'}", "load_s": round(load_s, 3),
        "peak_rss_mb": round(rss_mb(), 1),
        "audio_11s": probe("/data/probe/speech11.wav"),
        "audio_05s": probe("/data/probe/short05.wav")}
data["fixed_overhead_est_s"] = round(data["audio_05s"]["median_s"], 4)
data["marginal_per_audio_s"] = round((data["audio_11s"]["median_s"] - data["audio_05s"]["median_s"]) / 10.63, 5)
OUT.write_text(json.dumps(data, indent=1))
print(json.dumps(data))
