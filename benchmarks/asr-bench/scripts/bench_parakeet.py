#!/usr/bin/env python3
"""Parakeet TDT bench runner (offline batch via onnx-asr)."""
import json, os, time
from pathlib import Path
import onnx_asr

MANIFEST = json.loads(Path("/data/manifest.json").read_text())
OUT = Path("/results/parakeet_tdt-0.6b-v3_int8-offline.json")

def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024
    return None

t0 = time.perf_counter()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "/models/parakeet", quantization="int8")
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
    "system": "onnx-asr parakeet-tdt-0.6b-v3 int8 offline",
    "model": "NVIDIA Parakeet TDT 0.6B v3 (int8 ONNX)",
    "arch": "tdt-0.6b-v3-int8", "mode": "offline",
    "load_s": round(load_s, 3),
    "peak_rss_mb": round(rss_mb(), 1),
    "cpu_s": round((cpu.user - cpu0.user) + (cpu.system - cpu0.system), 1),
    "clips": results,
}
OUT.write_text(json.dumps(payload, indent=1))
print(f"done parakeet: {len(results)} clips, peak RSS {payload['peak_rss_mb']} MB")
