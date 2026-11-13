#!/usr/bin/env python3
"""Moonshine Voice bench runner. usage: bench_moonshine.py <arch> <offline|stream>"""
import json, os, sys, time
from pathlib import Path
from moonshine_voice import (Transcriber, TranscriptEventListener, ModelArch,
                             get_model_for_language, load_wav_file, string_to_model_arch)

ARCH, MODE = sys.argv[1], sys.argv[2]
MANIFEST = json.loads(Path("/data/manifest.json").read_text())
OUT = Path(f"/results/moonshine_{ARCH}_{MODE}.json")

def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024
    return None

arch_enum = ModelArch[ARCH]
model_path, resolved_arch = get_model_for_language("en", arch_enum, cache_root=Path("/models/ms"))

t_load0 = time.perf_counter()
tr = Transcriber(model_path=model_path, model_arch=resolved_arch)
load_s = time.perf_counter() - t_load0

cpu0 = os.times()
results = []

if MODE == "offline":
    for c in MANIFEST:
        audio, sr = load_wav_file(c["wav"])
        t0 = time.perf_counter()
        tr_text = tr.transcribe_without_streaming(list(audio), sr)
        text = tr_text.text if hasattr(tr_text, "text") else str(tr_text)
        results.append({"id": c["id"], "audio_s": c["duration_s"],
                        "wall_s": round(time.perf_counter() - t0, 4), "text": str(text)})
else:
    first_latency = {}
    class Collect(TranscriptEventListener):
        def __init__(self, cid):
            self.cid, self.parts, self.t_first, self.t0 = cid, [], None, time.perf_counter()
        def on_line_text_changed(self, event):
            if self.t_first is None:
                self.t_first = time.perf_counter() - self.t0
        def on_line_completed(self, event):
            self.parts.append(event.line.text)
    for c in MANIFEST:
        audio, sr = load_wav_file(c["wav"])
        stream = tr.create_stream(update_interval=0.5)
        col = Collect(c["id"])
        stream.add_listener(col)
        stream.start()
        t0 = time.perf_counter()
        chunk = int(0.1 * sr)
        for i in range(0, len(audio), chunk):
            stream.add_audio(list(audio[i:i + chunk]), sr)
        stream.stop()
        wall = time.perf_counter() - t0
        stream.close()
        first_latency[c["id"]] = round(col.t_first, 4) if col.t_first else None
        results.append({"id": c["id"], "audio_s": c["duration_s"], "wall_s": round(wall, 4),
                        "text": " ".join(col.parts)})
    if MODE == "stream":
        # re-run clip 0 to capture first-token latency cleanly
        pass

cpu = os.times()
payload = {
    "system": f"moonshine-voice {ARCH} {MODE}",
    "model": f"Moonshine Voice ({ARCH}, {MODE})",
    "arch": ARCH, "mode": MODE,
    "load_s": round(load_s, 3),
    "peak_rss_mb": round(rss_mb(), 1) if rss_mb() else None,
    "cpu_s": round((cpu.user - cpu0.user) + (cpu.system - cpu0.system), 1),
    "clips": results,
}
if MODE == "stream":
    fl = [r for r in (first_latency.values()) if r]
    payload["first_token_latency_s"] = {"n": len(fl), "median": sorted(fl)[len(fl)//2] if fl else None,
                                        "p95": fl[min(len(fl)-1, int(len(fl)*0.95))] if fl else None}
OUT.write_text(json.dumps(payload, indent=1))
print(f"done {ARCH}/{MODE}: {len(results)} clips, peak RSS {payload['peak_rss_mb']} MB")
