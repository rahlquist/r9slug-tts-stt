#!/usr/bin/env python3
"""Fetch labeled librispeech clips, convert to 16k mono wav, write manifest."""
import json, subprocess, sys, urllib.request
from pathlib import Path

BASE = Path("/home/rahlquist/.hermes/cache/scratch/asr-bench")
CLIPS = BASE / "data" / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)

API = ("https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing%2Flibrispeech_asr_dummy"
       "&config=clean&split=validation&offset={off}&length={ln}")

rows = []
off = 0
while True:
    with urllib.request.urlopen(API.format(off=off, ln=50), timeout=60) as r:
        d = json.load(r)
    batch = d.get("rows", [])
    if not batch:
        break
    rows.extend(batch)
    off += len(batch)
    if off >= d.get("num_rows_total", off):
        break

print(f"got {len(rows)} rows")
manifest = []
for i, row in enumerate(rows):
    row = row["row"]
    src = row["audio"]["src"] if isinstance(row["audio"], dict) else row["audio"][0]["src"]
    clip_id = f"{i:03d}_{row['id']}"
    flac = CLIPS / f"{clip_id}.flac"
    wav = CLIPS / f"{clip_id}.wav"
    if not wav.exists():
        urllib.request.urlretrieve(src, flac)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(flac),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        flac.unlink()
    probe = json.loads(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                       "-of", "json", str(wav)], capture_output=True, text=True).stdout)
    dur = float(probe["format"]["duration"])
    manifest.append({"id": clip_id, "wav": str(wav), "text": row["text"], "duration_s": round(dur, 3)})
    print(f"  {clip_id} {dur:.1f}s")

(BASE / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))
total = sum(m["duration_s"] for m in manifest)
print(f"TOTAL: {len(manifest)} clips, {total:.1f}s audio")
