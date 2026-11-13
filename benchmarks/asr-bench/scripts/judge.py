#!/usr/bin/env python3
"""Score all /results/*.json against manifest ground truth -> summary.json + markdown."""
import json, re, sys, glob
from pathlib import Path
import jiwer

MANIFEST = {m["id"]: m for m in json.loads(Path("/data/manifest.json").read_text())}

def norm(s):
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"'", "", s)
    return re.sub(r"\s+", " ", s).strip()

summary = []
files = sorted(glob.glob("/results/moonshine_*.json") + glob.glob("/results/parakeet_*.json"))
for f in files:
    r = json.loads(Path(f).read_text())
    tot_words, tot_errs = 0, 0
    cer_ref, cer_hyp = [], []
    for clip in r["clips"]:
        gt, hyp = MANIFEST[clip["id"]]["text"], clip["text"]
        gt_n, hyp_n = norm(gt), norm(hyp)
        m = jiwer.process_words(gt_n, hyp_n)
        tot_words += len(gt_n.split())
        tot_errs += m.substitutions + m.deletions + m.insertions
        cer_ref.append(gt_n); cer_hyp.append(hyp_n)
    meas = jiwer.process_characters(" ".join(cer_ref), " ".join(cer_hyp))
    audio = sum(c["audio_s"] for c in r["clips"])
    wall = sum(c["wall_s"] for c in r["clips"])
    walls = sorted(c["wall_s"] for c in r["clips"])
    empty = sum(1 for c in r["clips"] if not norm(c["text"]))
    summary.append({
        "system": r["system"], "mode": r.get("mode"),
        "wer_pct": round(100 * tot_errs / max(tot_words, 1), 2),
        "cer_pct": round(100 * meas.cer, 2),
        "empty_clips": empty,
        "rtfx": round(audio / wall, 2),
        "wall_total_s": round(wall, 1),
        "median_clip_s": walls[len(walls)//2],
        "p95_clip_s": walls[min(len(walls)-1, int(len(walls)*0.95))],
        "load_s": r["load_s"], "peak_rss_mb": r["peak_rss_mb"], "cpu_s": r["cpu_s"],
        "first_token_latency_s": r.get("first_token_latency_s"),
    })

Path("/results/summary.json").write_text(json.dumps(summary, indent=1))
cols = ["system", "wer_pct", "cer_pct", "rtfx", "median_clip_s", "peak_rss_mb", "load_s", "empty_clips"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for s in summary:
    print("| " + " | ".join(str(s.get(c)) for c in cols) + " |")
