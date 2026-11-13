#!/usr/bin/env python3
"""Aggregate the ASR bench matrix into a comparison chart (slug-benchmark format).

Reads results/<machine>/<device>/summary.json + probe_*.json + hostinfo and
writes benchmarks/benchmark-comparison-<date>.md.
"""
import json
import sys
from datetime import date
from pathlib import Path

KIT = Path(__file__).resolve().parent
RESULTS = KIT / "results"
OUT = KIT.parent / "benchmark-comparison.md"

MACHINE_ORDER = ["r9slug", "wimpy", "hermesvm01"]
DEVICE_ORDER = ["cpu", "cuda", "rocm"]

MODEL_ORDER = [
    "onnx-asr parakeet-tdt-0.6b-v3 int8 offline",
    "onnx-asr parakeet-tdt-0.6b-v3 int8 GPU (CUDAExecutionProvider,CPUExecutionProvider)",
    "onnx-asr parakeet-tdt-0.6b-v3 int8 GPU (ROCMExecutionProvider,CPUExecutionProvider)",
    "moonshine-voice BASE offline",
    "moonshine-voice TINY offline",
    "moonshine-voice TINY_STREAMING stream",
    "moonshine-voice SMALL_STREAMING stream",
    "moonshine-voice MEDIUM_STREAMING stream",
]

MODEL_SHORT = {
    "onnx-asr parakeet-tdt-0.6b-v3 int8 offline": "Parakeet TDT 0.6B v3 int8",
    "onnx-asr parakeet-tdt-0.6b-v3 int8 GPU (CUDAExecutionProvider,CPUExecutionProvider)": "Parakeet TDT 0.6B v3 int8",
    "onnx-asr parakeet-tdt-0.6b-v3 int8 GPU (ROCMExecutionProvider,CPUExecutionProvider)": "Parakeet TDT 0.6B v3 int8",
    "moonshine-voice BASE offline": "Moonshine Base (offline)",
    "moonshine-voice TINY offline": "Moonshine Tiny (offline)",
    "moonshine-voice TINY_STREAMING stream": "Moonshine Tiny Streaming",
    "moonshine-voice SMALL_STREAMING stream": "Moonshine Small Streaming",
    "moonshine-voice MEDIUM_STREAMING stream": "Moonshine Medium Streaming",
}


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def collect():
    """-> {(machine, device): {system: entry}} and {(machine, device): [probe...]}"""
    summaries, probes = {}, {}
    for m in MACHINE_ORDER:
        for d in DEVICE_ORDER:
            s = load_json(RESULTS / m / d / "summary.json")
            if s:
                summaries[(m, d)] = {e["system"]: e for e in s}
            pl = []
            for p in sorted((RESULTS / m / d).glob("probe_*.json")):
                j = load_json(p)
                if j:
                    pl.append(j)
            if pl:
                probes[(m, d)] = pl
    return summaries, probes


def hostinfo(m):
    txt = load_json  # noqa
    p = RESULTS / "hostinfo" / f"{m}.txt"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k.strip()] = v.strip()
    return out


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def main():
    summaries, probes = collect()
    if not summaries:
        sys.exit("no results found")

    lines = []
    a = lines.append
    a(f"# ASR bench comparison — r9slug / wimpy / hermesvm01, CPU + dGPU ({date.today()})")
    a("")
    a("Same kit, same 73-clip LibriSpeech corpus (481 s), same slug-protocol probe ")
    a("(0.5 s / 11 s audio pair, median of 5). Containers: `asr-parakeet` / `asr-moonshine`,")
    a("`OMP_NUM_THREADS=6`, models baked into the images.")
    a("")

    # ---- host stack ----
    a("## Host stack")
    a("")
    a("| Host | CPU | Cores | RAM | Accelerator | Runtime |")
    a("|---|---|---:|---:|---|---|")
    for m in MACHINE_ORDER:
        h = hostinfo(m)
        if not h:
            continue
        acc = []
        if h.get("nvidia_gpu"):
            acc.append(f"{h['nvidia_gpu']} (driver {h.get('nvidia_driver','?')})")
        if h.get("rocm_gpu"):
            acc.append(f"{h['rocm_gpu']} (ROCm runtime {h.get('rocm_runtime','?')})")
        a(f"| {m} | {h.get('cpu','—')} | {h.get('cores','—')} | {h.get('ram_gb','—')} GB | "
          f"{'; '.join(acc) if acc else 'CPU only'} | kernel {h.get('kernel','—')} |")
    a("")

    # ---- WER / RTFx per model ----
    a("## Word error rate and throughput by model")
    a("")
    a("WER over reference words (jiwer, normalized). RTFx = audio seconds / wall seconds;")
    a("higher is faster. `dgpu` = the host's discrete GPU (CUDA on r9slug, ROCm on wimpy).")
    a("`—` = not run on that host.")
    a("")
    hosts = [m for m in MACHINE_ORDER if any((m, d) in summaries for d in DEVICE_ORDER)]

    # (model, kind) -> {host: entry}
    table = {}
    for (m, d), systems in summaries.items():
        for system, e in systems.items():
            model = MODEL_SHORT.get(system, system)
            kind = "dgpu" if "GPU (" in system else "cpu"
            table.setdefault((model, kind), {})[m] = e

    header = ["Model", "Device", "Metric"] + hosts
    a("| " + " | ".join(header) + " |")
    a("|" + "---|" * len(header))
    seen = set()
    for system in MODEL_ORDER:
        model = MODEL_SHORT[system]
        kind = "dgpu" if "GPU (" in system else "cpu"
        if (model, kind) in seen:
            continue
        seen.add((model, kind))
        for metric, nd, label in (("wer_pct", 2, "WER %"), ("rtfx", 2, "RTFx")):
            row = [model, kind, label]
            for m in hosts:
                e = table.get((model, kind), {}).get(m)
                row.append(fmt(e.get(metric), nd) if e else "—")
            a("| " + " | ".join(row) + " |")
    a("")

    # ---- probe ----
    a("## Slug-protocol probe (per-request latency)")
    a("")
    a("Median of 5 requests, warmup excluded. `fixed` is the 0.5 s request median;")
    a("`marginal` is the per-audio-second slope between 0.5 s and 11 s.")
    a("")
    hdr = ["Engine", "Host / device", "0.5 s req (s)", "11 s req (s)", "fixed (s)", "marginal (s/audio-s)"]
    a("| " + " | ".join(hdr) + " |")
    a("|" + "---|" * len(hdr))
    for m in MACHINE_ORDER:
        for d in DEVICE_ORDER:
            for p in probes.get((m, d), []):
                a(f"| {p['engine']} | {m} / {d} | {fmt(p['audio_05s']['median_s'],4)} | "
                  f"{fmt(p['audio_11s']['median_s'],4)} | {fmt(p['fixed_overhead_est_s'],4)} | "
                  f"{fmt(p['marginal_per_audio_s'],5)} |")
    a("")

    # ---- reference ----
    a("## Reference — slug (2026-08-06)")
    a("")
    a("| Engine | Host | 11 s req (s) |")
    a("|---|---|---:|")
    a("| whisper large-v3-turbo (iGPU, Vulkan) | slug | 1.74 |")
    a("| whisper large-v3-turbo (NPU, FLM) | slug | 2.70 |")
    a("")
    a("Slug figures are end-to-end HTTP request latencies (~2.1 s fixed server overhead")
    a("included); the numbers above are in-process. Compare request latencies, not overheads.")
    a("")

    # ---- notes ----
    a("## Notes")
    a("")
    a("- Moonshine has no GPU path: `moonshine-voice` bundles a CPU-only onnxruntime")
    a("  (22 MB, no provider options), so it is CPU-only on every host. Only Parakeet")
    a("  (`onnx-asr`) was run on the dGPUs.")
    a("- Parakeet on the dGPU is **not** faster than 6 CPU threads on either host. Both")
    a("  GPU runs are host/device-copy bound (onnxruntime reports hundreds of Memcpy")
    a("  nodes for the CUDA EP), and the model is small enough that the CPU wins.")
    a("- r9slug CUDA and wimpy ROCm results were verified to actually execute on the")
    a("  GPU (CUDA EP session providers; AMD `gpu_busy_percent` 27–60% during the run).")
    a("- hermesvm01 is a VM on wimpy and has no dGPU, so it is CPU-only by design.")
    a("- WER differences of ±0.3 pp between hosts are run-to-run noise on a 73-clip")
    a("  corpus; the model and corpus are identical everywhere.")
    a("- **CPU numbers are load-sensitive and hermesvm01 shares a physical host with")
    a("  wimpy.** The first hermesvm01 run was made while the host was still busy and")
    a("  came out ~1.7× slow (parakeet RTFx 26.8 vs 46.7); it was discarded and re-run")
    a("  on an idle host. Run the CPU matrix only when the host is idle.")
    a("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
