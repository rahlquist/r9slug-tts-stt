# r9slug-tts-stt

LAN **voice service** on an NVIDIA RTX 5060 Ti (Blackwell, sm_120): text-to-speech from a
resident, pre-warmed **Kokoro-82M** service on the dGPU, and speech-to-text benchmarked on
CPU and dGPU across three hosts with the same protocol slug uses — so the numbers can be
put in one chart next to slug's.

* **TTS** → **RTX 5060 Ti** via PyTorch CUDA (`kokoro`, `Kokoro-82M` int8)
* **STT** → **CPU** (`onnx-asr` Parakeet TDT 0.6B v3 int8, `moonshine-voice`) and
  **dGPU** (Parakeet via onnxruntime CUDA/ROCm execution providers)

The host is the *audio edge* — ears and mouth. The reasoning happens elsewhere (an agent
on the LAN consumes the endpoints); no chat LLM is required on this box.

> Interactive version: `docs/architecture.html`

---

## Topology

```
LAN clients (192.168.8.0/24)
        │
        ├── POST :8093/v1/audio/speech ─────────► kokoro-tts ──────────────► dGPU (RTX 5060 Ti, CUDA)
        │                                          (LAN face, resident)
        │
        └── POST :8094/v1/audio/transcriptions ─► parakeet-stt ───────────► CPU (6 threads)
                                                   (LAN face, resident)
```

| Port | Bind | Service | Device | Purpose |
|---|---|---|---|---|
| 8093 | `0.0.0.0` | `kokoro-tts` | **dGPU (CUDA)** | LAN-facing TTS, OpenAI shape |
| 8094 | `0.0.0.0` | `parakeet-stt` | **CPU** | LAN-facing STT, OpenAI shape |
| 8081 | slug | `whisper-proxy` | NPU / iGPU | STT that Hermes currently uses (remote) |

Both r9slug services are resident systemd units that restart on boot and pre-warm before
reporting ready. Hermes on r9slug currently consumes **slug** for both STT (`:8081`) and
TTS (`:8092`); r9slug's own endpoints on `:8093` / `:8094` are available to LAN clients
and are what the numbers below describe.

## Models

| Role | Model | Quant | Runtime | Size |
|---|---|---|---|---|
| TTS talker | `hexgrad/Kokoro-82M` | int8 | PyTorch CUDA | 89 MB |
| STT | `nemo-parakeet-tdt-0.6b-v3` | int8 | onnxruntime (CPU / CUDA / ROCm EP) | 640 MB |
| STT | Moonshine `base` / `tiny` | — | moonshine-voice (CPU only) | — |
| STT | Moonshine `tiny` / `small` / `medium` streaming | — | moonshine-voice (CPU only) | — |

---

## Measured results

All STT figures are the same kit on all three hosts: 73 labelled LibriSpeech clips
(481 s of real speech), `OMP_NUM_THREADS=6`, models baked into the images, WER scored by
`jiwer`. Raw results: `benchmarks/asr-bench/results/<host>/<device>/`.

### Host stack

| Host | CPU | Cores | RAM | Accelerator | Kernel |
|---|---|---:|---:|---|---|
| r9slug | Ryzen 9 3900X | 24 | 62 GB | NVIDIA RTX 5060 Ti (driver 620.70.06) | 7.3.3-1-cachyos |
| wimpy | Ryzen 9 9900X | 24 | 60 GB | AMD Radeon AI PRO R9700 (ROCm 1.18) | 7.3.1-1-cachyos |
| hermesvm01 | Ryzen 9 9900X | 6 | 15 GB | CPU only (VM on wimpy) | 7.3.1-1-cachyos |

### STT — Parakeet TDT 0.6B v3 int8

| Host / device | WER % | RTFx | 11 s request | 0.5 s request |
|---|---:|---:|---:|---:|
| r9slug / CPU | 4.70 | 29.91 | 0.3505 s | 0.0688 s |
| r9slug / **dGPU (CUDA)** | 4.96 | 20.23 | 0.3356 s | 0.0581 s |
| wimpy / CPU | 4.78 | 42.98 | 0.1955 s | 0.0811 s |
| wimpy / **dGPU (ROCm)** | 4.96 | 43.58 | 0.2824 s | 0.0656 s |
| hermesvm01 / CPU | 4.78 | 46.70 | 0.1774 s | 0.0813 s |
| *slug / iGPU (whisper large-v3-turbo)* | *—* | *—* | *1.74 s* | *—* |
| *slug / NPU (whisper large-v3-turbo)* | *—* | *—* | *2.70 s* | *—* |

Parakeet is **5–15× faster than slug's whisper large-v3-turbo** on the identical 11 s
probe (0.177–0.351 s here against 1.74 s on slug's iGPU and 2.70 s on its NPU). Slug's
figures are end-to-end HTTP requests (~2.1 s fixed server overhead included); these are
in-process. Compare request latencies, not overheads.

### STT — Moonshine (CPU only)

| Model | WER % | RTFx (r9slug / wimpy / hermesvm01) |
|---|---:|---|
| Medium Streaming | 5.4–5.6 | 3.05 / 3.71 / 3.61 |
| Small Streaming | 7.2–7.3 | 4.42 / 4.87 / 5.05 |
| Tiny Streaming | 11.8 | 8.12 / 9.89 / 7.30 |
| Base (offline) | 31.4–31.7 | 22.45 / 40.35 / 40.16 |
| Tiny (offline) | 36.1–36.4 | 30.34 / 56.91 / 57.83 |

Moonshine's streaming builds are the competitive ones (5.4–11.8 % WER); the legacy
offline Base/Tiny builds are 31–36 % and not usable. Moonshine's differentiator is
first-token latency while you are still talking (median 0.03–0.10 s), which batch
benchmarks do not show — Parakeet cannot stream through this stack.

### TTS — Kokoro-82M on the RTX 5060 Ti

Resident, pre-warmed service; model load and warm-up are excluded from the timed region.

| Metric | Value |
|---|---|
| P1 (2.875 s audio) warm generation | **47.31 ms** (60.77× RTFx) |
| P2 (6.400 s audio) warm generation | **68.69 ms** (93.17× RTFx) |
| First HTTP request after ready | 204.5 ms |
| Subsequent requests | 63.4 / 48.2 ms |
| Peak GPU allocation | ~630–704 MiB |
| First generation in a fresh process | 15.6–16.4 s (why the service is resident) |

### The dGPU result, honestly

**Parakeet on the dGPU is not faster than 6 CPU threads on either host.** r9slug's CUDA
path is 20.23 RTFx against 29.91 on CPU; wimpy's ROCm path (43.58) is within noise of its
CPU path (42.98). Both GPU runs were verified to actually execute on the device, not fall back:
`nvidia-smi` on r9slug shows 1.4–2.8 GB allocated with 1–16 % utilization during the CUDA
run, and wimpy's Radeon reads 27–60 % `gpu_busy_percent` during the ROCm run. The low
utilization is itself the finding — the model is too small to keep the device busy, so
onnxruntime's host↔device transfers (the CUDA EP alone inserts 742 Memcpy nodes) cost more
than the CPU inference they save.

The dGPU is not wasted, though: it is the only path that serves **TTS** at 60–93× realtime
with 47 ms latency, and it leaves the CPU free for the STT workloads above.

### Simultaneous TTS + STT (Stage-3 concurrency)

Both services resident and pre-warmed, TTS on the dGPU. Same method as slug's Stage 3:
idle baselines, one strictly-simultaneous pair, then 5-way floods in each direction.
The STT clip is 5.855 s of LibriSpeech speech; TTS text is a fixed 6-word sentence.

| Scenario | STT idle | STT with TTS | TTS idle | TTS with STT |
|---|---:|---:|---:|---:|
| Parakeet STT on **CPU** | 0.212 s | 0.201 s (**−5.2 %**) | 0.054 s | 0.073 s (+35.1 %) |
| Parakeet STT on **dGPU** (CUDA) | 0.221 s | 0.218 s (**−1.3 %**) | 0.054 s | 0.071 s (+30.6 %) |

Under a 5-way flood:

| Load | STT on CPU | STT on dGPU |
|---|---:|---:|
| STT during a TTS flood ×5 | 0.245 s (+15.5 %) | 0.257 s (+16.3 %) |
| TTS during an STT flood ×5 | 0.075 s (+37.5 %) | 0.061 s (+13.1 %) |

**STT barely notices TTS; TTS pays a small tax for STT.** Kokoro is so fast (54 ms idle)
and Parakeet is the heavier job, so contention is asymmetric. Even the worst case measured
— TTS under a 5-way STT flood with Parakeet on CPU — is 75 ms, still ~118× realtime. Both
devices keep headroom: 3.3 GB of 16 GB VRAM with both loaded.

The TTS flood row is throughput, not latency: the service is single-worker with a
generation lock, so five concurrent TTS requests serialise (0.235 s each) instead of
running in parallel. Raising `--workers` will not add TTS throughput — the GPU is the
binding constraint.

**Recommendation:** run Parakeet on the **CPU** and leave the GPU to Kokoro. The dGPU path
is not faster for this model (20.23 vs 29.91 RTFx in the batch bench) and it adds GPU
memory pressure. CPU Parakeet costs the TTS +37 % in the worst flood case; dGPU Parakeet
costs +13 % but is slower per request. Both are fine — CPU is the simpler choice.

---

## Install

Requires: Docker with the NVIDIA Container Toolkit (r9slug) or a ROCm userspace (wimpy),
and `uv` for the TTS service.

```bash
# 1. TTS service — already installed as a systemd unit
systemctl is-enabled kokoro-cuda.service     # unit in systemd/

# 2. STT service (permanent, CPU) — unit in systemd/
sudo install -m 644 systemd/parakeet-stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now parakeet-stt
sudo ufw allow from 192.168.8.0/24 to any port 8094 proto tcp

# 3. ASR bench kit — build the images and run the matrix
cd benchmarks/asr-bench
bash run_device.sh cpu        # all six models, CPU
bash run_device.sh cuda       # parakeet on the dGPU (r9slug)
bash run_device.sh rocm       # parakeet on the dGPU (wimpy)
python3 aggregate.py          # rebuild benchmarks/benchmark-comparison.md
```

`run_device.sh` bakes the models into the images on first run and skips the build when
they are present. On r9slug it re-execs itself under `newgrp docker` if the shell is not
in the `docker` group.

## Usage

```bash
# TTS → file
curl -X POST http://r9slug.home.lan:8093/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"model":"kokoro","input":"Hello from r9slug.","voice":"af_bella","response_format":"wav"}' \
     -o out.wav

# TTS → streaming PCM (omit response_format)

# STT → text
curl -X POST http://r9slug.home.lan:8094/v1/audio/transcriptions \
     -F file=@clip.wav -F model=whisper-1
```

## Gotchas worth knowing

1. **Moonshine has no GPU path.** `moonshine-voice` bundles a 22 MB CPU-only onnxruntime
   with no provider options, so it is CPU-only on every host. Only Parakeet (`onnx-asr`)
   was run on the dGPUs.
2. **A silent CPU fallback looks like a GPU result.** The GPU runners now refuse to start
   if the requested execution provider is not in `onnxruntime.get_available_providers()`.
3. **onnxruntime's CUDA EP needs CUDA 13 libs.** The container image does not ship them;
   the runner mounts `/opt/cuda/lib64` and a cuDNN 9 from the host. A bare
   `pip install onnxruntime-gpu` in a slim image fails with `libcublasLt.so.13: cannot
   open shared object file` and falls back to CPU.
4. **ROCm in a container needs a glibc that matches the host.** wimpy's ROCm userspace
   requires GLIBC 2.43/2.44; a Debian slim container has 2.36. The ROCm run is therefore
   done host-side in a `uv` venv on wimpy, not in a container.
5. **The first generation in a fresh process costs ~15 s.** Kokoro's CUDA warm-up
   (kernel selection) is why the service is resident and pre-warmed before it reports
   ready.
6. **CPU numbers are load-sensitive, and hermesvm01 shares a physical host with wimpy.**
   A hermesvm01 run made while the host was busy came out ~1.7× slow and was discarded.
   Run the CPU matrix only when the host is idle.
7. **WER differences of ±0.3 pp between hosts are noise** on a 73-clip corpus; the model
   and corpus are identical everywhere.

## Repository layout

```
systemd/     kokoro-tts.service (resident CUDA TTS unit)
benchmarks/  asr-bench/            the bench kit (images, scripts, corpus, raw results)
             benchmark-comparison.md   cross-host chart in slug's comparison format
             stage3-cpu-stt.log / stage3-cuda-stt.log   concurrency runs
docs/        architecture.html    interactive topology diagram
scripts/     run_matrix.sh         cross-machine orchestration
             run_device.sh is in benchmarks/asr-bench/
```

## Security

No credentials, tokens, model weights, or private hostnames are stored in this repository.
The TTS endpoint is **unauthenticated plain HTTP** and is intended for a trusted LAN only —
do not expose it to the internet.
