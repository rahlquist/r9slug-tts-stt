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
        └── POST /v1/audio/transcriptions ─────► parakeet-stt (container) ► dGPU / CPU
                                                   (benchmarked, not yet a service)
```

| Port | Bind | Service | Device | Purpose |
|---|---|---|---|---|
| 8093 | `0.0.0.0` | `kokoro-tts` | **dGPU (CUDA)** | LAN-facing TTS, OpenAI shape |
| — | container | `parakeet-stt` | dGPU / CPU | Local STT, benchmarked |
| 8081 | slug | `whisper-proxy` | NPU / iGPU | STT that Hermes currently uses (remote) |

Hermes on r9slug currently consumes **slug** for both STT (`:8081`) and TTS (`:8092`);
r9slug's own Kokoro endpoint on `:8093` is available to LAN clients and is what the TTS
numbers below describe.

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

Parakeet is **10–26× faster than slug's whisper large-v3-turbo** on the identical 11 s
probe. Slug's figures are end-to-end HTTP requests (~2.1 s fixed server overhead
included); these are in-process. Compare request latencies, not overheads.

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
CPU path (42.98). Both GPU runs were verified to actually execute on the GPU — the CUDA
session reports `CUDAExecutionProvider`, and `gpu_busy_percent` on wimpy's Radeon reads
27–60 % during the run — so this is not a silent fallback. The cause is the model: at
0.6 B int8 the encoder is small enough that onnxruntime's host↔device transfers
(the CUDA EP alone inserts 742 Memcpy nodes) cost more than the CPU inference they save.

The dGPU is not wasted, though: it is the only path that serves **TTS** at 60–93× realtime
with 47 ms latency, and it leaves the CPU free for the STT workloads above.

---

## Install

Requires: Docker with the NVIDIA Container Toolkit (r9slug) or a ROCm userspace (wimpy),
and `uv` for the TTS service.

```bash
# 1. TTS service (r9slug) — already installed as a systemd user unit
systemctl --user enable --now kokoro-tts.service     # unit in systemd/

# 2. ASR bench kit — build the images and run the matrix
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
docs/        architecture.html    interactive topology diagram
scripts/     helper entry points
```

## Security

No credentials, tokens, model weights, or private hostnames are stored in this repository.
The TTS endpoint is **unauthenticated plain HTTP** and is intended for a trusted LAN only —
do not expose it to the internet.
