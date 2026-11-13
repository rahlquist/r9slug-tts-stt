# ASR bench comparison — r9slug / wimpy / hermesvm01, CPU + dGPU (2026-11-12)

Same kit, same 73-clip LibriSpeech corpus (481 s), same slug-protocol probe 
(0.5 s / 11 s audio pair, median of 5). Containers: `asr-parakeet` / `asr-moonshine`,
`OMP_NUM_THREADS=6`, models baked into the images.

## Host stack

| Host | CPU | Cores | RAM | Accelerator | Runtime |
|---|---|---:|---:|---|---|
| r9slug | AMD Ryzen 9 3900X 12-Core Processor | 24 | 62 GB | NVIDIA GeForce RTX 5060 Ti (driver 620.70.06) | kernel 7.3.3-1-cachyos |
| wimpy | AMD Ryzen 9 9900X 12-Core Processor | 24 | 60 GB | AMD Radeon AI PRO R9700 (ROCm runtime 1.18) | kernel 7.3.1-1-cachyos |
| hermesvm01 | AMD Ryzen 9 9900X 12-Core Processor | 6 | 15 GB | CPU only | kernel 7.3.1-1-cachyos |

## Word error rate and throughput by model

WER over reference words (jiwer, normalized). RTFx = audio seconds / wall seconds;
higher is faster. `dgpu` = the host's discrete GPU (CUDA on r9slug, ROCm on wimpy).
`—` = not run on that host.

| Model | Device | Metric | r9slug | wimpy | hermesvm01 |
|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3 int8 | cpu | WER % | 4.70 | 4.78 | 4.78 |
| Parakeet TDT 0.6B v3 int8 | cpu | RTFx | 29.91 | 42.98 | 46.70 |
| Parakeet TDT 0.6B v3 int8 | dgpu | WER % | 4.96 | 4.96 | — |
| Parakeet TDT 0.6B v3 int8 | dgpu | RTFx | 20.23 | 43.58 | — |
| Moonshine Base (offline) | cpu | WER % | 31.74 | 31.39 | 31.39 |
| Moonshine Base (offline) | cpu | RTFx | 22.45 | 40.35 | 40.16 |
| Moonshine Tiny (offline) | cpu | WER % | 36.26 | 36.09 | 36.35 |
| Moonshine Tiny (offline) | cpu | RTFx | 30.34 | 56.91 | 57.83 |
| Moonshine Tiny Streaming | cpu | WER % | 11.83 | 11.83 | 11.83 |
| Moonshine Tiny Streaming | cpu | RTFx | 8.12 | 9.89 | 7.30 |
| Moonshine Small Streaming | cpu | WER % | 7.30 | 7.22 | 7.22 |
| Moonshine Small Streaming | cpu | RTFx | 4.42 | 4.87 | 5.05 |
| Moonshine Medium Streaming | cpu | WER % | 5.48 | 5.39 | 5.57 |
| Moonshine Medium Streaming | cpu | RTFx | 3.05 | 3.71 | 3.61 |

## Slug-protocol probe (per-request latency)

Median of 5 requests, warmup excluded. `fixed` is the 0.5 s request median;
`marginal` is the per-audio-second slope between 0.5 s and 11 s.

| Engine | Host / device | 0.5 s req (s) | 11 s req (s) | fixed (s) | marginal (s/audio-s) |
|---|---|---|---|---|---|
| moonshine MEDIUM_STREAMING | r9slug / cpu | 0.0058 | 1.2288 | 0.0058 | 0.11505 |
| moonshine SMALL_STREAMING | r9slug / cpu | 0.0058 | 0.8263 | 0.0058 | 0.07719 |
| moonshine TINY_STREAMING | r9slug / cpu | 0.0059 | 0.4722 | 0.0059 | 0.04387 |
| parakeet tdt-0.6b-v3-int8 | r9slug / cpu | 0.0688 | 0.3505 | 0.0688 | 0.02650 |
| parakeet tdt-0.6b-v3-int8 GPU (CUDAExecutionProvider,CPUExecutionProvider) | r9slug / cuda | 0.0581 | 0.3356 | 0.0581 | 0.02611 |
| moonshine MEDIUM_STREAMING | wimpy / cpu | 0.0023 | 0.7414 | 0.0023 | 0.06953 |
| moonshine SMALL_STREAMING | wimpy / cpu | 0.0032 | 0.5159 | 0.0032 | 0.04823 |
| moonshine TINY_STREAMING | wimpy / cpu | 0.0033 | 0.2701 | 0.0033 | 0.02510 |
| parakeet tdt-0.6b-v3-int8 | wimpy / cpu | 0.0811 | 0.1955 | 0.0811 | 0.01076 |
| parakeet tdt-0.6b-v3-int8 GPU (ROCMExecutionProvider,CPUExecutionProvider) | wimpy / rocm | 0.0656 | 0.2824 | 0.0656 | 0.02040 |
| moonshine MEDIUM_STREAMING | hermesvm01 / cpu | 0.0025 | 0.7754 | 0.0025 | 0.07271 |
| moonshine SMALL_STREAMING | hermesvm01 / cpu | 0.0026 | 0.5301 | 0.0026 | 0.04962 |
| moonshine TINY_STREAMING | hermesvm01 / cpu | 0.0025 | 0.2993 | 0.0025 | 0.02792 |
| parakeet tdt-0.6b-v3-int8 | hermesvm01 / cpu | 0.0813 | 0.1774 | 0.0813 | 0.00904 |

## Reference — slug (2026-08-06)

| Engine | Host | 11 s req (s) |
|---|---|---:|
| whisper large-v3-turbo (iGPU, Vulkan) | slug | 1.74 |
| whisper large-v3-turbo (NPU, FLM) | slug | 2.70 |

Slug figures are end-to-end HTTP request latencies (~2.1 s fixed server overhead
included); the numbers above are in-process. Compare request latencies, not overheads.

## Notes

- Moonshine has no GPU path: `moonshine-voice` bundles a CPU-only onnxruntime
  (22 MB, no provider options), so it is CPU-only on every host. Only Parakeet
  (`onnx-asr`) was run on the dGPUs.
- Parakeet on the dGPU is **not** faster than 6 CPU threads on either host. Both
  GPU runs are host/device-copy bound (onnxruntime reports hundreds of Memcpy
  nodes for the CUDA EP), and the model is small enough that the CPU wins.
- r9slug CUDA and wimpy ROCm results were verified to actually execute on the
  GPU (CUDA EP session providers; AMD `gpu_busy_percent` 27–60% during the run).
- hermesvm01 is a VM on wimpy and has no dGPU, so it is CPU-only by design.
- WER differences of ±0.3 pp between hosts are run-to-run noise on a 73-clip
  corpus; the model and corpus are identical everywhere.
- **CPU numbers are load-sensitive and hermesvm01 shares a physical host with
  wimpy.** The first hermesvm01 run was made while the host was still busy and
  came out ~1.7× slow (parakeet RTFx 26.8 vs 46.7); it was discarded and re-run
  on an idle host. Run the CPU matrix only when the host is idle.
