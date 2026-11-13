#!/usr/bin/env bash
# Run the full cross-machine ASR bench matrix and rebuild the comparison chart.
#
#   r9slug      : CPU + dGPU (CUDA)      — local
#   wimpy       : CPU + dGPU (ROCm)      — the ROCm run is host-side (uv venv), not a container
#   hermesvm01  : CPU only (VM on wimpy) — must run when wimpy is idle
#
# usage: scripts/run_matrix.sh
set -euo pipefail

KIT="$(cd "$(dirname "$0")/../benchmarks/asr-bench" && pwd)"
SSH="ssh -o BatchMode=yes"

log() { echo "[matrix] $*"; }

# ---- r9slug (local) -------------------------------------------------------
log "r9slug: CPU"
BENCH_MACHINE=r9slug bash "$KIT/run_device.sh" cpu

log "r9slug: dGPU (CUDA)"
# run_device.sh cuda mounts the host CUDA/cuDNN libs the CUDA EP needs.
BENCH_MACHINE=r9slug bash "$KIT/run_device.sh" cuda

# ---- wimpy (CPU then ROCm) ------------------------------------------------
log "wimpy: syncing kit"
rsync -a -e "$SSH" "$KIT/" wimpy.home.lan:asr-bench-kit/

log "wimpy: CPU"
$SSH wimpy.home.lan 'cd ~/asr-bench-kit && BENCH_MACHINE=wimpy bash run_device.sh cpu'

log "wimpy: dGPU (ROCm, host venv — see README gotcha 4)"
$SSH wimpy.home.lan 'cd ~/asr-bench-kit && mkdir -p results/wimpy/rocm && \
  uv venv --python 3.14 .venv >/dev/null 2>&1 || true && \
  uv pip install --python .venv/bin/python "onnxruntime-rocm==1.22.2.post3" >/dev/null && \
  uv pip install --python .venv/bin/python --no-deps "onnx-asr[hub]==0.12.0" >/dev/null && \
  uv pip install --python .venv/bin/python numpy typing-extensions huggingface-hub >/dev/null && \
  OMP_NUM_THREADS=6 HF_HOME=$PWD/models/hf ASR_MODEL_PATH=$PWD/models/parakeet \
  ASR_MANIFEST=$PWD/data/manifest.json ASR_PROBE_DIR=$PWD/data/probe \
  ASR_PROVIDERS=ROCMExecutionProvider,CPUExecutionProvider \
  BENCH_OUT=$PWD/results/wimpy/rocm/parakeet_gpu.json \
  .venv/bin/python scripts/bench_parakeet_gpu.py'

# ---- hermesvm01 (CPU only, and only once wimpy is idle) -------------------
log "hermesvm01: syncing kit"
rsync -a -e "$SSH" "$KIT/" hermesvm01.home.lan:asr-bench-kit/

log "hermesvm01: CPU (VM on wimpy — host must be idle)"
$SSH hermesvm01.home.lan 'cd ~/asr-bench-kit && BENCH_MACHINE=hermesvm01 bash run_device.sh cpu'

# ---- collect + chart ------------------------------------------------------
log "collecting results"
mkdir -p "$KIT/results/wimpy/cpu" "$KIT/results/wimpy/rocm" "$KIT/results/hermesvm01/cpu"
rsync -a -e "$SSH" wimpy.home.lan:asr-bench-kit/results/wimpy/ "$KIT/results/wimpy/"
rsync -a -e "$SSH" hermesvm01.home.lan:asr-bench-kit/results/hermesvm01/ "$KIT/results/hermesvm01/"

for h in r9slug wimpy hermesvm01; do
  bash "$KIT/hostinfo.sh" > "$KIT/results/hostinfo/$h.txt" 2>/dev/null || \
    $SSH "$h.home.lan" 'cd ~/asr-bench-kit && bash hostinfo.sh' > "$KIT/results/hostinfo/$h.txt"
done

log "rebuilding chart"
python3 "$KIT/aggregate.py"
