#!/usr/bin/env bash
# Run the ASR bench matrix for one device on THIS host.
# usage: run_device.sh <cpu|cuda|rocm>
#
# Moonshine has no GPU path (it bundles a CPU-only onnxruntime), so the
# moonshine benches always run on CPU regardless of $1. Only Parakeet has a
# CUDA/ROCm variant.
set -euo pipefail

DEV="${1:?usage: run_device.sh <cpu|cuda|rocm>}"
KIT="$(cd "$(dirname "$0")" && pwd)"

# r9slug's own shell is not in the docker group; re-exec under newgrp if needed.
if ! docker info >/dev/null 2>&1; then
  echo "[setup] docker needs the docker group; re-execing via newgrp" >&2
  exec newgrp docker -c "bash '$0' $*"
fi

MACHINE="${BENCH_MACHINE:-$(hostname)}"
RESULTS="$KIT/results/$MACHINE/$DEV"
mkdir -p "$RESULTS"

log() { echo "[$DEV] $*"; }

# ---- images -------------------------------------------------------------
case "$DEV" in
  cpu)
    docker image inspect asr-parakeet:latest >/dev/null 2>&1 \
      || docker build -t asr-parakeet:latest -f "$KIT/parakeet/Dockerfile" "$KIT"
    ;;
  cuda|rocm)
    docker image inspect "asr-parakeet:$DEV" >/dev/null 2>&1 \
      || docker build -t "asr-parakeet:$DEV" -f "$KIT/parakeet/Dockerfile.$DEV" "$KIT"
    ;;
  *) echo "unknown device: $DEV" >&2; exit 1 ;;
esac
docker image inspect asr-moonshine:latest >/dev/null 2>&1 \
  || docker build -t asr-moonshine:latest -f "$KIT/moonshine/Dockerfile" "$KIT"

# ---- parakeet ------------------------------------------------------------
if [ "$DEV" = "cpu" ]; then
  log "parakeet cpu bench"
  docker run --rm -v "$KIT/data:/data" -v "$RESULTS:/results" asr-parakeet:latest

  log "parakeet cpu probe"
  docker run --rm -v "$KIT/data:/data" -v "$RESULTS:/results" -v "$KIT/scripts:/scripts" \
    asr-parakeet:latest python /scripts/bench_probe.py parakeet
else
  if [ "$DEV" = "cuda" ]; then
    PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
  else
    PROVIDERS="ROCMExecutionProvider,CPUExecutionProvider"
  fi
  log "parakeet $DEV bench ($PROVIDERS)"
  docker run --rm --gpus all -v "$KIT/data:/data" -v "$RESULTS:/results" \
    -e ASR_PROVIDERS="$PROVIDERS" \
    -e BENCH_OUT=/results/parakeet_tdt-0.6b-v3_int8-gpu.json \
    "asr-parakeet:$DEV"

  log "parakeet $DEV probe"
  docker run --rm --gpus all -v "$KIT/data:/data" -v "$RESULTS:/results" -v "$KIT/scripts:/scripts" \
    -e ASR_PROVIDERS="$PROVIDERS" -e BENCH_OUT=/results/probe_parakeet_gpu.json \
    "asr-parakeet:$DEV" python /scripts/bench_probe_gpu.py parakeet
fi

# ---- moonshine (CPU only) ------------------------------------------------
for spec in "BASE offline" "TINY offline" "TINY_STREAMING stream" "SMALL_STREAMING stream" "MEDIUM_STREAMING stream"; do
  set -- $spec
  log "moonshine $1 $2"
  docker run --rm -v "$KIT/data:/data" -v "$RESULTS:/results" \
    asr-moonshine:latest python /bench.py "$1" "$2"
done

for arch in TINY_STREAMING SMALL_STREAMING MEDIUM_STREAMING; do
  log "moonshine $arch probe"
  docker run --rm -v "$KIT/data:/data" -v "$RESULTS:/results" -v "$KIT/scripts:/scripts" \
    asr-moonshine:latest python /scripts/bench_probe.py moonshine "$arch"
done

# ---- score ---------------------------------------------------------------
log "judge"
docker run --rm -v "$KIT/data:/data" -v "$RESULTS:/results" -v "$KIT/scripts:/scripts" \
  asr-moonshine:latest python /scripts/judge.py

log "done -> $RESULTS"
