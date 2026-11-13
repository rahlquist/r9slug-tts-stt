#!/usr/bin/env bash
# Print a one-block host inventory for benchmark reports.
{
  echo "host: $(hostname)"
  echo "date: $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "cores: $(nproc)"
  echo "ram_gb: $(free -g | awk '/^Mem:/{print $2}')"
  echo "docker: $(docker --version 2>/dev/null || echo none)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia_driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "nvidia_gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  fi
  if [ -x /opt/rocm/bin/rocminfo ]; then
    echo "rocm_gpu: $(/opt/rocm/bin/rocminfo 2>/dev/null | grep 'Marketing Name' | grep -iv 'Ryzen\|Radeon 610M' | head -1 | cut -d: -f2- | sed 's/^ *//')"
    echo "rocm_runtime: $(/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 'Runtime Version' | cut -d: -f2- | sed 's/^ *//')"
  fi
}
