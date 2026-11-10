#!/usr/bin/env bash
# LatentMovement 4-Thread Launcher
# Restricts CPU affinity and PyTorch/OpenMP runtime to 4 CPU threads (cores 0-3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

echo "=================================================="
echo " 🎬 Launching LatentMovement (Limited to 4 Threads)"
echo " ⚙️ CPU Affinity Mask: Cores 0, 1, 2, 3"
echo " 📁 Directory: ${SCRIPT_DIR}"
echo "=================================================="

exec taskset -c 0-3 "${SCRIPT_DIR}/LatentMovement-x86_64.AppImage" "$@"
