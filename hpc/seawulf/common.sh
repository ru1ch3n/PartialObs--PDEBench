#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Set PDEOBS_ENV to the installed environment path.}"
: "${PDEOBS_DATA:?Set PDEOBS_DATA to the dataset root.}"
: "${PDEOBS_RUNS:?Set PDEOBS_RUNS to the run root.}"

module purge
module load anaconda/3
if [[ -n "${PDEOBS_CUDA_MODULE:-}" ]]; then
  module load "$PDEOBS_CUDA_MODULE"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HDF5_USE_FILE_LOCKING=FALSE

repository_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repository_dir"

if [[ ! -x "$PDEOBS_ENV/bin/python" ]]; then
  echo "PDEOBS_ENV does not contain an executable Python: $PDEOBS_ENV" >&2
  exit 2
fi
