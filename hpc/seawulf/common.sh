#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Set PDEOBS_ENV to the installed environment path.}"
: "${PDEOBS_DATA:?Set PDEOBS_DATA to the dataset root.}"
: "${PDEOBS_RUNS:?Set PDEOBS_RUNS to the run root.}"

module purge
module load slurm
module load anaconda/3
if [[ -n "${PDEOBS_CUDA_MODULE:-}" ]]; then
  module load "$PDEOBS_CUDA_MODULE"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HDF5_USE_FILE_LOCKING=FALSE

submission_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$submission_dir"
repository_dir="$(pwd -P)"

if [[ ! -x "$PDEOBS_ENV/bin/python" ]]; then
  echo "PDEOBS_ENV does not contain an executable Python: $PDEOBS_ENV" >&2
  exit 2
fi

if ! repository_commit="$(git rev-parse --verify HEAD 2>/dev/null)"; then
  echo "The submission directory is not a Git checkout: $repository_dir" >&2
  exit 2
fi
commit_marker="$PDEOBS_ENV/.pdeobs-git-commit"
if [[ ! -f "$commit_marker" ]]; then
  echo "The environment has no PDE-OBS commit marker: $commit_marker" >&2
  echo "Run hpc/seawulf/bootstrap.sh from this checkout before submitting jobs." >&2
  exit 2
fi
installed_commit="$(<"$commit_marker")"
if [[ "$installed_commit" != "$repository_commit" ]]; then
  echo "PDE-OBS environment/checkout mismatch." >&2
  echo "  installed commit: $installed_commit" >&2
  echo "  checkout commit:  $repository_commit" >&2
  echo "Use an environment built for this commit or rerun bootstrap.sh." >&2
  exit 2
fi
installed_module="$("$PDEOBS_ENV/bin/python" -c 'import pathlib, pdeobs; print(pathlib.Path(pdeobs.__file__).resolve())')"
case "$installed_module" in
  "$repository_dir"/*)
    echo "PDE-OBS resolves from the checkout instead of the installed wheel: $installed_module" >&2
    echo "Rerun hpc/seawulf/bootstrap.sh to replace the editable installation." >&2
    exit 2
    ;;
esac
