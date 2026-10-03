#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Set PDEOBS_ENV to the installed environment path.}"
: "${PDEOBS_DATA:?Set PDEOBS_DATA to the dataset root.}"
: "${PDEOBS_RUNS:?Set PDEOBS_RUNS to the run root.}"

if [[ -n "${PDEOBS_MODULE_SETUP:-}" ]]; then
  if [[ ! -r "$PDEOBS_MODULE_SETUP" ]]; then
    echo "PDEOBS_MODULE_SETUP is not readable: $PDEOBS_MODULE_SETUP" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$PDEOBS_MODULE_SETUP"
fi
if [[ -n "${PDEOBS_MODULES:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "PDEOBS_MODULES is set, but the environment-modules command is unavailable." >&2
    exit 2
  fi
  if [[ "${PDEOBS_MODULE_PURGE:-0}" == "1" ]]; then
    module purge
  fi
  read -r -a pdeobs_modules <<<"$PDEOBS_MODULES"
  module load "${pdeobs_modules[@]}"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a Slurm allocation." >&2
  exit 2
fi
if ! command -v "${PDEOBS_SRUN:-srun}" >/dev/null 2>&1; then
  echo "Slurm srun is unavailable; set PDEOBS_SRUN to its absolute path if needed." >&2
  exit 2
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
  echo "Run hpc/slurm/bootstrap.sh from this checkout before submitting jobs." >&2
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
    echo "Rerun hpc/slurm/bootstrap.sh to replace the editable installation." >&2
    exit 2
    ;;
esac
