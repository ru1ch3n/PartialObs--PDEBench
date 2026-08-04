#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run bootstrap.sh inside a SeaWulf interactive or batch allocation." >&2
  exit 2
fi

: "${PDEOBS_ENV:?Set PDEOBS_ENV to a persistent home or project path.}"

module purge
module load anaconda/3

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_dir"

if [[ -n "${PDEOBS_WHEELHOUSE:-}" ]]; then
  if [[ ! -x "$PDEOBS_ENV/bin/python" ]]; then
    conda create --yes --prefix "$PDEOBS_ENV" python=3.11 pip
  fi
  "$PDEOBS_ENV/bin/python" -m pip install \
    --no-index --find-links "$PDEOBS_WHEELHOUSE" -r requirements.txt
  "$PDEOBS_ENV/bin/python" -m pip install --no-deps -e .
else
  if [[ -x "$PDEOBS_ENV/bin/python" ]]; then
    conda env update --yes --prefix "$PDEOBS_ENV" --file environment.yml --prune
  else
    conda env create --yes --prefix "$PDEOBS_ENV" --file environment.yml
  fi
fi

doctor_args=(doctor --cluster seawulf --offline)
if [[ "${PDEOBS_REQUIRE_GPU:-0}" == "1" ]]; then
  doctor_args+=(--gpu)
fi
"$PDEOBS_ENV/bin/python" -m pdeobs "${doctor_args[@]}"
