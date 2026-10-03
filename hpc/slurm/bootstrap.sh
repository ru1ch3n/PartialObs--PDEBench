#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run bootstrap.sh inside a Slurm interactive or batch allocation." >&2
  exit 2
fi

: "${PDEOBS_ENV:?Set PDEOBS_ENV to a persistent home or project path.}"

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

environment_manager="${PDEOBS_ENV_MANAGER:-conda}"
if ! command -v "$environment_manager" >/dev/null 2>&1; then
  echo "Environment manager is unavailable: $environment_manager" >&2
  echo "Load it through PDEOBS_MODULES/PDEOBS_MODULE_SETUP or set PDEOBS_ENV_MANAGER." >&2
  exit 2
fi

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_dir"
environment_file="${PDEOBS_ENV_FILE:-environment-generation.yml}"
if [[ ! -f "$environment_file" ]]; then
  echo "PDEOBS environment file does not exist: $environment_file" >&2
  exit 2
fi

if ! repository_commit="$(git rev-parse --verify HEAD 2>/dev/null)"; then
  echo "bootstrap.sh must run from a Git checkout." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to build a production environment from a dirty Git checkout." >&2
  echo "Commit or stash the changes, then rerun bootstrap.sh." >&2
  exit 2
fi

if [[ -n "${PDEOBS_WHEELHOUSE:-}" ]]; then
  if [[ ! -x "$PDEOBS_ENV/bin/python" ]]; then
    "$environment_manager" create --yes --prefix "$PDEOBS_ENV" python=3.11 pip
  fi
  "$PDEOBS_ENV/bin/python" -m pip install \
    --no-index --find-links "$PDEOBS_WHEELHOUSE" -r requirements.txt
  "$PDEOBS_ENV/bin/python" -m pip install \
    --no-index --find-links "$PDEOBS_WHEELHOUSE" "setuptools>=68" wheel
else
  if [[ -x "$PDEOBS_ENV/bin/python" ]]; then
    "$environment_manager" env update --yes --prefix "$PDEOBS_ENV" --file "$environment_file" --prune
  else
    "$environment_manager" env create --yes --prefix "$PDEOBS_ENV" --file "$environment_file"
  fi
fi

wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/pdeobs-wheel.XXXXXX")"
trap 'rm -rf -- "$wheel_dir"' EXIT
"$PDEOBS_ENV/bin/python" -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir "$wheel_dir" .
wheels=("$wheel_dir"/pdeobs-*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
  echo "Expected exactly one PDE-OBS wheel under $wheel_dir." >&2
  exit 2
fi
"$PDEOBS_ENV/bin/python" -m pip install \
  --force-reinstall --no-deps "${wheels[0]}"

installed_module="$(
  cd /
  "$PDEOBS_ENV/bin/python" -c \
    'import pathlib, pdeobs; print(pathlib.Path(pdeobs.__file__).resolve())'
)"
case "$installed_module" in
  "$repository_dir"/*)
    echo "PDE-OBS still resolves from the checkout: $installed_module" >&2
    exit 2
    ;;
esac
commit_marker="$PDEOBS_ENV/.pdeobs-git-commit"
printf '%s\n' "$repository_commit" > "${commit_marker}.tmp"
mv -f "${commit_marker}.tmp" "$commit_marker"
printf '%s\n' "$environment_file" > "$PDEOBS_ENV/.pdeobs-environment-file"

doctor_args=(doctor --cluster slurm --offline)
if [[ "${PDEOBS_REQUIRE_GPU:-0}" == "1" ]]; then
  doctor_args+=(--gpu)
fi
"$PDEOBS_ENV/bin/python" -m pdeobs "${doctor_args[@]}"
