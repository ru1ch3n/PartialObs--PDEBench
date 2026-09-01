#!/usr/bin/env bash
set -Eeuo pipefail

plan_path="${1:?Usage: submit_generation.sh PLAN CONFIG OUTPUT [START STOP]}"
config_path="${2:?Usage: submit_generation.sh PLAN CONFIG OUTPUT [START STOP]}"
output_root="${3:?Usage: submit_generation.sh PLAN CONFIG OUTPUT [START STOP]}"
: "${PDEOBS_ENV:?Set PDEOBS_ENV to the installed environment path.}"

sbatch_command="${PDEOBS_SBATCH:-sbatch}"
if ! command -v "$sbatch_command" >/dev/null 2>&1; then
  echo "Slurm sbatch is unavailable; set PDEOBS_SBATCH to its absolute path if needed." >&2
  exit 2
fi

if [[ ! -x "$PDEOBS_ENV/bin/python" ]]; then
  echo "PDEOBS_ENV does not contain an executable Python: $PDEOBS_ENV" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${PDEOBS_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "Refusing to submit from a dirty Git tree. Commit changes or set PDEOBS_ALLOW_DIRTY=1." >&2
  exit 2
fi

mkdir -p logs "$output_root"
task_count="$("$PDEOBS_ENV/bin/python" -c 'import sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))' "$plan_path")"
if (( task_count < 1 )); then
  echo "Generation plan has no jobs: $plan_path" >&2
  exit 2
fi

max_index=$((task_count - 1))
start_index="${4:-0}"
stop_index="${5:-$max_index}"
concurrency="${PDEOBS_GENERATION_CONCURRENCY:-4}"
max_queued="${PDEOBS_MAX_QUEUED_TASKS:-100}"

for value_name in start_index stop_index concurrency max_queued; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( concurrency < 1 || max_queued < 1 )); then
  echo "PDEOBS_GENERATION_CONCURRENCY and PDEOBS_MAX_QUEUED_TASKS must be positive." >&2
  exit 2
fi
if (( start_index > stop_index || stop_index > max_index )); then
  echo "Requested array range ${start_index}-${stop_index}; valid plan indices are 0-${max_index}." >&2
  exit 2
fi
window_size=$((stop_index - start_index + 1))
if (( window_size > max_queued )); then
  echo "Refusing to queue $window_size array tasks; the configured limit is $max_queued." >&2
  echo "Pass bounded START STOP arguments, for example 0 $((max_queued - 1)), and wait for each window before submitting the next." >&2
  exit 2
fi

sbatch_args=(--array="${start_index}-${stop_index}%${concurrency}")
if [[ -n "${PDEOBS_CPU_PARTITION:-}" ]]; then
  sbatch_args+=(--partition="$PDEOBS_CPU_PARTITION")
fi
if [[ -n "${PDEOBS_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account="$PDEOBS_ACCOUNT")
fi
if [[ -n "${PDEOBS_QOS:-}" ]]; then
  sbatch_args+=(--qos="$PDEOBS_QOS")
fi
if [[ -n "${PDEOBS_RESERVATION:-}" ]]; then
  sbatch_args+=(--reservation="$PDEOBS_RESERVATION")
fi

job_id="$("$sbatch_command" --parsable "${sbatch_args[@]}" \
  hpc/slurm/generate_array.sbatch "$config_path" "$output_root" "$plan_path")"
printf '%s\n' "${job_id%%;*}"
