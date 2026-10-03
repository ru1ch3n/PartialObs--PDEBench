#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Source your scratch env.sh first.}"
: "${PDEOBS_DATA:?Source your scratch env.sh first.}"
: "${PDEOBS_RUNS:?Source your scratch env.sh first.}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty checkout." >&2
  exit 2
fi

sbatch_command="${PDEOBS_SBATCH:-sbatch}"
squeue_command="${PDEOBS_SQUEUE:-squeue}"
sacct_command="${PDEOBS_SACCT:-sacct}"
for command_path in "$sbatch_command" "$squeue_command" "$sacct_command"; do
  if ! command -v "$command_path" >/dev/null 2>&1; then
    echo "Required Slurm command is unavailable: $command_path" >&2
    exit 2
  fi
done
commit_short="$(git rev-parse --short=8 HEAD)"
campaign_name="${PDEOBS_VALIDATION_CAMPAIGN:-numerics-validation20-$commit_short}"
if [[ ! "$campaign_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "PDEOBS_VALIDATION_CAMPAIGN must contain only letters, numbers, dot, underscore, or dash." >&2
  exit 2
fi

mkdir -p logs "$PDEOBS_DATA/plans"
plan="$PDEOBS_DATA/plans/$campaign_name.jsonl"
resolved="$PDEOBS_DATA/plans/$campaign_name.resolved.yaml"
output="$PDEOBS_DATA/$campaign_name"
campaign="$PDEOBS_DATA/$campaign_name.campaign.txt"

if [[ ! -e "$plan" ]]; then
  if [[ -e "$campaign" || -d "$output" ]]; then
    echo "Refusing to create a new plan over an existing campaign/output." >&2
    exit 2
  fi
  "$PDEOBS_ENV/bin/python" -m pdeobs plan \
    --config configs/dataset/numerics_validation20.yaml \
    --output "$plan"
fi
test -s "$plan"
test -s "$resolved"
mkdir -p "$output"

read -r task_count sample_count < <(
  "$PDEOBS_ENV/bin/python" - "$plan" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(len(rows), sum(int(row["sample_count"]) for row in rows))
PY
)
if [[ "$task_count" != 840 || "$sample_count" != 5600 ]]; then
  echo "Unexpected validation plan: tasks=$task_count samples=$sample_count" >&2
  exit 2
fi

concurrency="${PDEOBS_GENERATION_CONCURRENCY:-20}"
partition="${PDEOBS_CPU_PARTITION:-}"
generation_time="${PDEOBS_GENERATION_TIME_LIMIT:-00:30:00}"
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "PDEOBS_GENERATION_CONCURRENCY must be a positive integer." >&2
  exit 2
fi

# Submit exactly one bounded window per invocation. Set the ceiling from the
# local site's MaxSubmitJobs/MaxJobs policy instead of assuming a cluster QOS.
max_queued="${PDEOBS_MAX_QUEUED_TASKS:-100}"
window_size="${PDEOBS_GENERATION_WINDOW_SIZE:-$max_queued}"
if [[ ! "$max_queued" =~ ^[1-9][0-9]*$ ]]; then
  echo "PDEOBS_MAX_QUEUED_TASKS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$window_size" =~ ^[1-9][0-9]*$ ]] || (( window_size > max_queued )); then
  echo "PDEOBS_GENERATION_WINDOW_SIZE must be between 1 and $max_queued." >&2
  exit 2
fi

if [[ ! -e "$campaign" ]]; then
  {
    echo "commit=$(git rev-parse HEAD)"
    echo "plan=$plan"
    echo "resolved_config=$resolved"
    echo "output=$output"
    echo "task_count=$task_count"
    echo "sample_count=$sample_count"
    echo "concurrency=$concurrency"
    echo "partition=${partition:-scheduler-default}"
    echo "generation_time=$generation_time"
    echo "window_size=$window_size"
    echo "next_start=0"
    echo "publication_ready=false"
  } >"$campaign"
fi

next_start="$(awk -F= '$1 == "next_start" {value=$2} END {print value}' "$campaign")"
requested_start="${PDEOBS_WINDOW_START:-$next_start}"
if [[ "$next_start" == "complete" ]]; then
  echo "Campaign generation is already fully submitted; inspect $campaign." >&2
  exit 2
fi
if [[ ! "$requested_start" =~ ^[0-9]+$ ]] || [[ "$requested_start" != "$next_start" ]]; then
  echo "Expected PDEOBS_WINDOW_START=$next_start, got $requested_start." >&2
  exit 2
fi

previous="$(awk -F= '$1 == "last_job" {value=$2} END {print value}' "$campaign")"
previous_count="$(awk -F= '$1 == "last_window_count" {value=$2} END {print value}' "$campaign")"
if [[ -n "$previous" ]]; then
  mapfile -t previous_states < <(
    "$sacct_command" -j "$previous" -X -n -o State | tr -d ' ' | sed '/^$/d'
  )
  if (( ${#previous_states[@]} != previous_count )); then
    echo "Previous window $previous is not fully present in accounting yet." >&2
    exit 2
  fi
  for state in "${previous_states[@]}"; do
    if [[ "$state" != "COMPLETED" ]]; then
      echo "Previous window $previous has state $state; inspect failures before continuing." >&2
      exit 2
    fi
  done
fi

start="$requested_start"
stop=$((start + window_size - 1))
(( stop >= task_count )) && stop=$((task_count - 1))
site_args=()
if [[ -n "$partition" ]]; then
  site_args+=(--partition="$partition")
fi
if [[ -n "${PDEOBS_ACCOUNT:-}" ]]; then
  site_args+=(--account="$PDEOBS_ACCOUNT")
fi
if [[ -n "${PDEOBS_QOS:-}" ]]; then
  site_args+=(--qos="$PDEOBS_QOS")
fi
if [[ -n "${PDEOBS_RESERVATION:-}" ]]; then
  site_args+=(--reservation="$PDEOBS_RESERVATION")
fi

job="$("$sbatch_command" --parsable \
  "${site_args[@]}" \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time="$generation_time" \
  --array="${start}-${stop}%${concurrency}" \
  hpc/slurm/generate_array.sbatch "$resolved" "$output" "$plan")"
job="${job%%;*}"
count=$((stop - start + 1))
following=$((stop + 1))
{
  echo "window_${start}_${stop}=$job"
  echo "last_job=$job"
  echo "last_window_count=$count"
  if (( following < task_count )); then
    echo "next_start=$following"
  else
    echo "next_start=complete"
  fi
} >>"$campaign"

echo "Submitted generation window $start-$stop as job $job."
if (( following < task_count )); then
  echo "After every array element is COMPLETED, rerun with PDEOBS_WINDOW_START=$following."
else
  aggregate="$("$sbatch_command" --parsable \
    "${site_args[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=04:00:00 \
    --dependency="afterok:$job" \
    hpc/slurm/aggregate_cpu.sbatch \
    "$output" "$output/summary.json" "$plan" \
    --quality-strict --require-all-pdes)"
  aggregate="${aggregate%%;*}"
  echo "aggregate_job=$aggregate" >>"$campaign"
  echo "Final strict aggregate job: $aggregate"
fi
echo "Campaign record: $campaign"
echo "No model-training job was submitted."
"$squeue_command" -j "$job"
