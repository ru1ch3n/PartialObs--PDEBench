#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Source the scratch env.sh first.}"
: "${PDEOBS_DATA:?Source the scratch env.sh first.}"
: "${PDEOBS_RUNS:?Source the scratch env.sh first.}"

sbatch_command="${PDEOBS_SBATCH:-sbatch}"
squeue_command="${PDEOBS_SQUEUE:-squeue}"
for command_path in "$sbatch_command" "$squeue_command"; do
  if ! command -v "$command_path" >/dev/null 2>&1; then
    echo "Required Slurm command is unavailable: $command_path" >&2
    exit 2
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty checkout." >&2
  exit 2
fi

commit="$(git rev-parse HEAD)"
commit_short="$(git rev-parse --short=8 HEAD)"
campaign_name="${PDEOBS_FULL_CAMPAIGN:-numerics-full-t15-$commit_short}"
if [[ ! "$campaign_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "PDEOBS_FULL_CAMPAIGN contains unsupported characters: $campaign_name" >&2
  exit 2
fi

mkdir -p logs "$PDEOBS_DATA/plans"
plan="$PDEOBS_DATA/plans/$campaign_name.jsonl"
resolved="$PDEOBS_DATA/plans/$campaign_name.resolved.yaml"
output="$PDEOBS_DATA/$campaign_name"
campaign="$PDEOBS_DATA/$campaign_name.campaign.txt"

if [[ -e "$campaign" ]]; then
  echo "Campaign already submitted; refusing a duplicate: $campaign" >&2
  echo "Use hpc/slurm/monitor_full_t15.sh $campaign" >&2
  exit 2
fi
if [[ ! -e "$plan" ]]; then
  "$PDEOBS_ENV/bin/python" -m pdeobs plan \
    --config configs/dataset/numerics_full_t15.yaml \
    --output "$plan"
fi
test -s "$plan"
test -s "$resolved"
mkdir -p "$output"

read -r task_count sample_count temporal_jobs bad_stored_steps < <(
  "$PDEOBS_ENV/bin/python" - "$plan" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
temporal = [row for row in rows if row["pde"] in {"heat", "reaction_diffusion", "burgers", "navier_stokes"}]
bad = sum(int(row.get("stored_time_steps", -1)) != 15 for row in temporal)
print(len(rows), sum(int(row["sample_count"]) for row in rows), len(temporal), bad)
PY
)
if [[ "$task_count" != 3360 || "$sample_count" != 560000 ]]; then
  echo "Unexpected full plan: tasks=$task_count samples=$sample_count" >&2
  exit 2
fi
if [[ "$temporal_jobs" != 1920 || "$bad_stored_steps" != 0 ]]; then
  echo "Full plan does not have T=15 for every temporal job." >&2
  exit 2
fi

# One array element owns BUNDLE_SIZE independent shards. Its process pool uses
# one process per allocated CPU. Tune these defaults from a measured pilot and
# the limits published by the local Slurm administrator.
bundle_size="${PDEOBS_FULL_BUNDLE_SIZE:-40}"
cpus_per_task="${PDEOBS_FULL_CPUS_PER_TASK:-8}"
concurrency="${PDEOBS_FULL_CONCURRENCY:-4}"
memory="${PDEOBS_FULL_MEMORY:-32G}"
partition="${PDEOBS_CPU_PARTITION:-}"
time_limit="${PDEOBS_FULL_TIME_LIMIT:-1-00:00:00}"
max_queued="${PDEOBS_MAX_QUEUED_TASKS:-100}"

for pair in \
  "bundle_size:$bundle_size" \
  "cpus_per_task:$cpus_per_task" \
  "concurrency:$concurrency" \
  "max_queued:$max_queued"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$key must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( bundle_size < cpus_per_task )); then
  echo "bundle size must be at least cpus per task so allocated CPUs can be used." >&2
  exit 2
fi

array_count=$(( (task_count + bundle_size - 1) / bundle_size ))
array_max=$((array_count - 1))
if (( array_count + 1 > max_queued )); then
  echo "Array plus aggregate would exceed PDEOBS_MAX_QUEUED_TASKS=$max_queued." >&2
  echo "Increase the bundle size or set a site-approved queue limit." >&2
  exit 2
fi

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

generation_job="$("$sbatch_command" --parsable \
  "${site_args[@]}" \
  --job-name=pdeobs-full-t15 \
  --nodes=1 --ntasks=1 --cpus-per-task="$cpus_per_task" \
  --mem="$memory" --time="$time_limit" \
  --array="0-${array_max}%${concurrency}" \
  hpc/slurm/generate_array.sbatch \
  "$resolved" "$output" "$plan" "$bundle_size")"
generation_job="${generation_job%%;*}"

aggregate_job="$("$sbatch_command" --parsable \
  "${site_args[@]}" \
  --job-name=pdeobs-full-qc \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=08:00:00 \
  --dependency="afterok:$generation_job" \
  hpc/slurm/aggregate_cpu.sbatch \
  "$output" "$output/summary.json" "$plan" \
  --quality-strict --max-pde-loss 0.05 --require-all-pdes)"
aggregate_job="${aggregate_job%%;*}"

{
  echo "schema_version=1"
  echo "campaign=$campaign_name"
  echo "commit=$commit"
  echo "config=configs/dataset/numerics_full_t15.yaml"
  echo "plan=$plan"
  echo "resolved_config=$resolved"
  echo "output=$output"
  echo "task_count=$task_count"
  echo "sample_count=$sample_count"
  echo "stored_temporal_steps=15"
  echo "bundle_size=$bundle_size"
  echo "array_count=$array_count"
  echo "cpus_per_task=$cpus_per_task"
  echo "concurrency=$concurrency"
  echo "partition=${partition:-scheduler-default}"
  echo "time_limit=$time_limit"
  echo "generation_job=$generation_job"
  echo "aggregate_job=$aggregate_job"
  echo "publication_ready=false"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$campaign"

echo "Submitted full T=15 generation: $generation_job"
echo "Submitted dependent strict aggregate: $aggregate_job"
echo "Campaign record: $campaign"
echo "Monitor: bash hpc/slurm/monitor_full_t15.sh $campaign"
echo "No model-training job was submitted."
"$squeue_command" -j "$generation_job,$aggregate_job"
