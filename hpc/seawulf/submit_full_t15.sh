#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Source the scratch env.sh first.}"
: "${PDEOBS_DATA:?Source the scratch env.sh first.}"
: "${PDEOBS_RUNS:?Source the scratch env.sh first.}"

module load slurm
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
  echo "Use hpc/seawulf/monitor_full_t15.sh $campaign" >&2
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

# One array element owns BUNDLE_SIZE independent shards. Its ProcessPool uses
# one process per allocated CPU. Forty CPUs map naturally to a 40-core node;
# at most six elements run together, matching the current shared-long QOS cap
# of 240 CPUs while staying below its 322-GiB user-memory cap.
bundle_size="${PDEOBS_FULL_BUNDLE_SIZE:-40}"
cpus_per_task="${PDEOBS_FULL_CPUS_PER_TASK:-40}"
concurrency="${PDEOBS_FULL_CONCURRENCY:-6}"
memory="${PDEOBS_FULL_MEMORY:-52G}"
partition="${PDEOBS_FULL_PARTITION:-long-40core-shared}"
time_limit="${PDEOBS_FULL_TIME_LIMIT:-2-00:00:00}"

for pair in \
  "bundle_size:$bundle_size" \
  "cpus_per_task:$cpus_per_task" \
  "concurrency:$concurrency"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$key must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( cpus_per_task > 40 || concurrency > 6 )); then
  echo "Refusing to exceed 40 CPUs per node or six concurrent full workers." >&2
  exit 2
fi
if (( bundle_size < cpus_per_task )); then
  echo "bundle size must be at least cpus per task so allocated CPUs can be used." >&2
  exit 2
fi

array_count=$(( (task_count + bundle_size - 1) / bundle_size ))
array_max=$((array_count - 1))
if (( array_count + 1 > 100 )); then
  echo "Array plus aggregate would exceed SeaWulf MaxSubmitJobsPU=100." >&2
  exit 2
fi

generation_job="$(sbatch --parsable \
  --job-name=pdeobs-full-t15 \
  --partition="$partition" \
  --nodes=1 --ntasks=1 --cpus-per-task="$cpus_per_task" \
  --mem="$memory" --time="$time_limit" \
  --array="0-${array_max}%${concurrency}" \
  hpc/seawulf/generate_array.sbatch \
  "$resolved" "$output" "$plan" "$bundle_size")"
generation_job="${generation_job%%;*}"

aggregate_job="$(sbatch --parsable \
  --job-name=pdeobs-full-qc \
  --partition="$partition" \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=08:00:00 \
  --dependency="afterok:$generation_job" \
  hpc/seawulf/aggregate_cpu.sbatch \
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
  echo "partition=$partition"
  echo "time_limit=$time_limit"
  echo "generation_job=$generation_job"
  echo "aggregate_job=$aggregate_job"
  echo "publication_ready=false"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$campaign"

echo "Submitted full T=15 generation: $generation_job"
echo "Submitted dependent strict aggregate: $aggregate_job"
echo "Campaign record: $campaign"
echo "Monitor: bash hpc/seawulf/monitor_full_t15.sh $campaign"
echo "No model-training job was submitted."
squeue -j "$generation_job,$aggregate_job"
