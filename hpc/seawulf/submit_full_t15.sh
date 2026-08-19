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
submission_lock="$PDEOBS_DATA/$campaign_name.submission-lock"

if [[ -e "$campaign" ]]; then
  echo "Campaign already submitted; refusing a duplicate: $campaign" >&2
  echo "Use hpc/seawulf/monitor_full_t15.sh $campaign" >&2
  exit 2
fi
if ! mkdir "$submission_lock" 2>/dev/null; then
  echo "A full-campaign submission is already in progress: $submission_lock" >&2
  exit 2
fi
submitted_jobs=()
submission_succeeded=0
cleanup_submission() {
  status=$?
  trap - EXIT
  if (( status != 0 && submission_succeeded == 0 && ${#submitted_jobs[@]} > 0 )); then
    echo "Submission failed; canceling partially submitted jobs: ${submitted_jobs[*]}" >&2
    scancel "${submitted_jobs[@]}" || true
  fi
  rmdir "$submission_lock" 2>/dev/null || true
  exit "$status"
}
trap cleanup_submission EXIT
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

# Round-robin rows across the four standard CPU long queues. This preserves an
# even PDE/boundary/setting/regime mix and makes every output shard owned by
# exactly one array. GPU and HBM queues are intentionally excluded.
split_prefix="$PDEOBS_DATA/plans/$campaign_name"
"$PDEOBS_ENV/bin/python" - "$plan" "$split_prefix" <<'PY'
import json
import os
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
prefix = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 3360:
    raise SystemExit(f"expected 3360 plan rows, found {len(rows)}")

names = ("long40-exclusive", "long40-shared", "long96-exclusive", "long96-shared")
chunks = {name: rows[index::4] for index, name in enumerate(names)}
if any(len(chunk) != 840 for chunk in chunks.values()):
    raise SystemExit("four-way plan split is not exactly balanced")

outputs = [str(row.get("output_path", "")) for row in rows]
if len(outputs) != len(set(outputs)):
    raise SystemExit("full plan contains duplicate output paths")

for name, chunk in chunks.items():
    target = pathlib.Path(f"{prefix}.{name}.jsonl")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in chunk:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, target)
PY

plan_40e="$split_prefix.long40-exclusive.jsonl"
plan_40s="$split_prefix.long40-shared.jsonl"
plan_96e="$split_prefix.long96-exclusive.jsonl"
plan_96s="$split_prefix.long96-shared.jsonl"
for subplan in "$plan_40e" "$plan_40s" "$plan_96e" "$plan_96s"; do
  test "$(awk 'NF {count++} END {print count + 0}' "$subplan")" -eq 840
done

submit_array() {
  local job_name="$1"
  local partition="$2"
  local cpus="$3"
  local bundle="$4"
  local concurrency="$5"
  local subplan="$6"
  local count=840
  local array_count=$(( (count + bundle - 1) / bundle ))
  local array_max=$(( array_count - 1 ))
  local submitted

  submitted="$(sbatch --parsable \
    --job-name="$job_name" \
    --partition="$partition" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" \
    --mem=52G --time=2-00:00:00 \
    --array="0-${array_max}%${concurrency}" \
    hpc/seawulf/generate_array.sbatch \
    "$resolved" "$output" "$subplan" "$bundle")"
  printf '%s\n' "${submitted%%;*}"
}

# Exclusive queues fill whole nodes. Shared queues use the published six-node
# per-user QOS ceiling. The 96-core bundles contain 96 independent shard jobs,
# so they use the allocated CPU cores instead of reserving idle processes.
job_40e="$(submit_array pdeobs-f15-l40e long-40core 40 40 3 "$plan_40e")"
submitted_jobs+=("$job_40e")
job_40s="$(submit_array pdeobs-f15-l40s long-40core-shared 40 40 6 "$plan_40s")"
submitted_jobs+=("$job_40s")
job_96e="$(submit_array pdeobs-f15-l96e long-96core 96 96 3 "$plan_96e")"
submitted_jobs+=("$job_96e")
job_96s="$(submit_array pdeobs-f15-l96s long-96core-shared 96 96 6 "$plan_96s")"
submitted_jobs+=("$job_96s")

generation_jobs="$job_40e,$job_40s,$job_96e,$job_96s"
dependency_jobs="$job_40e:$job_40s:$job_96e:$job_96s"

aggregate_job="$(sbatch --parsable \
  --job-name=pdeobs-full-qc \
  --partition=long-40core-shared \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=08:00:00 \
  --dependency="afterok:$dependency_jobs" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$output" "$output/summary.json" "$plan" \
  --quality-strict --max-pde-loss 0.05 --require-all-pdes)"
aggregate_job="${aggregate_job%%;*}"
submitted_jobs+=("$aggregate_job")

campaign_temporary="$campaign.tmp"
{
  echo "schema_version=2"
  echo "campaign=$campaign_name"
  echo "commit=$commit"
  echo "config=configs/dataset/numerics_full_t15.yaml"
  echo "plan=$plan"
  echo "resolved_config=$resolved"
  echo "output=$output"
  echo "task_count=$task_count"
  echo "sample_count=$sample_count"
  echo "stored_temporal_steps=15"
  echo "partition_strategy=balanced_round_robin_all_standard_cpu_long"
  echo "partitions=long-40core,long-40core-shared,long-96core,long-96core-shared"
  echo "subplan_rows=840,840,840,840"
  echo "bundle_sizes=40,40,96,96"
  echo "array_counts=21,21,9,9"
  echo "cpus_per_task=40,40,96,96"
  echo "concurrency=3,6,3,6"
  echo "generation_job=$generation_jobs"
  echo "generation_jobs=$generation_jobs"
  echo "generation_job_long40_exclusive=$job_40e"
  echo "generation_job_long40_shared=$job_40s"
  echo "generation_job_long96_exclusive=$job_96e"
  echo "generation_job_long96_shared=$job_96s"
  echo "aggregate_job=$aggregate_job"
  echo "publication_ready=false"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_training_submitted=false"
} >"$campaign_temporary"
mv "$campaign_temporary" "$campaign"
submission_succeeded=1

echo "Submitted balanced full T=15 generation across four CPU long queues:"
echo "  long-40core:        $job_40e"
echo "  long-40core-shared: $job_40s"
echo "  long-96core:        $job_96e"
echo "  long-96core-shared: $job_96s"
echo "  dependent QC:       $aggregate_job"
echo "Campaign record: $campaign"
echo "Monitor: bash hpc/seawulf/monitor_full_t15.sh $campaign"
echo "No model-training job was submitted."
squeue -j "$generation_jobs,$aggregate_job"
