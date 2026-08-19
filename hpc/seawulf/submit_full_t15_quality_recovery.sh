#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Set PDEOBS_ENV to the exact-commit environment.}"
: "${PDEOBS_DATA:?Set PDEOBS_DATA to the scratch data root.}"
: "${PDEOBS_RUNS:?Set PDEOBS_RUNS to the scratch run root.}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load slurm

repo="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo"
campaign="${1:-${PDEOBS_FULL_CAMPAIGN:-numerics-full-t15-6c7c7e31}}"
superseded_aggregate="${PDEOBS_SUPERSEDED_AGGREGATE_JOB:-2129030}"
if [[ ! "$campaign" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsupported campaign name: $campaign" >&2
  exit 2
fi
root="$PDEOBS_DATA/$campaign"
source_plan="$PDEOBS_DATA/plans/$campaign.jsonl"
resolved="$PDEOBS_DATA/plans/$campaign.resolved.yaml"
recovery_plan="$PDEOBS_DATA/plans/$campaign.quality-recovery.jsonl"
combined_plan="$PDEOBS_DATA/plans/$campaign.quality-recovered-combined.jsonl"
record="$PDEOBS_DATA/$campaign.quality-recovery.campaign.txt"
lock="$PDEOBS_DATA/$campaign.quality-recovery-submission-lock"
prepare="${PDEOBS_QUALITY_RECOVERY_PREPARE:-$repo/hpc/seawulf/prepare_quality_recovery.py}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
quarantine="$PDEOBS_DATA/quarantine/$campaign-quality-recovery-$stamp"
plan_summary="$PDEOBS_DATA/plans/$campaign.quality-recovery.summary.json"

if [[ -s "$record" ]]; then
  echo "Quality recovery was already submitted: $record" >&2
  cat "$record"
  exit 0
fi
if ! mkdir "$lock" 2>/dev/null; then
  echo "Quality recovery submission is already in progress: $lock" >&2
  exit 2
fi
submitted=()
finished=0
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 && finished == 0 && ${#submitted[@]} > 0 )); then
    echo "Recovery submission failed; canceling only newly submitted jobs: ${submitted[*]}" >&2
    scancel "${submitted[@]}" || true
  fi
  rmdir "$lock" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

test -s "$source_plan"
test -s "$resolved"
test -f "$prepare"
"$PDEOBS_ENV/bin/python" "$prepare" \
  --mode recovery \
  --plan "$source_plan" \
  --dataset-root "$root" \
  --output-plan "$recovery_plan" \
  --combined-plan "$combined_plan" \
  --refinement-factor 2 \
  --refine-all-temporal \
  --quarantine-dir "$quarantine" \
  | tee "$plan_summary"
cp "$plan_summary" "$quarantine/recovery-plan-summary.json"

read -r recovery_rows combined_rows combined_samples < <(
  "$PDEOBS_ENV/bin/python" - "$recovery_plan" "$combined_plan" <<'PY'
import json
import sys

recovery = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
combined = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
print(len(recovery), len(combined), sum(int(row["sample_count"]) for row in combined))
PY
)
if (( recovery_rows < 1 )); then
  echo "No incomplete shards remain; refusing an empty recovery submission." >&2
  exit 2
fi
if [[ "$combined_rows" != 3360 || "$combined_samples" != 560000 ]]; then
  echo "Combined recovery plan is not the frozen 3,360-shard/560,000-sample product." >&2
  exit 2
fi

# The original aggregate is permanently blocked by failed array elements.  It
# has never started and is replaced by the aggregate submitted below.
if [[ -n "$superseded_aggregate" ]]; then
  scancel "$superseded_aggregate" 2>/dev/null || true
fi

prefix="$PDEOBS_DATA/plans/$campaign.quality-recovery"
"$PDEOBS_ENV/bin/python" - "$recovery_plan" "$prefix" <<'PY'
import json
import os
import pathlib
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
names = ("long40-exclusive", "long40-shared", "long96-exclusive", "long96-shared")
for index, name in enumerate(names):
    target = pathlib.Path(f"{sys.argv[2]}.{name}.jsonl")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows[index::4]:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, target)
PY

submit_array() {
  local name="$1" partition="$2" cpus="$3" bundle="$4" concurrency="$5" plan="$6"
  local count arrays job
  count="$(awk 'NF {n++} END {print n + 0}' "$plan")"
  if (( count == 0 )); then
    return 0
  fi
  arrays=$(( (count + bundle - 1) / bundle ))
  job="$(sbatch --parsable \
    --job-name="$name" --partition="$partition" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem=52G --time=2-00:00:00 \
    --array="0-$((arrays - 1))%$concurrency" \
    hpc/seawulf/generate_array.sbatch "$resolved" "$root" "$plan" "$bundle")"
  printf '%s\n' "${job%%;*}"
}

job_40e="$(submit_array pdeobs-f15r-l40e long-40core 40 40 3 "$prefix.long40-exclusive.jsonl")"
job_40s="$(submit_array pdeobs-f15r-l40s long-40core-shared 40 40 6 "$prefix.long40-shared.jsonl")"
job_96e="$(submit_array pdeobs-f15r-l96e long-96core 96 96 3 "$prefix.long96-exclusive.jsonl")"
job_96s="$(submit_array pdeobs-f15r-l96s long-96core-shared 96 96 6 "$prefix.long96-shared.jsonl")"
generation_jobs=()
for job in "$job_40e" "$job_40s" "$job_96e" "$job_96s"; do
  if [[ -n "$job" ]]; then
    generation_jobs+=("$job")
    submitted+=("$job")
  fi
done
if (( ${#generation_jobs[@]} == 0 )); then
  echo "Recovery plan produced no generation arrays." >&2
  exit 2
fi
dependency="$(IFS=:; echo "${generation_jobs[*]}")"
aggregate="$(sbatch --parsable \
  --job-name=pdeobs-full-qc-recovered --partition=long-40core-shared \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=08:00:00 \
  --dependency="afterok:$dependency" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$root" "$root/summary.json" "$combined_plan" \
  --quality-strict --max-pde-loss 0.05 --require-all-pdes)"
aggregate="${aggregate%%;*}"
submitted+=("$aggregate")

{
  echo "schema_version=1"
  echo "base_campaign=$campaign"
  echo "source_plan=$source_plan"
  echo "recovery_plan=$recovery_plan"
  echo "combined_plan=$combined_plan"
  echo "recovery_rows=$recovery_rows"
  echo "combined_rows=$combined_rows"
  echo "combined_samples=$combined_samples"
  echo "refinement_policy=same-seed-double-internal-time-grid-output-T15"
  echo "quality_threshold=0.05"
  echo "quarantine=$quarantine"
  echo "generation_jobs=$(IFS=,; echo "${generation_jobs[*]}")"
  echo "aggregate_job=$aggregate"
  echo "superseded_aggregate_job=$superseded_aggregate"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_training_submitted=false"
} > "$record.tmp"
mv "$record.tmp" "$record"
finished=1

echo "Submitted strict same-seed T=15 quality recovery."
cat "$record"
squeue -j "$(IFS=,; echo "${submitted[*]}")"
