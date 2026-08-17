#!/usr/bin/env bash
set -Eeuo pipefail

: "${PDEOBS_ENV:?Source your scratch env.sh first.}"
: "${PDEOBS_DATA:?Source your scratch env.sh first.}"
: "${PDEOBS_RUNS:?Source your scratch env.sh first.}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty checkout." >&2
  exit 2
fi

module load slurm
mkdir -p logs "$PDEOBS_DATA/plans" "$PDEOBS_DATA/numerics-validation20"

plan="$PDEOBS_DATA/plans/numerics-validation20.jsonl"
output="$PDEOBS_DATA/numerics-validation20"
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/numerics_validation20.yaml \
  --output "$plan"
resolved="$PDEOBS_DATA/plans/numerics-validation20.resolved.yaml"

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
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]] || (( concurrency > 40 )); then
  echo "PDEOBS_GENERATION_CONCURRENCY must be an integer from 1 through 40." >&2
  exit 2
fi

window_ids=()
previous=""
for start in 0 100 200 300 400 500 600 700 800; do
  stop=$((start + 99))
  (( stop >= task_count )) && stop=$((task_count - 1))
  dependency=()
  [[ -n "$previous" ]] && dependency=(--dependency="afterok:$previous")
  job="$(sbatch --parsable \
    --partition=short-40core-shared \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=04:00:00 \
    --array="${start}-${stop}%${concurrency}" \
    "${dependency[@]}" \
    hpc/seawulf/generate_array.sbatch "$resolved" "$output" "$plan")"
  job="${job%%;*}"
  window_ids+=("$job")
  previous="$job"
done

aggregate="$(sbatch --parsable \
  --partition=short-40core-shared \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=04:00:00 \
  --dependency="afterok:$previous" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$output" "$output/summary.json" "$plan" \
  --quality-strict --require-all-pdes)"
aggregate="${aggregate%%;*}"

campaign="$PDEOBS_DATA/numerics-validation20.campaign.txt"
{
  echo "commit=$(git rev-parse HEAD)"
  echo "plan=$plan"
  echo "resolved_config=$resolved"
  echo "output=$output"
  echo "task_count=$task_count"
  echo "sample_count=$sample_count"
  echo "concurrency=$concurrency"
  echo "window_jobs=${window_ids[*]}"
  echo "aggregate_job=$aggregate"
  echo "publication_ready=false"
} >"$campaign"

echo "Submitted the 5,600-sample numerical validation campaign."
echo "generation windows: ${window_ids[*]}"
echo "aggregate: $aggregate"
echo "campaign record: $campaign"
echo "No model-training job was submitted."
squeue -j "$(IFS=,; echo "${window_ids[*]}")",$aggregate

