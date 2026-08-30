#!/usr/bin/env bash
# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
set -Eeuo pipefail

: "${PDEOBS_DATA:?Source the scratch env.sh first.}"
: "${PDEOBS_ENV:?Source the scratch env.sh first.}"
module load slurm

campaign="${1:-}"
if [[ -z "$campaign" ]]; then
  campaign="$(find "$PDEOBS_DATA" -maxdepth 1 -name 'numerics-full-t15-*.campaign.txt' -type f -print | sort | tail -1)"
fi
if [[ -z "$campaign" || ! -f "$campaign" ]]; then
  echo "Usage: monitor_full_t15.sh CAMPAIGN_FILE" >&2
  exit 2
fi

value() {
  awk -F= -v key="$1" '$1 == key {value=$2} END {print value}' "$campaign"
}

generation_job="$(value generation_job)"
aggregate_job="$(value aggregate_job)"
output="$(value output)"
expected_shards="$(value task_count)"
expected_samples="$(value sample_count)"

echo "Campaign: $(value campaign)"
echo "Commit:   $(value commit)"
echo "Output:   $output"
echo "Jobs:     generation=$generation_job aggregate=$aggregate_job"
squeue -j "$generation_job,$aggregate_job" || true
echo
sacct -X -j "$generation_job,$aggregate_job" \
  --format=JobIDRaw,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode || true

read -r complete_shards complete_samples < <(
  "$PDEOBS_ENV/bin/python" - "$output" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
count = 0
samples = 0
for path in root.rglob("*.manifest.json") if root.exists() else ():
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "complete":
            count += 1
            samples += int(row.get("count", 0))
    except (OSError, ValueError, TypeError):
        continue
print(count, samples)
PY
)
size="$(du -sh "$output" 2>/dev/null | awk '{print $1}')"
echo
echo "Progress: $complete_shards/$expected_shards shards; $complete_samples/$expected_samples samples"
echo "Storage:  ${size:-0}"
if [[ -f "$output/summary.json" ]]; then
  echo "Strict summary is available: $output/summary.json"
fi
