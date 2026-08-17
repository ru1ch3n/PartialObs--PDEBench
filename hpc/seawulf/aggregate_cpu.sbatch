#!/usr/bin/env bash
#SBATCH --job-name=pdeobs-aggregate
#SBATCH --partition=short-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/seawulf/common.sh"

input_root="${1:?Usage: aggregate_cpu.sbatch INPUT_ROOT OUTPUT_JSON [EXPECTED_PLAN] [QUALITY_FLAGS...]}"
output_path="${2:?Usage: aggregate_cpu.sbatch INPUT_ROOT OUTPUT_JSON [EXPECTED_PLAN] [QUALITY_FLAGS...]}"
expected_plan="${3:-}"
if (( $# >= 3 )); then
  shift 3
else
  shift "$#"
fi
args=(aggregate --input "$input_root" --output "$output_path" --validate-shards)
if [[ -n "$expected_plan" ]]; then
  args+=(--expected-plan "$expected_plan")
fi
args+=("$@")
srun "$PDEOBS_ENV/bin/python" -m pdeobs "${args[@]}"
