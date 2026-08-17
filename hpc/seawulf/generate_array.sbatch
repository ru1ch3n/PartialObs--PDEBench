#!/usr/bin/env bash
#SBATCH --job-name=pdeobs-generate
#SBATCH --partition=short-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -Eeuo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/seawulf/common.sh"

config_path="${1:?Usage: generate_array.sbatch CONFIG OUTPUT [PLAN]}"
output_root="${2:?Usage: generate_array.sbatch CONFIG OUTPUT [PLAN]}"
plan_path="${3:-}"

args=(generate --config "$config_path" --output "$output_root" --array-index "${SLURM_ARRAY_TASK_ID:-0}")
if [[ -n "$plan_path" ]]; then
  args+=(--plan "$plan_path")
fi

srun "$PDEOBS_ENV/bin/python" -m pdeobs "${args[@]}"
