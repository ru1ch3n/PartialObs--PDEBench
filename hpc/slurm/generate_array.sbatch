#!/usr/bin/env bash
#SBATCH --job-name=pdeobs-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -Eeuo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/slurm/common.sh"

config_path="${1:?Usage: generate_array.sbatch CONFIG OUTPUT [PLAN]}"
output_root="${2:?Usage: generate_array.sbatch CONFIG OUTPUT [PLAN]}"
plan_path="${3:-}"
bundle_size="${4:-1}"

if [[ ! "$bundle_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUNDLE_SIZE must be a positive integer, got: $bundle_size" >&2
  exit 2
fi

# The bundle parallelizes across independent processes. Prevent every process
# from creating another SLURM_CPUS_PER_TASK-sized BLAS/OpenMP pool.
if (( bundle_size > 1 )); then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

args=(
  generate --config "$config_path" --output "$output_root"
  --array-index "${SLURM_ARRAY_TASK_ID:-0}"
  --array-bundle-size "$bundle_size"
  --num-workers "${SLURM_CPUS_PER_TASK:-1}"
)
if [[ -n "$plan_path" ]]; then
  args+=(--plan "$plan_path")
fi

"${PDEOBS_SRUN:-srun}" "$PDEOBS_ENV/bin/python" -m pdeobs "${args[@]}"
