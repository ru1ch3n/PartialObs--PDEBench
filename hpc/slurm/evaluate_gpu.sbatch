#!/usr/bin/env bash
#SBATCH --job-name=pdeobs-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/slurm/common.sh"

config_path="${1:?Usage: evaluate_gpu.sbatch CONFIG CHECKPOINT [extra arguments...]}"
checkpoint_path="${2:?Usage: evaluate_gpu.sbatch CONFIG CHECKPOINT [extra arguments...]}"
shift 2
"${PDEOBS_SRUN:-srun}" "$PDEOBS_ENV/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"'
"${PDEOBS_SRUN:-srun}" "$PDEOBS_ENV/bin/python" -m pdeobs eval \
  --config "$config_path" --checkpoint "$checkpoint_path" "$@"
