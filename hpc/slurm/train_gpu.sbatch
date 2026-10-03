#!/usr/bin/env bash
#SBATCH --job-name=pdeobs-train
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

config_path="${1:?Usage: train_gpu.sbatch CONFIG [extra pdeobs arguments...]}"
shift
"${PDEOBS_SRUN:-srun}" "$PDEOBS_ENV/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"'
"${PDEOBS_SRUN:-srun}" "$PDEOBS_ENV/bin/python" -m pdeobs train --config "$config_path" "$@"
