#!/usr/bin/env bash
# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
#SBATCH --job-name=pdeobs-eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/seawulf/common.sh"

config_path="${1:?Usage: evaluate_gpu.sbatch CONFIG CHECKPOINT [extra arguments...]}"
checkpoint_path="${2:?Usage: evaluate_gpu.sbatch CONFIG CHECKPOINT [extra arguments...]}"
shift 2
srun "$PDEOBS_ENV/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"'
srun "$PDEOBS_ENV/bin/python" -m pdeobs eval \
  --config "$config_path" --checkpoint "$checkpoint_path" "$@"
