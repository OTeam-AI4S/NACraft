#!/usr/bin/env bash
#SBATCH --job-name=oga_nampnn_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u experiments/OGA/scripts/run_nampnn_pilot.py \
  --root "${ROOT}" \
  --group-index "${SLURM_ARRAY_TASK_ID}" \
  --num-seqs 3 \
  --overwrite
