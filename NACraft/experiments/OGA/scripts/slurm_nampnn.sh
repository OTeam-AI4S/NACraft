#!/usr/bin/env bash
#SBATCH --job-name=oga504_nampnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nampnn_%A_%a.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"

"${NACRAFT_PYTHON}" -u "${SCRIPT_DIR}/run_nampnn_redesign.py" \
  --manifest "${ROOT}/manifests/parent_manifest.csv" \
  --boltz-root "${ROOT}/boltz_parent" \
  --output-root "${ROOT}/nampnn_children" \
  --worker-id "${SLURM_ARRAY_TASK_ID}" \
  --num-workers 60
