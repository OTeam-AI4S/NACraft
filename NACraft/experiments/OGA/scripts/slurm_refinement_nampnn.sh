#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_nampnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

GROUP=${OGA_GROUP:?set OGA_GROUP}
cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u experiments/OGA/scripts/run_nampnn_pilot.py \
  --root "${OGA_RUN_ROOT}" \
  --group-name "${GROUP}" \
  --parent-index "${SLURM_ARRAY_TASK_ID}" \
  --num-seqs "${OGA_CHILDREN_PER_PARENT:-10}"
