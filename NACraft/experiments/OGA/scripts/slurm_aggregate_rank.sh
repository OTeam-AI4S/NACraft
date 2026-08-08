#!/usr/bin/env bash
#SBATCH --job-name=oga504_rank
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aggregate_rank_%j.out

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"${NACRAFT_PYTHON}" -u "${SCRIPT_DIR}/aggregate_and_rank.py" \
  --root "${OGA_RUN_ROOT}"
