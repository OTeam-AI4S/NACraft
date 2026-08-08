#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_manifest
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

"${NACRAFT_PYTHON}" "${SCRIPT_DIR}/build_refinement_parent_manifest.py" \
  --root "${OGA_RUN_ROOT}" \
  --group-manifest "${OGA_RUN_ROOT}/group_manifest.json" \
  --output "${OGA_RUN_ROOT}/manifests/parent_manifest.csv"
