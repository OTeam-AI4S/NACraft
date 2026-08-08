#!/usr/bin/env bash
#SBATCH --job-name=oga504_boltz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99-00:00:00
#SBATCH --output=logs/boltz_parent_%A_%a.out

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"

mkdir -p "${ROOT}/logs" "${ROOT}/boltz_parent"

"${NACRAFT_PYTHON}" -u "${SCRIPT_DIR}/run_parent_boltz.py" \
  --manifest "${ROOT}/manifests/parent_manifest.csv" \
  --target-manifest "${OGA_ASSET_ROOT}/targets/target_manifest.json" \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  --output-root "${ROOT}/boltz_parent" \
  --worker-id "${SLURM_ARRAY_TASK_ID}" \
  --num-workers "${OGA_NUM_WORKERS:-60}" \
  --samples 5
