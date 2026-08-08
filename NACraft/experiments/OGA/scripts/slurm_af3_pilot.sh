#!/usr/bin/env bash
#SBATCH --job-name=oga_af3_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
PILOT_ROOT="${OGA_RUN_ROOT}"
export AF3_PRESEARCH_DIR="${AF3_PRESEARCH_DIR:-${OGA_ASSET_ROOT}/presearch/out}"
: "${AF3_CODE_DIR:?set AF3_CODE_DIR}"
: "${AF3_MODEL_DIR:?set AF3_MODEL_DIR}"
: "${AF3_SANDBOX_DIR:=${AF3_SIF_PATH:-}}"
: "${AF3_SANDBOX_DIR:?set AF3_SANDBOX_DIR or AF3_SIF_PATH}"

"${NACRAFT_PYTHON}" -u "${SCRIPT_DIR}/run_af3_pilot.py" \
  --pilot-root "${PILOT_ROOT}" \
  --target-manifest "${OGA_ASSET_ROOT}/targets/target_manifest.json" \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  --template-manifest "${OGA_ASSET_ROOT}/presearch/out/5vvo_only/template_manifest.json" \
  --output-root "${PILOT_ROOT}/af3_validate" \
  --worker-id "${SLURM_ARRAY_TASK_ID}"
