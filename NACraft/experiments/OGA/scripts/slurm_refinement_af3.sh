#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_af3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

MODE=${OGA_AF3_MODE:?set OGA_AF3_MODE=parent or child}
NUM_WORKERS=${OGA_AF3_NUM_WORKERS:?set OGA_AF3_NUM_WORKERS}
export AF3_PRESEARCH_DIR="${AF3_PRESEARCH_DIR:-${OGA_ASSET_ROOT}/presearch/out}"
: "${AF3_CODE_DIR:?set AF3_CODE_DIR}"
: "${AF3_MODEL_DIR:?set AF3_MODEL_DIR}"
if [[ -z "${AF3_SANDBOX_DIR:-}" && -z "${AF3_SIF_PATH:-}" ]]; then
  echo "set AF3_SANDBOX_DIR or AF3_SIF_PATH" >&2
  exit 2
fi
export AF3_SANDBOX_DIR="${AF3_SANDBOX_DIR:-${AF3_SIF_PATH}}"
export APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"

"${NACRAFT_PYTHON}" -u "${SCRIPT_DIR}/run_af3_validate.py" \
  --mode "${MODE}" \
  --parent-manifest "${OGA_RUN_ROOT}/manifests/parent_manifest.csv" \
  --child-root "${OGA_RUN_ROOT}/nampnn_children" \
  --target-manifest "${OGA_ASSET_ROOT}/targets/target_manifest.json" \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  --template-manifest "${OGA_ASSET_ROOT}/presearch/out/5vvo_only/template_manifest.json" \
  --output-root "${OGA_RUN_ROOT}/af3_${MODE}" \
  --worker-id "${SLURM_ARRAY_TASK_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --children-per-parent "${OGA_CHILDREN_PER_PARENT:-10}"
