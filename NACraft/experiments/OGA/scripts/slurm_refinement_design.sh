#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_design
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

GROUP=${OGA_GROUP:?set OGA_GROUP}
PARENT_COUNT=${OGA_PARENT_COUNT:?set OGA_PARENT_COUNT}
CONFIG="${OGA_RUN_ROOT}/configs/${GROUP}.yaml"
OUT="${OGA_RUN_ROOT}/designs/${GROUP}"

export NACRAFT_BOLTZ_DEVICES="${NACRAFT_BOLTZ_DEVICES:-cuda:0,cuda:1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUT}"

init_args=()
if [[ -n "${OGA_INIT_SEQ:-}" ]]; then
  init_args+=(--init-seq "${OGA_INIT_SEQ}")
fi

cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u main.py \
  --config "${CONFIG}" \
  --num_designs "${PARENT_COUNT}" \
  --num_workers "${PARENT_COUNT}" \
  --worker_id "${SLURM_ARRAY_TASK_ID}" \
  --skip_existing \
  --early_stopping \
  --ligandmpnn_seqs 0 \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  --outpath "${OUT}" \
  "${init_args[@]}"
