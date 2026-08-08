#!/usr/bin/env bash
#SBATCH --job-name=oga_post_es
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99-00:00:00
#SBATCH --output=logs/post_es_%x_%A_%a.out

set -euo pipefail

GROUP=${OGA_GROUP:?set OGA_GROUP}
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"
CONFIG=${ROOT}/configs/${GROUP}.yaml
OUT=${ROOT}/designs/${GROUP}

GPU_COUNT=${OGA_GPU_COUNT:-2}
if [[ "${GPU_COUNT}" == "5" ]]; then
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
elif [[ "${GPU_COUNT}" == "4" ]]; then
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
else
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u main.py \
  --config "${CONFIG}" \
  --num_designs 50 \
  --num_workers 50 \
  --worker_id "${SLURM_ARRAY_TASK_ID}" \
  --resume-oga-post-early-stop \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  -o "${OUT}"
