#!/usr/bin/env bash
#SBATCH --job-name=oga_2gpu_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"
SMOKE_GROUP=${OGA_SMOKE_GROUP:-denovo_20}
GPU_COUNT=${OGA_GPU_COUNT:-2}
SMOKE_STEPS=${OGA_SMOKE_STEPS:-3}
if [[ "${GPU_COUNT}" == "5" ]]; then
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
elif [[ "${GPU_COUNT}" == "4" ]]; then
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
elif [[ "${GPU_COUNT}" == "3" ]]; then
  DEVICES=cuda:0,cuda:1,cuda:2
else
  DEVICES=cuda:0,cuda:1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${ROOT}/logs" "${ROOT}/smoke"

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
echo "model_parallel_devices=${DEVICES}"

cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u experiments/OGA/scripts/run_multigpu_smoke.py \
  --config "${ROOT}/configs/${SMOKE_GROUP}.yaml" \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  --devices "${DEVICES}" \
  --steps "${SMOKE_STEPS}" \
  --report "${ROOT}/smoke/${SMOKE_GROUP}_${SLURM_JOB_NUM_NODES}node_${SLURM_JOB_ID}.json"
