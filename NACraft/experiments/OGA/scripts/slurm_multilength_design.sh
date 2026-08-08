#!/usr/bin/env bash
#SBATCH --job-name=oga_ml_design
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99-00:00:00
#SBATCH --output=logs/design_%x_%A_%a.out

set -euo pipefail

GROUP=${OGA_GROUP:?set OGA_GROUP}
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"
ROOT="${OGA_RUN_ROOT}"
CONFIG_ROOT=${OGA_CONFIG_ROOT:-${ROOT}/configs}
CONFIG=${CONFIG_ROOT}/${GROUP}.yaml
OUT=${ROOT}/designs/${GROUP}
NUM_DESIGNS=${OGA_NUM_DESIGNS:-50}
NUM_WORKERS=${OGA_NUM_WORKERS:-${NUM_DESIGNS}}

GPU_COUNT=${OGA_GPU_COUNT:-2}
if [[ "${GPU_COUNT}" == "5" ]]; then
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
elif [[ "${GPU_COUNT}" == "4" ]]; then
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
elif [[ "${GPU_COUNT}" == "3" ]]; then
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1,cuda:2
else
  export NACRAFT_BOLTZ_DEVICES=cuda:0,cuda:1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUT}" "${ROOT}/logs"

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

extra=()
if [[ "${GROUP}" == "a3_guided_41" ]]; then
  extra+=(--init-seq GGGCGACUGCUGAGUGACAUACCAUCUGGUUCGUCACGAAG)
fi
resume_args=(--skip_existing)
if [[ "${OGA_FORCE_REDESIGN:-0}" == "1" ]]; then
  resume_args=()
fi
early_args=()
if [[ "${OGA_EARLY_STOPPING:-1}" == "1" ]]; then
  early_args+=(--early_stopping)
fi

cd "${NACRAFT_DIR}"
"${NACRAFT_PYTHON}" -u main.py \
  --config "${CONFIG}" \
  --num_designs "${NUM_DESIGNS}" \
  --num_workers "${NUM_WORKERS}" \
  --worker_id "${SLURM_ARRAY_TASK_ID}" \
  "${resume_args[@]}" \
  "${early_args[@]}" \
  --ligandmpnn_seqs "${OGA_NAMPNN_SEQS:-0}" \
  --presearch-json "${OGA_ASSET_ROOT}/presearch/out/index.json" \
  --presearch-outdir "${OGA_ASSET_ROOT}/presearch/out" \
  -o "${OUT}" \
  "${extra[@]}"
