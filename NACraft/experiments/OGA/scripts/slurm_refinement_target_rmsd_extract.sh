#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_rmsd
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

SHARDS=${OGA_RMSD_SHARDS:-24}
ANALYSIS=${OGA_COMBINED_ANALYSIS_ROOT:-${OGA_RUN_ROOT}/analysis_combined_wetlab}
OUT=${ANALYSIS}/evidence_shards
NATIVE_STRUCTURE=${OGA_NATIVE_STRUCTURE:-${OGA_ASSET_ROOT}/inputs/Human_O-GlcNAcase_5VVO.cif}
CANDIDATE_PREFIX=${OGA_CANDIDATE_PREFIX:-refinement__}
mkdir -p "${OUT}"
"${NACRAFT_PYTHON}" "${SCRIPT_DIR}/rank_multilength_af3.py" extract \
  --root "${OGA_RUN_ROOT}" --waves af3_parent af3_child \
  --native-structure "${NATIVE_STRUCTURE}" \
  --native-chains A B --candidate-prefix "${CANDIDATE_PREFIX}" \
  --num-shards "${SHARDS}" --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output "${OUT}/new_evidence_${SLURM_ARRAY_TASK_ID}.csv"
