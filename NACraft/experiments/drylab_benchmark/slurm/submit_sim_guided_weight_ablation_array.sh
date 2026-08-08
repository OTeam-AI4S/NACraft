#!/usr/bin/env bash
#SBATCH --job-name=nacraft_simw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-120
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_ROOT="${NACRAFT_BENCHMARK_ROOT:-data/drylab_benchmark}"
COMMAND_FILE="${DATA_ROOT}/configs/nacraft_sim_guided_weight_ablation/nacraft_design_commands_sim_guided_weight_ablation.sh"

cd "${REPO_ROOT}"
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${COMMAND_FILE}")
if [[ -z "${COMMAND}" ]]; then
  echo "No command for array index ${SLURM_ARRAY_TASK_ID} in ${COMMAND_FILE}" >&2
  exit 2
fi
echo "[drylab sim-guided weight ablation] ${COMMAND}"
eval "${COMMAND}"
