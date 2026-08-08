#!/usr/bin/env bash
#SBATCH --job-name=oga_ref_rank
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

BASE=${OGA_BASE_EVIDENCE:?set OGA_BASE_EVIDENCE to the baseline seed-evidence CSV}
ANALYSIS=${OGA_COMBINED_ANALYSIS_ROOT:-${OGA_RUN_ROOT}/analysis_combined_wetlab}
shards=("${ANALYSIS}"/evidence_shards/new_evidence_*.csv)
[[ ${#shards[@]} -eq 24 ]] || { echo "expected 24 evidence shards" >&2; exit 2; }
evidence_args=(--evidence "${BASE}")
for shard in "${shards[@]}"; do evidence_args+=(--evidence "${shard}"); done
for threshold in 5 10; do
  out="${ANALYSIS}/target_RMSD_lt${threshold}"
  "${NACRAFT_PYTHON}" "${SCRIPT_DIR}/rank_multilength_af3.py" rank \
    "${evidence_args[@]}" --output-dir "${out}" \
    --plddt-threshold 0.5 --target-rmsd-threshold "${threshold}" \
    --top-pool 50 --select-count 20 --max-sequence-identity 0.90
done
