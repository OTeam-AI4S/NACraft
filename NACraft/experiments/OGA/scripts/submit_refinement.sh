#!/usr/bin/env bash
# Submit the complete OGA refinement dependency graph.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/slurm_common.sh"

mkdir -p "${OGA_RUN_ROOT}/logs" "${OGA_RUN_ROOT}/manifests"
GROUP_MANIFEST="${OGA_RUN_ROOT}/group_manifest.json"
[[ -f "${GROUP_MANIFEST}" ]] || {
  echo "missing refinement group manifest: ${GROUP_MANIFEST}" >&2
  exit 2
}

design_ids=()
nampnn_ids=()
while IFS=$'\t' read -r group parent_count init_seq; do
  design_id=$(sbatch --parsable \
    --array="0-$((parent_count - 1))" \
    --output="${OGA_RUN_ROOT}/logs/design_${group}_%A_%a.out" \
    --export="ALL,OGA_RUN_ROOT=${OGA_RUN_ROOT},OGA_ASSET_ROOT=${OGA_ASSET_ROOT},OGA_GROUP=${group},OGA_PARENT_COUNT=${parent_count},OGA_INIT_SEQ=${init_seq}" \
    "${SCRIPT_DIR}/slurm_refinement_design.sh")
  nampnn_id=$(sbatch --parsable \
    --dependency="afterok:${design_id}" \
    --array="0-$((parent_count - 1))" \
    --output="${OGA_RUN_ROOT}/logs/nampnn_${group}_%A_%a.out" \
    --export="ALL,OGA_RUN_ROOT=${OGA_RUN_ROOT},OGA_ASSET_ROOT=${OGA_ASSET_ROOT},OGA_GROUP=${group}" \
    "${SCRIPT_DIR}/slurm_refinement_nampnn.sh")
  design_ids+=("${design_id}")
  nampnn_ids+=("${nampnn_id}")
  echo "group=${group} design=${design_id} nampnn=${nampnn_id}"
done < <(
  "${NACRAFT_PYTHON}" -c \
    'import json,sys; m=json.load(open(sys.argv[1])); [print(g["group"],g["parent_count"],g["init_seq"],sep="\t") for g in m["groups"]]' \
    "${GROUP_MANIFEST}"
)

design_dependency=$(IFS=:; echo "${design_ids[*]}")
nampnn_dependency=$(IFS=:; echo "${nampnn_ids[*]}")
manifest_id=$(sbatch --parsable \
  --dependency="afterok:${design_dependency}" \
  --output="${OGA_RUN_ROOT}/logs/manifest_%j.out" \
  --export="ALL,OGA_RUN_ROOT=${OGA_RUN_ROOT},OGA_ASSET_ROOT=${OGA_ASSET_ROOT}" \
  "${SCRIPT_DIR}/slurm_refinement_manifest.sh")
parent_af3_id=$(sbatch --parsable \
  --dependency="afterok:${manifest_id}" \
  --array=0-59 \
  --output="${OGA_RUN_ROOT}/logs/af3_parent_%A_%a.out" \
  --export="ALL,OGA_RUN_ROOT=${OGA_RUN_ROOT},OGA_ASSET_ROOT=${OGA_ASSET_ROOT},OGA_AF3_MODE=parent,OGA_AF3_NUM_WORKERS=60" \
  "${SCRIPT_DIR}/slurm_refinement_af3.sh")
child_af3_id=$(sbatch --parsable \
  --dependency="afterok:${manifest_id}:${nampnn_dependency}" \
  --array=0-599 \
  --output="${OGA_RUN_ROOT}/logs/af3_child_%A_%a.out" \
  --export="ALL,OGA_RUN_ROOT=${OGA_RUN_ROOT},OGA_ASSET_ROOT=${OGA_ASSET_ROOT},OGA_AF3_MODE=child,OGA_AF3_NUM_WORKERS=600" \
  "${SCRIPT_DIR}/slurm_refinement_af3.sh")

{
  echo "design=${design_ids[*]}"
  echo "nampnn=${nampnn_ids[*]}"
  echo "manifest=${manifest_id}"
  echo "af3_parent=${parent_af3_id}"
  echo "af3_child=${child_af3_id}"
} | tee "${OGA_RUN_ROOT}/submission_jobs.txt"
