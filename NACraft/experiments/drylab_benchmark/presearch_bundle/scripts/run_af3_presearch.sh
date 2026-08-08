#!/usr/bin/env bash
set -euo pipefail

# Portable AF3 data-pipeline-only presearch runner.
#
# Required environment variables on the target cluster:
#   AF3_SIF        AlphaFold3 Apptainer/Singularity image
#   AF3_MODEL_DIR  AF3 model directory mounted to /root/models
#   AF3_DB_DIR     AF3 database directory mounted to /root/public_databases
#   AF3_CODE_DIR   official AF3 source or a compatible legacy launcher
# Optional:
#   APPTAINER      apptainer/singularity executable (default: apptainer)
#   WORK_ROOT      bundle root (default: parent of this script)
#   NUM_WORKERS    number of AF3 data-pipeline workers (default: 4)
#   EXP_NAME       AF3 output experiment name (default: presearch)
#   LAUNCH_PATCHED patched run_af3_multiprocess.py that preserves *_data.json

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${BUNDLE_ROOT}}"
INPUT_JSON="${INPUT_JSON:-${WORK_ROOT}/input/presearch_input.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_ROOT}/raw}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
LAUNCH_TMP="${LAUNCH_TMP:-${WORK_ROOT}/launch_tmp}"
LAUNCH_LOG="${LAUNCH_LOG:-${WORK_ROOT}/launch_log}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXP_NAME="${EXP_NAME:-presearch}"
APPTAINER="${APPTAINER:-apptainer}"

: "${AF3_SIF:?set AF3_SIF to the AF3 container image}"
: "${AF3_MODEL_DIR:?set AF3_MODEL_DIR to the AF3 model directory}"
: "${AF3_DB_DIR:?set AF3_DB_DIR to the AF3 database directory}"
: "${AF3_CODE_DIR:?set AF3_CODE_DIR to the AF3 code directory}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${LAUNCH_TMP}" "${LAUNCH_LOG}"

echo "=== AF3 presearch ==="
echo "date:        $(date)"
echo "host:        $(hostname)"
echo "bundle:      ${WORK_ROOT}"
echo "input:       ${INPUT_JSON}"
echo "raw output:  ${OUTPUT_DIR}"
echo "workers:     ${NUM_WORKERS}"
echo "AF3_SIF:     ${AF3_SIF}"
echo "AF3_DB_DIR:  ${AF3_DB_DIR}"
echo "AF3_CODE_DIR:${AF3_CODE_DIR}"
echo

if [[ -f "${AF3_CODE_DIR}/run_alphafold.py" ]]; then
  python "${BUNDLE_ROOT}/scripts/run_official_presearch.py" \
    --input-json "${INPUT_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --task-dir "${LAUNCH_TMP}/tasks" \
    --image "${AF3_SIF}" \
    --model-dir "${AF3_MODEL_DIR}" \
    --db-dir "${AF3_DB_DIR}" \
    --code-dir "${AF3_CODE_DIR}" \
    --apptainer "${APPTAINER}" \
    --workers "${NUM_WORKERS}"
else
  binds=(
    -B "${AF3_MODEL_DIR}:/root/models"
    -B "${AF3_DB_DIR}:/root/public_databases"
    -B "${INPUT_JSON}:${INPUT_JSON}"
    -B "${OUTPUT_DIR}:${OUTPUT_DIR}"
    -B "${AF3_CODE_DIR}:/app/alphafold"
    -B "${LAUNCH_TMP}:/app/alphafold/tmp"
    -B "${LAUNCH_LOG}:/app/alphafold/log"
  )
  if [[ -n "${LAUNCH_PATCHED:-}" ]]; then
    binds+=(-B "${LAUNCH_PATCHED}:/app/alphafold/run_af3_multiprocess.py")
  fi
  "${APPTAINER}" exec --nv --writable-tmpfs \
    "${binds[@]}" \
    "${AF3_SIF}" \
    python /app/alphafold/launch.py \
      --input_json "${INPUT_JSON}" \
      --output_dir "${OUTPUT_DIR}" \
      --run_data_pipeline True \
      --run_inference False \
      --num_workers "${NUM_WORKERS}" \
      --exp_name "${EXP_NAME}" \
      --num_diffusion_samples 1
fi

echo
echo "=== AF3 presearch finished at $(date) ==="
echo "Next: bash scripts/extract_presearch.sh"
