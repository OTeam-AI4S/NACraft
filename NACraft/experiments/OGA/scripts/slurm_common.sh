#!/usr/bin/env bash
# Shared, cluster-independent paths for OGA Slurm jobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NACRAFT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OGA_EXPERIMENT_DIR="${NACRAFT_DIR}/experiments/OGA"

OGA_ROOT="${OGA_ROOT:-${OGA_EXPERIMENT_DIR}}"
OGA_RUN_ROOT="${OGA_RUN_ROOT:-${OGA_EXPERIMENT_DIR}/work}"
OGA_ASSET_ROOT="${OGA_ASSET_ROOT:-${OGA_ROOT}}"
NACRAFT_PYTHON="${NACRAFT_PYTHON:-python}"

export NACRAFT_DIR
export PYTHONPATH="${NACRAFT_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${OGA_RUN_ROOT}" "${OGA_RUN_ROOT}/logs"
