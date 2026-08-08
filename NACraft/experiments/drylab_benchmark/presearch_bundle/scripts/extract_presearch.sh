#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_OUT="${RAW_OUT:-${BUNDLE_ROOT}/raw}"
OUT_DIR="${OUT_DIR:-${BUNDLE_ROOT}/out}"

if [[ ! -d "${RAW_OUT}" ]]; then
  echo "[ERR] AF3 raw output not found: ${RAW_OUT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
python "${BUNDLE_ROOT}/scripts/extract_presearch.py" \
  --root "${RAW_OUT}" \
  --presearch-outdir "${OUT_DIR}"

echo
echo "Generated:"
echo "  ${OUT_DIR}/index.json"
