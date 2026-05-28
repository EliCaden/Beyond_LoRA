#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.yml}"
TARGET_NAME="${2:-beyond_lora}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Load conda first, e.g. module load anaconda" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${TARGET_NAME}"; then
  echo "Environment ${TARGET_NAME} already exists; updating with --prune."
  conda env update -n "${TARGET_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

echo "Done. Activate with: conda activate ${TARGET_NAME}"
