#!/usr/bin/env bash
# Prepares features.csv / endpoints.csv for Radiomics_Model_Development, using the shared Docker image.
# The docker command itself never needs to change -- just edit the paths below,
# then run: ./run_prepare_features.sh
set -euo pipefail

# On Windows Git Bash, MSYS/Cygwin auto-translates POSIX-looking paths (like
# /workdir/... or /data below) into Windows paths before Docker sees them,
# breaking this command. This disables that translation; harmless elsewhere
# (Linux, macOS, WSL).
export MSYS_NO_PATHCONV=1

# --- Edit these lines --------------------------------------------------------
DATA_DIR="/path/to/your/data"                        # local folder with your extracted features + raw outcomes
FEATURES_FILE="your_radiomic_features.csv"     # path relative to DATA_DIR
ENDPOINTS_FILE="your_raw_treatment_outcome.csv"      # path relative to DATA_DIR -- leave as "" to skip
# -----------------------------------------------------------------------------

ENDPOINTS_ARGS=()
if [ -n "${ENDPOINTS_FILE}" ]; then
  ENDPOINTS_ARGS=(--endpoints_path "/data/${ENDPOINTS_FILE}")
fi

docker run --rm \
  -v "${DATA_DIR}:/data" \
  -w /workdir/Radiomics_Model_Development \
  matto-radiomics \
  python -m matto_radiomics.data_preparation.prepare_features_and_endpoints \
    --features_path "/data/${FEATURES_FILE}" \
    "${ENDPOINTS_ARGS[@]}"
