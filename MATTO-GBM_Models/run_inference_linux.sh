#!/usr/bin/env bash
# Runs MATTO-GBM pretrained model inference from a JSON config.
set -euo pipefail

# --- Edit these four lines --------------------------------------------------
INPUT_DIR="/path/to/your/input"
OUTPUT_DIR="/path/to/your/output"
MODELS_DIR="/path/to/your/models"
CONFIG_FILE="inference_config.json"
# ----------------------------------------------------------------------------

docker run --rm \
  -v "${INPUT_DIR}:/input" \
  -v "${OUTPUT_DIR}:/output" \
  -v "${MODELS_DIR}:/models" \
  matto-radiomics \
  sh -c "umask 000 && python MATTO-GBM_Models/code/run_inference.py '/input/${CONFIG_FILE}'"

