#!/usr/bin/env bash
# Runs radiomic feature extraction using the shared Docker image.
# The docker command itself never needs to change -- just edit the lines
# below to point at your own data, then run: ./run_extraction.sh
set -euo pipefail

# On Windows Git Bash, MSYS/Cygwin auto-translates POSIX-looking paths (like
# /workdir/... or /input, /output below) into Windows paths before Docker sees
# them, breaking this command. This disables that translation; harmless
# elsewhere (Linux, macOS, WSL).
export MSYS_NO_PATHCONV=1

# --- Edit these lines --------------------------------------------------------
INPUT_DIR="/path/to/your/input"      # local folder with input.csv, extraction_parameters.yaml, and your images/masks
OUTPUT_DIR="/path/to/your/output"    # local folder where results will be written
INPUT_CSV_NAME="input.csv"           # name of your CSV inside INPUT_DIR (change only if it isn't called input.csv)
# -----------------------------------------------------------------------------

docker run --rm \
  --user $(id -u):$(id -g) \
  -v "${INPUT_DIR}:/input" \
  -v "${OUTPUT_DIR}:/output" \
  matto-radiomics \
  python -m matto_radiomics.feature_extraction.batch_parallel_extraction_pyradiomics \
    --input_csv "/input/${INPUT_CSV_NAME}"
