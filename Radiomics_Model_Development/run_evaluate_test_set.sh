#!/usr/bin/env bash
# Evaluates an already-chosen feature subset, using the shared Docker image.
# The docker command itself never needs to change -- just edit the two paths
# below to point at your own project, then run: ./run_evaluate_test_set.sh
set -euo pipefail

# On Windows Git Bash, MSYS/Cygwin auto-translates POSIX-looking paths (like
# /workdir/... or /data/... below) into Windows paths before Docker sees them,
# breaking this command. This disables that translation; harmless elsewhere
# (Linux, macOS, WSL).
export MSYS_NO_PATHCONV=1

# --- Edit these two lines ---------------------------------------------------
PROJECT_DIR="/path/to/your/project"    # local folder with your data; mounted as /data in the container
CONFIG_FILE="your_config.json"         # path to your config file, relative to PROJECT_DIR
# -----------------------------------------------------------------------------

docker run --rm \
  -v "${PROJECT_DIR}:/data" \
  -w /workdir/Radiomics_Model_Development \
  --ulimit nofile=65536:65536 \
  matto-radiomics \
  python -m matto_radiomics.train_val_test.evaluate_feature_subset "/data/${CONFIG_FILE}"
