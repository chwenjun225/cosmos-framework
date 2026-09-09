#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR to the completed Cosmos3 G1 training run directory}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
: "${OUTPUT_PATH:=$RUN_DIR/Cosmos3-Edge-Policy-G1}"

[[ -x "$VENV_PYTHON" ]] || {
    echo "ERROR: existing Cosmos3 environment is unavailable: $REPO_ROOT/.venv" >&2
    exit 1
}
if [[ "$(realpath -m "$OUTPUT_PATH")" == "$(realpath -m "$REPO_ROOT/examples/checkpoints/Cosmos3-Edge")" ]]; then
    echo "ERROR: refusing to overwrite the base Cosmos3-Edge DCP" >&2
    exit 1
fi

LATEST_FILE="$RUN_DIR/checkpoints/latest_checkpoint.txt"
CONFIG_FILE="$RUN_DIR/config.yaml"
[[ -f "$LATEST_FILE" ]] || { echo "ERROR: missing $LATEST_FILE" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: missing $CONFIG_FILE" >&2; exit 1; }
CHECKPOINT_ITER="$(<"$LATEST_FILE")"
CHECKPOINT_PATH="$RUN_DIR/checkpoints/$CHECKPOINT_ITER"
[[ -d "$CHECKPOINT_PATH" ]] || {
    echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
}
[[ ! -e "$OUTPUT_PATH" ]] || { echo "ERROR: export output exists: $OUTPUT_PATH" >&2; exit 1; }

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --config-file "$CONFIG_FILE" \
    -o "$OUTPUT_PATH"
