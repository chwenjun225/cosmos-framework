#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR to the completed Cosmos3 G1 training run directory}"
: "${OUTPUT_PATH:=$RUN_DIR/model}"

LATEST_FILE="$RUN_DIR/checkpoints/latest_checkpoint.txt"
CONFIG_FILE="$RUN_DIR/config.yaml"
[[ -f "$LATEST_FILE" ]] || { echo "ERROR: missing $LATEST_FILE" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: missing $CONFIG_FILE" >&2; exit 1; }
CHECKPOINT_ITER="$(<"$LATEST_FILE")"
CHECKPOINT_PATH="$RUN_DIR/checkpoints/$CHECKPOINT_ITER"
[[ -d "$CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_PATH" ]] || { echo "ERROR: export output exists: $OUTPUT_PATH" >&2; exit 1; }

python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --config-file "$CONFIG_FILE" \
    -o "$OUTPUT_PATH"
