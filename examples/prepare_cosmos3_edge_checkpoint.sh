#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BASE_MODEL_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Cosmos3-Edge}"
: "${OUTPUT_PATH:=$REPO_ROOT/examples/checkpoints/Cosmos3-Edge}"

[[ -d "$BASE_MODEL_PATH" ]] || { echo "ERROR: local Cosmos3-Edge not found: $BASE_MODEL_PATH" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/modular_model_index.json" ]] || { echo "ERROR: missing modular_model_index.json" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/transformer/diffusion_pytorch_model.safetensors.index.json" ]] || { echo "ERROR: missing Edge transformer index" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/tokenizer.json" ]] || { echo "ERROR: missing local Edge tokenizer" >&2; exit 1; }
if [[ -e "$OUTPUT_PATH/model/.metadata" ]]; then
    echo "ERROR: DCP already exists: $OUTPUT_PATH (choose OUTPUT_PATH to avoid overwriting it)" >&2
    exit 1
fi

cd "$REPO_ROOT"
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path "$BASE_MODEL_PATH" \
    -o "$OUTPUT_PATH"
