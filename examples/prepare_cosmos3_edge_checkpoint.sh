#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
: "${BASE_MODEL_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${OUTPUT_PATH:=$REPO_ROOT/examples/checkpoints/Cosmos3-Edge}"
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"

export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export LD_LIBRARY_PATH=""

[[ -x "$VENV_PYTHON" ]] || {
    echo "ERROR: existing Cosmos3 environment is unavailable: $REPO_ROOT/.venv" >&2
    exit 1
}
[[ -d "$BASE_MODEL_PATH" ]] || { echo "ERROR: local Cosmos3-Edge not found: $BASE_MODEL_PATH" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/modular_model_index.json" ]] || { echo "ERROR: missing modular_model_index.json" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/transformer/diffusion_pytorch_model.safetensors.index.json" ]] || { echo "ERROR: missing Edge transformer index" >&2; exit 1; }
[[ -f "$BASE_MODEL_PATH/tokenizer.json" ]] || { echo "ERROR: missing local Edge tokenizer" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: local Wan2.2 VAE not found: $WAN_VAE_PATH" >&2; exit 1; }
if [[ -e "$OUTPUT_PATH/model/.metadata" ]]; then
    BASE_CHECKPOINT_PATH="$OUTPUT_PATH" COSMOS3_EDGE_PATH="$BASE_MODEL_PATH" \
        PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" \
        -m cosmos_framework.scripts.validate_g1_edge_checkpoint
    echo "Reusing verified Cosmos3-Edge DCP: $OUTPUT_PATH"
    exit 0
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path "$BASE_MODEL_PATH" \
    --wan-vae-path "$WAN_VAE_PATH" \
    -o "$OUTPUT_PATH"

BASE_CHECKPOINT_PATH="$OUTPUT_PATH" COSMOS3_EDGE_PATH="$BASE_MODEL_PATH" \
    PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -m cosmos_framework.scripts.validate_g1_edge_checkpoint \
    --write-provenance
