#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Native Cosmos3-Edge action-policy SFT for one, several, or all local G1 Dex3
# LeRobot v3 datasets. Run from the Cosmos3 repository with its .venv active.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOML_FILE="examples/toml/sft_config/action_policy_g1_dex3_edge.toml"
: "${DATASET_PATH:=/home/tuan/projects/cibo/datasets}"
: "${G1_DEX3_DATASETS:=all}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${COSMOS3_EDGE_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${HF_HOME:=$REPO_ROOT/.cache/huggingface}"
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"

[[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$REPO_ROOT/$DATASET_PATH"
[[ "$COSMOS3_EDGE_PATH" = /* ]] || COSMOS3_EDGE_PATH="$REPO_ROOT/$COSMOS3_EDGE_PATH"
[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$REPO_ROOT/$WAN_VAE_PATH"

export G1_DEX3_ROOT="$DATASET_PATH"
export G1_DEX3_DATASETS COSMOS3_EDGE_PATH WAN_VAE_PATH HF_HOME PYTORCH_ALLOC_CONF

# Keep host ROS/Isaac/CUDA libraries from overriding the CUDA 13.0 libraries
# bundled with this venv. In particular, a CUDA 13.3 host path makes even a
# small torch BF16/FP32 Linear fail with CUBLAS_STATUS_NOT_INITIALIZED.
export LD_LIBRARY_PATH=""

# This metadata-only validation walks every episode reference, so a missing
# Parquet or video shard fails before torchrun starts.
EXTRA_DATASET_CHECK='[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || { echo "ERROR: missing converted DCP metadata: $BASE_CHECKPOINT_PATH/model/.metadata (run examples/prepare_cosmos3_edge_checkpoint.sh)" >&2; exit 1; }; PYTHONPATH="$WORKDIR" python -m cosmos_framework.scripts.validate_g1_dex3 --dataset-root "$G1_DEX3_ROOT" --datasets "$G1_DEX3_DATASETS" --edge-checkpoint "$COSMOS3_EDGE_PATH" --metadata-only; PYTHONPATH="$WORKDIR" python -m cosmos_framework.scripts.validate_g1_training_profile --sft-toml "$TOML_FILE"'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
