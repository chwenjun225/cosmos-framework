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
: "${NPROC_PER_NODE:=1}"

[[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$REPO_ROOT/$DATASET_PATH"
[[ "$COSMOS3_EDGE_PATH" = /* ]] || COSMOS3_EDGE_PATH="$REPO_ROOT/$COSMOS3_EDGE_PATH"
[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$REPO_ROOT/$WAN_VAE_PATH"

export G1_DEX3_ROOT="$DATASET_PATH"
export G1_DEX3_DATASETS COSMOS3_EDGE_PATH WAN_VAE_PATH HF_HOME NPROC_PER_NODE PYTORCH_ALLOC_CONF

# Keep host ROS/Isaac/CUDA libraries from overriding the CUDA 13.0 libraries
# bundled with this venv. In particular, a CUDA 13.3 host path makes even a
# small torch BF16/FP32 Linear fail with CUBLAS_STATUS_NOT_INITIALIZED.
export LD_LIBRARY_PATH=""

VENV_BIN="$REPO_ROOT/.venv/bin"
[[ -x "$VENV_BIN/python" && -x "$VENV_BIN/torchrun" ]] || {
    echo "ERROR: existing Cosmos3 environment is unavailable: $REPO_ROOT/.venv" >&2
    exit 1
}
export PATH="$VENV_BIN:$PATH"

# This metadata-only validation walks every episode reference, so a missing
# Parquet or video shard fails before torchrun starts.
g1_local_preflight() {
    PYTHONPATH="$WORKDIR" python -m cosmos_framework.scripts.validate_g1_dex3 \
        --dataset-root "$G1_DEX3_ROOT" \
        --datasets "$G1_DEX3_DATASETS" \
        --edge-checkpoint "$COSMOS3_EDGE_PATH" \
        --metadata-only || return
    PYTHONPATH="$WORKDIR" python -m cosmos_framework.scripts.validate_g1_edge_checkpoint \
        --checkpoint-path "$BASE_CHECKPOINT_PATH" \
        --edge-model-path "$COSMOS3_EDGE_PATH" || return
    PYTHONPATH="$WORKDIR" python -m cosmos_framework.scripts.validate_g1_training_profile \
        --sft-toml "$TOML_FILE" || return
    python -c 'import os, torch; expected=int(os.environ["NPROC_PER_NODE"]); available=torch.cuda.device_count(); assert torch.cuda.is_available(), "CUDA is not available"; assert expected == 1, f"RTX 4090 profile requires NPROC_PER_NODE=1, got {expected}"; assert available >= expected, f"requested {expected} GPU but only {available} visible"; print(f"PASS CUDA: torch={torch.__version__}, runtime={torch.version.cuda}, devices={available}, gpu={torch.cuda.get_device_name(0)}")' || return
}
EXTRA_DATASET_CHECK=g1_local_preflight

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
