#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATASET_PATH:=/home/tuan/projects/cibo/datasets}"
: "${G1_DEX3_DATASETS:=all}"
: "${COSMOS3_EDGE_PATH:=/home/tuan/projects/cibo/hf_pretrained_models/Cosmos3-Edge}"
: "${HF_HOME:=$REPO_ROOT/.cache/huggingface}"

export HF_HOME
cd "$REPO_ROOT"
PYTHONPATH=. python -m cosmos_framework.scripts.validate_g1_dex3 \
    --dataset-root "$DATASET_PATH" \
    --datasets "$G1_DEX3_DATASETS" \
    --edge-checkpoint "$COSMOS3_EDGE_PATH" \
    "$@"
