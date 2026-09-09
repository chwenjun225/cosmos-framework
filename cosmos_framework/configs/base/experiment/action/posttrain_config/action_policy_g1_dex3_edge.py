# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge policy SFT recipe for Unitree G1 + dual Dex3 hands.

This adapts the native DROID action-policy recipe to the G1 embodiment and Edge
model while retaining its WAM loss, packing, transforms, DCP loading, and
checkpoint writing.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_g1_dex3_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

_G1_EDGE_MODEL_CONFIG = copy.deepcopy(EDGE_MODEL_CONFIG)
_G1_EDGE_MODEL_CONFIG["ema"]["enabled"] = False
_G1_EDGE_MODEL_CONFIG["parallelism"]["fsdp_master_dtype"] = "bfloat16"
_G1_EDGE_MODEL_CONFIG["parallelism"]["fsdp_reduce_dtype"] = "bfloat16"
_G1_EDGE_MODEL_CONFIG["tokenizer"]["encode_exact_durations"] = [33]
_G1_EDGE_MODEL_CONFIG["tokenizer"]["vae_path"] = "${oc.env:WAN_VAE_PATH}"
_G1_EDGE_MODEL_CONFIG["max_num_tokens_after_packing"] = -1
# Reuse processor/tokenizer/config assets from the already-downloaded Edge
# checkpoint. ``build_processor_lazy`` accepts this local directory directly.
_G1_EDGE_MODEL_CONFIG["vlm_config"]["tokenizer"]["repository"] = "${oc.env:COSMOS3_EDGE_PATH}"
_G1_EDGE_MODEL_CONFIG["vlm_config"]["pretrained_weights"]["backbone_path"] = "${oc.env:COSMOS3_EDGE_PATH}"

action_policy_g1_dex3_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_g1_dex3_edge["job"].update(
    project="cosmos3_action",
    group="g1_dex3_action_sft",
    name="action_policy_g1_dex3_edge",
    wandb_mode="disabled",
)
action_policy_g1_dex3_edge["model"]["config"] = _G1_EDGE_MODEL_CONFIG
# The native FusedAdam path keeps FP32 master weights and FP32 moments, which
# cannot fit the selected 1.42B action/generation parameters on a 24 GiB GPU.
# Cosmos3's native fused torch AdamW keeps the same AdamW hyperparameters while
# storing moments in the BF16 parameter dtype selected above.
action_policy_g1_dex3_edge["optimizer"]["optimizer_type"] = "AdamW"
action_policy_g1_dex3_edge["dataloader_train"]["dataset_name"] = "action_g1_dex3"
action_policy_g1_dex3_edge["dataloader_train"]["max_samples_per_batch"] = 1

rank_loader = action_policy_g1_dex3_edge["dataloader_train"]["dataloader"]
rank_loader["batch_size"] = 1
rank_loader["num_workers"] = 2
rank_loader["datasets"] = {
    "g1_dex3": {
        "ratio": 1,
        "dataset": L(get_action_g1_dex3_sft_dataset)(
            root="${oc.env:G1_DEX3_ROOT}",
            datasets="${oc.env:G1_DEX3_DATASETS}",
            fps=30.0,
            chunk_length=32,
            mode="wam",
            use_state=True,
            split="train",
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            use_image_augmentation=True,
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    }
}

# DROID's new-embodiment initialization is intentionally retained: the G1
# domain-specific projections start fresh while the generation backbone loads
# from the converted Edge DCP.
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_g1_dex3_edge",
    node=action_policy_g1_dex3_edge,
)
