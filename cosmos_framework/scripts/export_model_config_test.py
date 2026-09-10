# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression tests for Edge action-policy export config loading."""

from pathlib import Path

import yaml
from omegaconf import DictConfig

from cosmos_framework.inference.common.args import ConfigArgs, ConfigFileType
from cosmos_framework.scripts.export_model import (
    _build_edge_policy_metadata,
    _load_edge_policy_metadata_config,
)
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string

_G1_FACTORY = "cosmos_framework.data.generator.action.datasets.action_sft_dataset.get_action_g1_dex3_sft_dataset"
_EXPECTED_POLICY_METADATA = {
    "action_chunk_size": 32,
    "conditioning_fps": 30.0,
    "domain_name": "unitree_g1_dex3",
}


def _training_config_dict() -> dict:
    return {
        "model": {
            "_target_": "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
            "config": {"action_gen": True, "ema": {"enabled": False}},
        },
        "dataloader_train": {
            "dataloader": {
                "datasets": {
                    "g1_dex3": {
                        "ratio": 1,
                        "dataset": {
                            "_target_": _G1_FACTORY,
                            "chunk_length": 32,
                            "fps": 30.0,
                        },
                    }
                }
            }
        },
    }


def _config_args(config_path: Path) -> ConfigArgs:
    return ConfigArgs(
        config_file=str(config_path),
        config_file_type=ConfigFileType.from_path(str(config_path)),
        experiment="",
        experiment_overrides=[],
    )


def test_edge_policy_metadata_loads_training_yaml_without_root_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_training_config_dict()), encoding="utf-8")
    config_args = _config_args(config_path)

    model_dict = config_args.load_model_config_dict()
    metadata_config = _load_edge_policy_metadata_config(config_args)

    assert model_dict["config"]["action_gen"] is True
    assert isinstance(metadata_config, DictConfig)
    assert metadata_config.get("_type") is None
    assert _build_edge_policy_metadata(metadata_config) == _EXPECTED_POLICY_METADATA


def test_edge_policy_metadata_keeps_structured_inference_yaml_path(tmp_path: Path) -> None:
    config_dict = _training_config_dict()
    config_dict.update(
        {
            "_type": convert_target_to_string(Config),
            "optimizer": {},
            "scheduler": {},
            "dataloader_val": None,
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    config_args = _config_args(config_path)

    metadata_config = _load_edge_policy_metadata_config(config_args)

    assert isinstance(metadata_config, Config)
    assert _build_edge_policy_metadata(metadata_config) == _EXPECTED_POLICY_METADATA
