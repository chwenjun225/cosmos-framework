# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the resolved single-GPU G1 Edge memory profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-toml",
        type=Path,
        default=Path("examples/toml/sft_config/action_policy_g1_dex3_edge.toml"),
    )
    return parser.parse_args()


def _resolved_values(config: Any) -> dict[str, Any]:
    """Returns memory-critical values from the resolved native config."""

    model = config.model.config
    return {
        "ema.enabled": model.ema.enabled,
        "precision": model.precision,
        "fsdp_master_dtype": model.parallelism.fsdp_master_dtype,
        "fsdp_reduce_dtype": model.parallelism.fsdp_reduce_dtype,
        "activation_checkpointing.mode": model.activation_checkpointing.mode,
        "optimizer_type": config.optimizer.optimizer_type,
        "batch_size": config.dataloader_train.dataloader.batch_size,
        "max_samples_per_batch": config.dataloader_train.max_samples_per_batch,
        "compile_tokenizer.enabled": config.trainer.callbacks.compile_tokenizer.enabled,
    }


def main() -> None:
    args = _parse_args()
    config = load_experiment_from_toml(args.sft_toml)
    values = _resolved_values(config)
    expected = {
        "ema.enabled": False,
        "precision": "bfloat16",
        "fsdp_master_dtype": "bfloat16",
        "fsdp_reduce_dtype": "bfloat16",
        "activation_checkpointing.mode": "full",
        "optimizer_type": "AdamW",
        "batch_size": 1,
        "max_samples_per_batch": 1,
        "compile_tokenizer.enabled": False,
    }
    failures = [
        f"{key}={values[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if values[key] != value
    ]
    for key, value in values.items():
        print(f"{key} = {value}")
    if failures:
        raise SystemExit("Invalid G1 Edge memory profile:\n- " + "\n- ".join(failures))
    print("PASS G1 Edge RTX 4090 memory profile")


if __name__ == "__main__":
    main()
