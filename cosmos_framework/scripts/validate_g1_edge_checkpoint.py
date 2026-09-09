# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the local Cosmos3-Edge DCP used for G1 post-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIBO_ROOT = _REPO_ROOT.parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "examples/checkpoints/Cosmos3-Edge"
_DEFAULT_EDGE_MODEL = _CIBO_ROOT / "hf_pretrained_models/Cosmos3-Edge"
_PROVENANCE_FILENAME = "g1_edge_base_provenance.json"
_DROID_MARKERS = ("policy-droid", "policy_droid", "edge-policy-droid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(os.environ.get("BASE_CHECKPOINT_PATH", _DEFAULT_CHECKPOINT)),
    )
    parser.add_argument(
        "--edge-model-path",
        type=Path,
        default=Path(os.environ.get("COSMOS3_EDGE_PATH", _DEFAULT_EDGE_MODEL)),
    )
    parser.add_argument(
        "--write-provenance",
        action="store_true",
        help="Write a verification receipt after successful DCP conversion.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")


def _provenance_values(checkpoint_path: Path, edge_model_path: Path) -> dict[str, Any]:
    source_index = edge_model_path / "transformer/diffusion_pytorch_model.safetensors.index.json"
    source_model_index = edge_model_path / "modular_model_index.json"
    dcp_metadata = checkpoint_path / "model/.metadata"
    dcp_config = checkpoint_path / "model/config.json"
    for path in (source_index, source_model_index, dcp_metadata, dcp_config):
        _require_file(path)
    return {
        "schema_version": 1,
        "source_model": "Cosmos3-Edge",
        "source_checkpoint": str(edge_model_path),
        "source_transformer_index_sha256": _sha256(source_index),
        "source_model_index_sha256": _sha256(source_model_index),
        "dcp_metadata_sha256": _sha256(dcp_metadata),
        "dcp_config_sha256": _sha256(dcp_config),
    }


def validate_checkpoint(
    checkpoint_path: Path,
    edge_model_path: Path,
    *,
    require_receipt: bool = True,
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    """Validates that the G1 warm start is the local Cosmos3-Edge base DCP."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    edge_model_path = edge_model_path.expanduser().resolve()
    canonical_checkpoint = _DEFAULT_CHECKPOINT.resolve()

    checkpoint_text = str(checkpoint_path).lower()
    if "droid" in checkpoint_text or any(marker in checkpoint_text for marker in _DROID_MARKERS):
        raise ValueError(
            "DROID policy checkpoints are forbidden for the G1 base warm start: "
            f"{checkpoint_path}"
        )
    if require_canonical_path and checkpoint_path != canonical_checkpoint:
        raise ValueError(
            "G1 post-training must warm-start from the canonical Cosmos3-Edge DCP: "
            f"expected {canonical_checkpoint}, got {checkpoint_path}"
        )

    expected = _provenance_values(checkpoint_path, edge_model_path)
    dcp_config = _load_json(checkpoint_path / "model/config.json")
    model_config = dcp_config.get("model", {}).get("config", {})
    if dcp_config.get("model_type") != "cosmos3_omni":
        raise ValueError("DCP config is not a Cosmos3 Omni model")
    if model_config.get("action_gen") is not True or model_config.get("max_action_dim") != 64:
        raise ValueError(
            "DCP config is not the Cosmos3-Edge action-capable base "
            "(action_gen=true, max_action_dim=64)"
        )
    tokenizer_repository = model_config.get("vlm_config", {}).get("tokenizer", {}).get("repository")
    if not tokenizer_repository or Path(tokenizer_repository).expanduser().resolve() != edge_model_path:
        raise ValueError(
            "DCP config does not point back to the selected local Cosmos3-Edge source: "
            f"repository={tokenizer_repository!r}, expected={str(edge_model_path)!r}"
        )

    receipt_path = checkpoint_path / _PROVENANCE_FILENAME
    if require_receipt:
        _require_file(receipt_path)
        receipt = _load_json(receipt_path)
        mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
        if mismatches:
            raise ValueError(f"Cosmos3-Edge provenance receipt mismatch for: {', '.join(mismatches)}")
    return expected


def main() -> None:
    args = _parse_args()
    try:
        values = validate_checkpoint(
            args.checkpoint_path,
            args.edge_model_path,
            require_receipt=not args.write_provenance,
            require_canonical_path=not args.write_provenance,
        )
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if args.write_provenance:
        receipt_path = checkpoint_path / _PROVENANCE_FILENAME
        receipt_path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
        print(f"Wrote Cosmos3-Edge provenance receipt: {receipt_path}")
    print(f"PASS G1 base checkpoint: {checkpoint_path}")
    print(f"source Cosmos3-Edge: {args.edge_model_path.expanduser().resolve()}")


if __name__ == "__main__":
    main()
