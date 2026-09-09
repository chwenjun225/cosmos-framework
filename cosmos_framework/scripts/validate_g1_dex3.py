# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate local G1 Dex3 data and build real Cosmos3 action SFT samples."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_g1_dex3_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.g1_dex3_lerobot_dataset import (
    G1_DEX3_ACTION_DIM,
    G1_DEX3_ACTION_REPRESENTATION,
    G1_DEX3_ACTION_UNITS,
    G1_DEX3_DOMAIN_ID,
    G1_DEX3_TASK_CORRECTIONS,
    _read_and_validate_info,
    resolve_g1_dex3_roots,
    validate_g1_dex3_file_closure,
)
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides
from cosmos_framework.utils.lazy_config import LazyCall as L


def _expected_video_files(
    root: Path,
    meta: LeRobotDatasetMetadata,
    camera_keys: tuple[str, ...],
) -> set[Path]:
    template = meta.info["video_path"]
    paths: set[Path] = set()
    for episode in meta.episodes:
        for camera_key in camera_keys:
            prefix = f"videos/{camera_key}"
            paths.add(
                root
                / template.format(
                    video_key=camera_key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]),
                    file_index=int(episode[f"{prefix}/file_index"]),
                )
            )
    return paths


def _scan_parquet(root: Path, expected_frames: int, fps: float) -> None:
    import pyarrow.parquet as pq

    required = {
        "action",
        "episode_index",
        "frame_index",
        "index",
        "observation.state",
        "task_index",
        "timestamp",
    }
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{root}: no data Parquet shards")

    rows = 0
    previous_index: int | None = None
    previous_episode: int | None = None
    previous_frame: int | None = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows += parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=65536, columns=sorted(required)):
            columns = {name: batch.column(name) for name in batch.schema.names}
            state = np.asarray(columns["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(columns["action"].to_pylist(), dtype=np.float32)
            if state.ndim != 2 or state.shape[1] != G1_DEX3_ACTION_DIM:
                raise ValueError(f"{path}: state batch shape is {state.shape}, expected (*, 28)")
            if action.ndim != 2 or action.shape[1] != G1_DEX3_ACTION_DIM:
                raise ValueError(f"{path}: action batch shape is {action.shape}, expected (*, 28)")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{path}: state/action contains NaN or Inf")

            indexes = np.asarray(columns["index"].to_numpy(), dtype=np.int64)
            episodes = np.asarray(columns["episode_index"].to_numpy(), dtype=np.int64)
            frames = np.asarray(columns["frame_index"].to_numpy(), dtype=np.int64)
            timestamps = np.asarray(columns["timestamp"].to_numpy(), dtype=np.float64)
            task_indexes = np.asarray(columns["task_index"].to_numpy(), dtype=np.int64)
            if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
                raise ValueError(f"{path}: invalid timestamps")
            if np.any(task_indexes < 0):
                raise ValueError(f"{path}: negative task index")

            if indexes.size:
                start = 0 if previous_index is None else previous_index + 1
                if not np.array_equal(indexes, np.arange(start, start + indexes.size)):
                    raise ValueError(f"{path}: global index is not contiguous")
                previous_index = int(indexes[-1])

                all_episodes = episodes
                all_frames = frames
                if previous_episode is not None and previous_frame is not None:
                    all_episodes = np.concatenate(([previous_episode], episodes))
                    all_frames = np.concatenate(([previous_frame], frames))
                episode_delta = np.diff(all_episodes)
                frame_delta = np.diff(all_frames)
                if np.any(episode_delta < 0):
                    raise ValueError(f"{path}: episode index is not monotonic")
                if np.any(frame_delta[episode_delta == 0] != 1):
                    raise ValueError(f"{path}: frame index is not contiguous within episodes")
                if np.any(all_frames[1:][episode_delta > 0] != 0):
                    raise ValueError(f"{path}: frame index does not reset at episode boundaries")
                previous_episode = int(episodes[-1])
                previous_frame = int(frames[-1])

            expected_timestamps = frames.astype(np.float64) / fps
            if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=5e-4):
                raise ValueError(f"{path}: timestamps do not match frame_index / {fps:g} Hz")
    if rows != expected_frames:
        raise ValueError(f"{root}: Parquet rows {rows} != metadata total_frames {expected_frames}")


def _decode_videos(paths: set[Path], expected_fps: float) -> None:
    import av

    for path in sorted(paths):
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError("no video stream")
                stream = container.streams.video[0]
                actual_fps = float(stream.average_rate) if stream.average_rate else 0.0
                if not math.isclose(actual_fps, expected_fps, abs_tol=1e-3):
                    raise ValueError(f"FPS {actual_fps:g} != {expected_fps:g}")
                if next(container.decode(video=0), None) is None:
                    raise ValueError("no decodable frame")
        except Exception as error:
            raise ValueError(f"{path}: video decode failed: {error}") from error


def _validate_local_edge_checkpoint(path: Path) -> dict[str, Any]:
    required = (
        "config.json",
        "modular_model_index.json",
        "processor_config.json",
        "tokenizer.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing local Edge assets {missing}")
    checkpoint = CheckpointOverrides(checkpoint_path=str(path)).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )
    resolved = checkpoint.download_checkpoint().resolve()
    if resolved != path.resolve():
        raise RuntimeError(f"Local Edge checkpoint resolved to {resolved}, expected {path.resolve()}")
    model_config = checkpoint.load_model_config_dict()["config"]
    if int(model_config["max_action_dim"]) < G1_DEX3_ACTION_DIM:
        raise ValueError("Cosmos3-Edge max_action_dim is smaller than the G1 action")
    if int(model_config["num_embodiment_domains"]) <= G1_DEX3_DOMAIN_ID:
        raise ValueError("Cosmos3-Edge has no row for the G1 embodiment domain")
    return model_config


def _validate_transformed_sample(
    *,
    dataset_root: Path,
    selection: str,
    edge_path: Path,
    max_action_dim: int,
) -> tuple[int, ...]:
    tokenizer_config = L(build_processor_lazy)(
        repository=str(edge_path),
        revision="main",
    )
    dataset = get_action_g1_dex3_sft_dataset(
        root=str(dataset_root),
        datasets=selection,
        fps=30.0,
        chunk_length=32,
        mode="wam",
        use_state=True,
        split="full",
        use_image_augmentation=False,
        resolution="480",
        max_action_dim=max_action_dim,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=0.0,
        iterable_shuffle=False,
    )
    sample = dataset[0]
    action = sample["action"]
    action_raw = sample["action_raw"]
    video = sample["video"]
    if tuple(action.shape) != (33, max_action_dim):
        raise ValueError(f"transformed action shape {tuple(action.shape)} != (33, {max_action_dim})")
    if tuple(action_raw.shape) != (33, G1_DEX3_ACTION_DIM):
        raise ValueError(f"raw action shape {tuple(action_raw.shape)} != (33, 28)")
    if action.dtype != torch.float32 or action_raw.dtype != torch.float32:
        raise TypeError("G1 actions must remain float32 through preprocessing")
    if video.dtype != torch.uint8 or video.ndim != 4 or tuple(video.shape[:2]) != (3, 33):
        raise ValueError(f"transformed video must be uint8 [3,33,H,W], got {video.dtype} {tuple(video.shape)}")
    if "text_token_ids" not in sample or "sequence_plan" not in sample:
        raise ValueError("Cosmos3 tokenizer/sequence-plan output is missing")
    domain = int(sample["domain_id"].item())
    if domain != G1_DEX3_DOMAIN_ID:
        raise ValueError(f"G1 domain ID {domain} != {G1_DEX3_DOMAIN_ID}")
    return tuple(video.shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/home/tuan/projects/cibo/datasets")
    parser.add_argument("--datasets", default="all", help="all or comma-separated task/directory names")
    parser.add_argument(
        "--edge-checkpoint",
        default="/home/tuan/projects/cibo/hf_pretrained_models/Cosmos3-Edge",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Check schemas and every referenced shard without decoding/scanning samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    edge_path = Path(args.edge_checkpoint).expanduser().resolve()
    roots = resolve_g1_dex3_roots(dataset_root, args.datasets)
    model_config = _validate_local_edge_checkpoint(edge_path)

    failures: list[str] = []
    for root in roots:
        try:
            info, camera_keys = _read_and_validate_info(root)
            meta = LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")
            validate_g1_dex3_file_closure(root, meta, camera_keys)
            if not args.metadata_only:
                _scan_parquet(root, int(info["total_frames"]), float(info["fps"]))
                videos = _expected_video_files(root, meta, camera_keys)
                _decode_videos(videos, float(info["fps"]))
                shape = _validate_transformed_sample(
                    dataset_root=dataset_root,
                    selection=root.name,
                    edge_path=edge_path,
                    max_action_dim=int(model_config["max_action_dim"]),
                )
                task_note = G1_DEX3_TASK_CORRECTIONS.get(root.name)
                correction = f", corrected task={task_note!r}" if task_note else ""
                print(
                    f"PASS {root.name}: episodes={meta.total_episodes}, cameras={len(camera_keys)}, "
                    f"video={shape}, raw_action=(33,28), padded_action=(33,{model_config['max_action_dim']})"
                    f"{correction}"
                )
            else:
                print(
                    f"PASS {root.name}: LeRobot {info['codebase_version']}, episodes={meta.total_episodes}, "
                    f"cameras={len(camera_keys)}, referenced shards present"
                )
        except Exception as error:
            failures.append(f"{root}: {type(error).__name__}: {error}")

    if failures:
        print("G1 Dex3 validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"Validated {len(roots)} dataset(s): {G1_DEX3_ACTION_REPRESENTATION}, "
        f"{G1_DEX3_ACTION_DIM}D, units={G1_DEX3_ACTION_UNITS}, domain={G1_DEX3_DOMAIN_ID}."
    )


if __name__ == "__main__":
    main()
