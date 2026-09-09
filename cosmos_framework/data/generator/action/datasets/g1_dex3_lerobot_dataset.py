# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unitree G1 Dex3 LeRobot v3 dataset for Cosmos3 action-policy SFT.

The implementation follows :class:`DROIDLeRobotDataset`: LeRobot supplies
timestamp-aligned state, action, and video windows; the initial robot state is
prepended to the absolute joint-position target chunk; and the shared
``ActionTransformPipeline`` performs model-space padding and masking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionSpec,
    BaseActionLeRobotDataset,
    DimType,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id

G1_DEX3_EMBODIMENT = "unitree_g1_dex3"
G1_DEX3_ACTION_DIM = 28
G1_DEX3_DOMAIN_ID = get_domain_id(G1_DEX3_EMBODIMENT)
G1_DEX3_ACTION_REPRESENTATION = "absolute_joint_position_target"
G1_DEX3_ACTION_UNITS = "radian"

G1_DEX3_DATASET_DIRECTORIES: dict[str, str] = {
    "CameraPackaging": "G1_Dex3_CameraPackaging_Dataset",
    "ObjectPlacement": "G1_Dex3_ObjectPlacement_Dataset",
    "GraspSquare": "G1_Dex3_GraspSquare_Dataset",
    "PickBottle": "G1_Dex3_PickBottle_Dataset",
    "Pouring": "G1_Dex3_Pouring_Dataset",
    "PickDoll": "G1_Dex3_PickDoll_Dataset",
}

G1_DEX3_JOINT_NAMES: tuple[str, ...] = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
)

G1_DEX3_HIGH_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.cam_left_high",
    "observation.images.cam_right_high",
)
G1_DEX3_WRIST_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
G1_DEX3_CAMERA_LAYOUTS = (
    frozenset(G1_DEX3_HIGH_CAMERA_KEYS),
    frozenset((*G1_DEX3_HIGH_CAMERA_KEYS, *G1_DEX3_WRIST_CAMERA_KEYS)),
)

# The GraspSquare release has a copied ``camera packaging`` task row. Keep the
# source dataset immutable while preventing that known metadata error from
# training the wrong instruction.
G1_DEX3_TASK_CORRECTIONS = {
    "G1_Dex3_GraspSquare_Dataset": "grasp square",
}

_SUPPORTED_MODES = {"forward_dynamics", "inverse_dynamics", "joint", "policy", "wam"}


def resolve_g1_dex3_roots(
    dataset_root: str | Path,
    datasets: str | Sequence[str] = "all",
) -> tuple[Path, ...]:
    """Resolve friendly names or exact directory names without copying data."""

    root = Path(dataset_root).expanduser().resolve()
    if isinstance(datasets, str):
        requested = [item.strip() for item in datasets.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in datasets if str(item).strip()]
    if not requested or [item.lower() for item in requested] == ["all"]:
        requested = list(G1_DEX3_DATASET_DIRECTORIES)

    aliases = {name.lower(): directory for name, directory in G1_DEX3_DATASET_DIRECTORIES.items()}
    aliases.update({directory.lower(): directory for directory in G1_DEX3_DATASET_DIRECTORIES.values()})
    unknown = [item for item in requested if item.lower() not in aliases]
    if unknown:
        raise ValueError(
            f"Unknown G1 Dex3 dataset selection {unknown}; choose from {list(G1_DEX3_DATASET_DIRECTORIES)} or 'all'"
        )

    roots = tuple(root / aliases[item.lower()] for item in requested)
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing G1 Dex3 dataset directories: " + ", ".join(missing))
    return roots


def _flatten_feature_names(feature: dict[str, Any]) -> tuple[str, ...]:
    names = feature.get("names") or ()
    if len(names) == 1 and isinstance(names[0], (list, tuple)):
        names = names[0]
    return tuple(str(name) for name in names)


def _read_and_validate_info(root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid LeRobot metadata {info_path}: {error}") from error

    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{root}: expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if info.get("robot_type") not in {"Unitree_G1", "Unitree_G1_Dex3"}:
        raise ValueError(f"{root}: unsupported robot_type {info.get('robot_type')!r}")
    if float(info.get("fps", 0.0)) != 30.0:
        raise ValueError(f"{root}: G1 Dex3 data must be 30 Hz, got {info.get('fps')!r}")
    if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError(f"{root}: dataset has no episodes or frames")

    features = info.get("features") or {}
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"{root}: missing feature {key!r}")
        if feature.get("dtype") != "float32" or tuple(feature.get("shape", ())) != (G1_DEX3_ACTION_DIM,):
            raise ValueError(
                f"{root}: {key} must be float32[{G1_DEX3_ACTION_DIM}], got "
                f"{feature.get('dtype')} {feature.get('shape')}"
            )
        names = _flatten_feature_names(feature)
        if names != G1_DEX3_JOINT_NAMES:
            raise ValueError(f"{root}: {key} joint order does not match the pinned G1 Dex3 contract")

    camera_keys = tuple(key for key, feature in features.items() if feature.get("dtype") == "video")
    if frozenset(camera_keys) not in G1_DEX3_CAMERA_LAYOUTS:
        raise ValueError(f"{root}: expected stereo high cameras with an optional stereo wrist pair, got {camera_keys}")
    ordered_camera_keys = G1_DEX3_HIGH_CAMERA_KEYS + tuple(
        key for key in G1_DEX3_WRIST_CAMERA_KEYS if key in camera_keys
    )
    for key in ordered_camera_keys:
        feature = features[key]
        video_info = feature.get("info") or {}
        height = int(video_info.get("video.height", 0))
        width = int(video_info.get("video.width", 0))
        channels = int(video_info.get("video.channels", 0))
        shape = tuple(feature.get("shape", ()))
        if channels != 3 or shape not in {(3, height, width), (height, width, 3)}:
            raise ValueError(f"{root}: inconsistent RGB video metadata for {key}")
        if float(video_info.get("video.fps", 0.0)) != 30.0:
            raise ValueError(f"{root}: {key} is not 30 Hz")
    return info, ordered_camera_keys


def validate_g1_dex3_file_closure(root: Path, meta: Any, camera_keys: Sequence[str]) -> None:
    """Ensure every Parquet/video shard referenced by episode metadata exists."""

    info = meta.info
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise ValueError(f"{root}: missing LeRobot v3 data_path/video_path templates")
    if meta.tasks is None or meta.tasks.empty or any(not str(task).strip() for task in meta.tasks.index):
        raise ValueError(f"{root}: task metadata is empty")

    expected_files: set[Path] = set()
    for episode in meta.episodes:
        data_chunk = int(episode["data/chunk_index"])
        data_file = int(episode["data/file_index"])
        expected_files.add(root / data_template.format(chunk_index=data_chunk, file_index=data_file))
        for key in camera_keys:
            prefix = f"videos/{key}"
            expected_files.add(
                root
                / video_template.format(
                    video_key=key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]),
                    file_index=int(episode[f"{prefix}/file_index"]),
                )
            )
    missing = sorted(str(path) for path in expected_files if not path.is_file())
    if missing:
        preview = "\n".join(missing[:20])
        suffix = f"\n... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"{root}: missing referenced LeRobot shards:\n{preview}{suffix}")


class G1Dex3LeRobotDataset(BaseActionLeRobotDataset):
    """Read one or more local G1 Dex3 datasets for native Cosmos3 policy SFT.

    ``observation.state`` is the current 28-D joint position and ``action`` is
    the next absolute 28-D joint-position target, both in radians and in
    :data:`G1_DEX3_JOINT_NAMES` order. These are robot commands, not SONIC
    latent actions.
    """

    EMBODIMENT_TYPE = G1_DEX3_EMBODIMENT

    def __init__(
        self,
        *,
        root: str,
        datasets: str | Sequence[str] = "all",
        fps: float = 30.0,
        chunk_length: int = 32,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "wam",
        use_state: bool = True,
        viewpoint: str = "concat_view",
        action_representation: str = G1_DEX3_ACTION_REPRESENTATION,
        action_normalization: str | None = None,
        use_image_augmentation: bool = True,
        tolerance_s: float = 2e-4,
        skip_video_loading: bool = False,
    ) -> None:
        if fps != 30.0:
            raise ValueError(f"G1 Dex3 policy uses the recorded 30 Hz rate, got {fps}")
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {chunk_length}")
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported action training mode {mode!r}")
        if viewpoint != "concat_view":
            raise ValueError("G1 Dex3 preserves all available cameras via concat_view")
        if action_representation != G1_DEX3_ACTION_REPRESENTATION:
            raise ValueError(
                f"Unsupported G1 action representation {action_representation!r}; "
                f"expected {G1_DEX3_ACTION_REPRESENTATION!r}"
            )
        if action_normalization is not None:
            raise ValueError("No pinned G1 normalization is available; use raw joint positions")

        roots = resolve_g1_dex3_roots(root, datasets)
        source_info = [_read_and_validate_info(path) for path in roots]
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=G1_DEX3_EMBODIMENT,
            viewpoint="concat_view",
            pose_convention="absolute_joint_position",
            rotation_format=None,
            action_normalization=None,
            tolerance_s=tolerance_s,
            skip_video_loading=skip_video_loading,
        )
        self._use_state = use_state
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None
        self._source_roots = roots
        self._camera_keys: list[tuple[str, ...]] = []
        self._all_shard_roots = [str(path) for path in roots]

        observation_ts = [index * self._dt for index in range(chunk_length + 1)]
        action_ts = [index * self._dt for index in range(chunk_length)]
        for root_path, (_, camera_keys) in zip(roots, source_info):
            delta_timestamps = {
                "observation.state": observation_ts,
                "action": action_ts,
                **{key: observation_ts for key in camera_keys},
            }
            meta = self._register_source(
                root=str(root_path),
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                dataset_label=root_path.name,
            )
            validate_g1_dex3_file_closure(root_path, meta, camera_keys)
            self._camera_keys.append(camera_keys)

    @property
    def action_dim(self) -> int:
        """Return the unpadded G1 action width."""

        return G1_DEX3_ACTION_DIM

    @property
    def state_dim(self) -> int:
        """Return the G1 state width."""

        return G1_DEX3_ACTION_DIM

    def _build_action_spec(self) -> ActionSpec:
        return ActionSpec(
            names=list(G1_DEX3_JOINT_NAMES),
            types=[DimType.JOINT] * G1_DEX3_ACTION_DIM,
        )

    def _compute_idle_frames(self, raw_action: torch.Tensor) -> None:
        del raw_action
        return None

    def _compose_multi_view(self, sample: dict[str, Any], dataset_index: int) -> torch.Tensor:
        """Compose every available camera into a stable 2x2 stereo layout."""

        camera_keys = self._camera_keys[dataset_index]
        views = [sample[key] for key in camera_keys]
        if len({tuple(view.shape) for view in views}) != 1:
            raise ValueError(f"G1 camera tensors have incompatible shapes: {[tuple(v.shape) for v in views]}")
        if any(view.ndim != 4 or not torch.is_floating_point(view) for view in views):
            raise ValueError("G1 cameras must decode as float [T,C,H,W] tensors")

        if self._use_image_augmentation:
            if self._image_augmentor is None:
                _, _, height, width = views[0].shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(height * 0.95), int(width * 0.95))),
                        T.Resize((height, width), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            frames = views[0].shape[0]
            views = list(self._image_augmentor(torch.cat(views, dim=0)).split(frames, dim=0))

        high_row = torch.cat(views[:2], dim=-1)
        if len(views) == 4:
            wrist_row = torch.cat(views[2:], dim=-1)
        else:
            wrist_row = torch.zeros_like(high_row)
        return torch.cat([high_row, wrist_row], dim=-2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one aligned video/state/action window."""

        mode, dataset_index, _, sample = self._fetch_sample(idx)
        state = sample["observation.state"].float()
        target = sample["action"].float()
        expected_state_shape = (self._chunk_length + 1, G1_DEX3_ACTION_DIM)
        expected_action_shape = (self._chunk_length, G1_DEX3_ACTION_DIM)
        if tuple(state.shape) != expected_state_shape:
            raise ValueError(f"G1 state shape {tuple(state.shape)} != {expected_state_shape}")
        if tuple(target.shape) != expected_action_shape:
            raise ValueError(f"G1 action shape {tuple(target.shape)} != {expected_action_shape}")
        if not torch.isfinite(state).all() or not torch.isfinite(target).all():
            raise ValueError("G1 state/action contains NaN or Inf")

        action = torch.cat([state[:1], target], dim=0) if self._use_state else target
        source_name = self._source_roots[dataset_index].name
        caption = G1_DEX3_TASK_CORRECTIONS.get(source_name, str(sample["task"]).strip())
        if not caption:
            raise ValueError(f"{source_name}: sample has an empty task instruction")
        video = None if self._skip_video_loading else self._compose_multi_view(sample, dataset_index)
        if len(self._camera_keys[dataset_index]) == 4:
            view_description = (
                "The top row shows the left and right high cameras. "
                "The bottom row shows the left and right wrist cameras."
            )
        else:
            view_description = (
                "The top row shows the left and right high cameras. "
                "Wrist cameras were not recorded, so the bottom row is blank."
            )
        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=caption,
            additional_view_description=view_description,
        )
