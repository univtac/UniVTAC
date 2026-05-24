#!/usr/bin/env python
# Copyright 2026 The vitac_smolvla team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Convert raw visuo-tactile HDF5 episodes into a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

POLICY_ROOT = Path(__file__).resolve().parents[2]
SMOLVLA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POLICY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POLICY_ROOT))
sys.path.insert(0, str(SMOLVLA_ROOT / "src"))

from _base_data_preprocessor import BaseDataPreprocessor, HDF5Handler  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocess")


def _sorted_hdf5_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.hdf5"))
    try:
        return sorted(files, key=lambda x: int(x.stem))
    except ValueError:
        return sorted(files)


def _task_string_from_name(task_name: str) -> str:
    words = task_name.replace("-", "_").split("_")
    if len(words) >= 2:
        return f"{words[0]} the {' '.join(words[1:])}"
    return task_name.replace("_", " ")


class SmolVLADataPreprocessor(BaseDataPreprocessor):
    """BaseDataPreprocessor adapter that exports UniVTAC HDF5 data to LeRobot."""

    VISUAL_FEATURES = {
        "head": "observation.images.head",
        "wrist": "observation.images.wrist",
    }
    TACTILE_FEATURES = {
        "left": ("observation.images.tactile_ll", "observation.images.tactile_ll_marker"),
        "right": ("observation.images.tactile_lr", "observation.images.tactile_lr_marker"),
    }

    def __init__(self, task_name: str, collect_config_name: str):
        super().__init__(task_name, collect_config_name)

        self._tactile_stream_keys: dict[str, dict[str, str]] = {}

    def load_data(
        self,
        visual_cameras=("head", "wrist"),
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        episode_num=50,
        random_select=False,
    ):
        assert episode_num <= len(self.raw_hdf5_path), (
            f"Requested {episode_num} episodes, but only found {len(self.raw_hdf5_path)} under {self.raw_root_path}"
        )

        if random_select:
            self.selected_raw_hdf5_paths = np.random.choice(self.raw_hdf5_path, episode_num, replace=False)
        else:
            self.selected_raw_hdf5_paths = self.raw_hdf5_path[:episode_num]

        self.visual_cameras = list(visual_cameras)
        self.tactile_cameras = list(tactile_cameras)
        data_paths = [
            ("embodiment/ee", self.joint_transform),
            ("embodiment/joint", self.joint_transform),
        ]

        for cam in self.visual_cameras:
            cam_cfg = self.camera_key_map[f"visual/{cam}"]
            data_paths.append((cam_cfg["raw_key"], cam_cfg.get("transform")))

        self._tactile_stream_keys = self._resolve_tactile_stream_keys(self.selected_raw_hdf5_paths[0])
        for cam in self.tactile_cameras:
            stream_keys = self._tactile_stream_keys[cam]
            data_paths.append((stream_keys["rgb"], self.tactile_transform))
            data_paths.append((stream_keys["rgb_marker"], self.tactile_transform))

        self._data = HDF5Handler().batch_gather_hdf5(
            hdf5_paths=self.selected_raw_hdf5_paths,
            data_paths=data_paths,
            downsample_factor=downsample_factor,
        )
        return self._data

    def export_to_lerobot(
        self,
        repo_id: str,
        out_root: str | Path,
        fps: int = 30,
        task_text: str | None = None,
        robot_type: str = "vitac_arm_left",
        use_videos: bool = True,
        overwrite: bool = True,
    ) -> dict:
        assert self._data is not None, "Data not loaded. Please call load_data() before export_to_lerobot()."

        out_root = Path(out_root).expanduser().resolve()
        if out_root.exists():
            if not overwrite:
                raise FileExistsError(f"LeRobot dataset already exists: {out_root}")
            shutil.rmtree(out_root)

        image_arrays = self._lerobot_image_arrays()
        state_all = np.concatenate(
            [
                np.asarray(self._data["embodiment/ee_state"], dtype=np.float32),
                np.asarray(self._data["embodiment/joint_state"], dtype=np.float32),
            ],
            axis=1,
        )
        action_all = np.asarray(self._data["embodiment/joint_action"], dtype=np.float32)
        features = self._build_feature_schema(image_arrays, state_all.shape[1], action_all.shape[1])

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=out_root,
            robot_type=robot_type,
            use_videos=use_videos,
        )

        task_text = task_text or self._sample_instruction()
        start_idx = 0
        for ep_idx in tqdm(range(len(self.selected_raw_hdf5_paths)), desc="Writing LeRobot episodes"):
            end_idx = self._data["episode_ends"][ep_idx]
            for frame_idx in range(start_idx, end_idx):
                frame = {
                    "observation.state": state_all[frame_idx],
                    "action": action_all[frame_idx],
                    "task": task_text,
                }
                for key, images in image_arrays.items():
                    frame[key] = images[frame_idx]
                dataset.add_frame(frame)
            dataset.save_episode()
            start_idx = end_idx

        dataset.finalize()
        self.save_root_path = dataset.root
        metadata = {
            "repo_id": repo_id,
            "dataset_dir": str(dataset.root),
            "num_episodes": len(self.selected_raw_hdf5_paths),
            "camera_names": list(image_arrays.keys()),
            "episode_map": {i: str(path) for i, path in enumerate(self.selected_raw_hdf5_paths)},
        }
        with open(dataset.root / "source_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        return metadata

    def run(
        self,
        repo_id: str,
        out_root: str | Path,
        fps: int = 30,
        task_text: str | None = None,
        visual_cameras=("head", "wrist"),
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        episode_num=50,
        random_select=False,
        overwrite=True,
    ) -> dict:
        self.load_data(
            visual_cameras=visual_cameras,
            tactile_cameras=tactile_cameras,
            downsample_factor=downsample_factor,
            episode_num=episode_num,
            random_select=random_select,
        )
        return self.export_to_lerobot(
            repo_id=repo_id,
            out_root=out_root,
            fps=fps,
            task_text=task_text,
            overwrite=overwrite,
        )

    def visual_transform(self, images: np.ndarray) -> np.ndarray:
        resized = super().visual_transform(images)
        return resized

    def tactile_transform(self, images: np.ndarray) -> np.ndarray:
        resized = super().tactile_transform(images)
        return resized

    def _resolve_tactile_stream_keys(self, hdf5_path: Path) -> dict[str, dict[str, str]]:
        keys: dict[str, dict[str, str]] = {}
        with h5py.File(str(hdf5_path), "r") as f:
            for cam in self.tactile_cameras:
                candidates = (f"tactile/{cam}_gsmini", f"tactile/{cam}_tactile")
                for prefix in candidates:
                    if f"{prefix}/rgb" in f and f"{prefix}/rgb_marker" in f:
                        keys[cam] = {"rgb": f"{prefix}/rgb", "rgb_marker": f"{prefix}/rgb_marker"}
                        break
                if cam not in keys:
                    raise KeyError(
                        f"Could not find tactile rgb/rgb_marker datasets for '{cam}' in {hdf5_path}. "
                        f"Tried: {', '.join(candidates)}"
                    )
        return keys

    def _lerobot_image_arrays(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for cam in self.visual_cameras:
            raw_key = self.camera_key_map[f"visual/{cam}"]["raw_key"]
            images[self.VISUAL_FEATURES[cam]] = np.asarray(self._data[raw_key], dtype=np.uint8)
        for cam in self.tactile_cameras:
            rgb_key, marker_key = self.TACTILE_FEATURES[cam]
            images[rgb_key] = np.asarray(self._data[self._tactile_stream_keys[cam]["rgb"]], dtype=np.uint8)
            images[marker_key] = np.asarray(self._data[self._tactile_stream_keys[cam]["rgb_marker"]], dtype=np.uint8)
        return images

    @staticmethod
    def _build_feature_schema(image_arrays: dict[str, np.ndarray], state_dim: int, action_dim: int) -> dict:
        features: dict = {}
        for key, images in image_arrays.items():
            h, w = images.shape[1], images.shape[2]
            features[key] = {
                "dtype": "video",
                "shape": (3, int(h), int(w)),
                "names": ["channel", "height", "width"],
                "info": None,
            }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        }
        return features

    def _sample_instruction(self) -> str:
        for path in (
            POLICY_ROOT / "instructions" / f"{self.task_name}.json",
            self.raw_root_path.parent / "instructions.json",
        ):
            if path.exists():
                with open(path, "r") as f:
                    instructions = json.load(f).get("instructions", {"seen": []})["seen"]
                if instructions:
                    return str(np.random.choice(instructions))
        return _task_string_from_name(self.task_name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_name", nargs="?", help="Task name for BaseDataPreprocessor mode, e.g. lift_bottle.")
    p.add_argument("task_config", nargs="?", help="Task config for BaseDataPreprocessor mode, e.g. clean.")
    p.add_argument("expert_data_num", nargs="?", type=int, help="Number of episodes to process.")
    p.add_argument("--out", type=Path, default=None, help="Output LeRobot v3.0 dataset root directory.")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", type=str, default=None, help="Task description string.")
    p.add_argument("--downsample-factor", type=int, default=1)
    p.add_argument("--no-overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task_name is None or args.task_config is None or args.expert_data_num is None:
        raise SystemExit("Provide positional: task_name task_config expert_data_num")
    repo_id = args.repo_id or f"local/{args.task_name}-{args.task_config}-{args.expert_data_num}_vitac"
    out_root = args.out or (SMOLVLA_ROOT / "data" / f"{args.task_name}_{args.task_config}_{args.expert_data_num}_lerobot")
    processor = SmolVLADataPreprocessor(args.task_name, args.task_config)
    metadata = processor.run(
        repo_id=repo_id,
        out_root=out_root,
        fps=args.fps,
        task_text=args.task,
        downsample_factor=args.downsample_factor,
        episode_num=args.expert_data_num,
        overwrite=not args.no_overwrite,
    )

    log.info("LeRobot repo_id: %s", metadata["repo_id"])
    log.info("LeRobot dataset_dir: %s", metadata["dataset_dir"])


if __name__ == "__main__":
    main()
