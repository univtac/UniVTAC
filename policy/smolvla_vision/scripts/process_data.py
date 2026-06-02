#!/usr/bin/env python
"""Convert UniVTAC HDF5 episodes into a vision-only LeRobot v3.0 dataset.

The exported dataset uses the original LeRobot SmolVLA feature surface:
head + wrist RGB cameras, robot state, action, and task text. Tactile streams
are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

POLICY_ROOT = Path(__file__).resolve().parents[2]
SMOLVLA_VISION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POLICY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POLICY_ROOT))
sys.path.insert(0, str(SMOLVLA_VISION_ROOT / "lerobot" / "src"))

from _base_data_preprocessor import BaseDataPreprocessor, HDF5Handler  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smolvla_vision.process_data")


def _task_string_from_name(task_name: str) -> str:
    words = task_name.replace("-", "_").split("_")
    if len(words) >= 2:
        return f"{words[0]} the {' '.join(words[1:])}"
    return task_name.replace("_", " ")


class SmolVLAVisionDataPreprocessor(BaseDataPreprocessor):
    """Export only visual observations from UniVTAC raw HDF5 episodes."""

    VISUAL_FEATURES = {
        "head": "observation.images.head",
        "wrist": "observation.images.wrist",
    }

    def load_data(
        self,
        visual_cameras=("head", "wrist"),
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
        self.tactile_cameras = []
        data_paths = [
            ("embodiment/ee", self.joint_transform),
            ("embodiment/joint", self.joint_transform),
        ]
        for cam in self.visual_cameras:
            cam_cfg = self.camera_key_map[f"visual/{cam}"]
            data_paths.append((cam_cfg["raw_key"], cam_cfg.get("transform")))

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
        features = self._build_feature_schema(image_arrays, state_all.shape[1], action_all.shape[1], use_videos)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=out_root,
            robot_type=robot_type,
            use_videos=use_videos,
        )

        task_text = task_text or self._sample_instruction()
        episode_slices = self._episode_slices()
        for _ep_idx, start_idx, end_idx in tqdm(episode_slices, desc="Writing LeRobot episodes"):
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

        dataset.finalize()
        self.save_root_path = dataset.root
        metadata = {
            "repo_id": repo_id,
            "dataset_dir": str(dataset.root),
            "num_episodes": len(self.selected_raw_hdf5_paths),
            "camera_names": list(image_arrays.keys()),
            "ignored_modalities": ["tactile"],
            "episode_map": {i: str(path) for i, path in enumerate(self.selected_raw_hdf5_paths)},
            "episode_slices": {
                i: {
                    "source": str(self.selected_raw_hdf5_paths[i]),
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start),
                }
                for i, start, end in episode_slices
            },
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
        downsample_factor=1,
        episode_num=50,
        random_select=False,
        overwrite=True,
        use_videos: bool = True,
    ) -> dict:
        self.load_data(
            visual_cameras=visual_cameras,
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
            use_videos=use_videos,
        )

    def _lerobot_image_arrays(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for cam in self.visual_cameras:
            raw_key = self.camera_key_map[f"visual/{cam}"]["raw_key"]
            images[self.VISUAL_FEATURES[cam]] = np.asarray(self._data[raw_key], dtype=np.uint8)
        return images

    def _episode_slices(self) -> list[tuple[int, int, int]]:
        episode_ends = np.asarray(self._data["episode_ends"], dtype=np.int64)
        if len(episode_ends) != len(self.selected_raw_hdf5_paths):
            raise ValueError(
                "HDF5Handler returned episode_ends with length "
                f"{len(episode_ends)}, but {len(self.selected_raw_hdf5_paths)} source HDF5 files were selected."
            )

        slices: list[tuple[int, int, int]] = []
        start_idx = 0
        for ep_idx, end_idx in enumerate(episode_ends):
            end = int(end_idx)
            if end <= start_idx:
                raise ValueError(f"Invalid episode slice for episode {ep_idx}: start={start_idx}, end={end}")
            slices.append((ep_idx, start_idx, end))
            start_idx = end
        return slices

    @staticmethod
    def _build_feature_schema(
        image_arrays: dict[str, np.ndarray], state_dim: int, action_dim: int, use_videos: bool
    ) -> dict:
        features: dict = {}
        for key, images in image_arrays.items():
            h, w = images.shape[1], images.shape[2]
            features[key] = {
                "dtype": "video" if use_videos else "image",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", nargs="?", help="Task name, e.g. lift_bottle.")
    parser.add_argument("task_config", nargs="?", help="Task config, e.g. demo.")
    parser.add_argument("expert_data_num", nargs="?", type=int, help="Number of episodes to process.")
    parser.add_argument("--out", type=Path, default=None, help="Output LeRobot v3.0 dataset root directory.")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str, default=None, help="Task description string.")
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--random-select", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--images", action="store_true", help="Store visual modalities as image files.")
    storage.add_argument("--videos", action="store_true", help="Store visual modalities as MP4 videos. Default.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task_name is None or args.task_config is None or args.expert_data_num is None:
        raise SystemExit("Provide positional: task_name task_config expert_data_num")

    settings = f"{args.task_name}_{args.task_config}_{args.expert_data_num}"
    repo_id = args.repo_id or f"local/{settings}_vision"
    out_root = args.out or (SMOLVLA_VISION_ROOT / "data" / repo_id)
    processor = SmolVLAVisionDataPreprocessor(args.task_name, args.task_config)
    metadata = processor.run(
        repo_id=repo_id,
        out_root=out_root,
        fps=args.fps,
        task_text=args.task,
        downsample_factor=args.downsample_factor,
        episode_num=args.expert_data_num,
        random_select=args.random_select,
        overwrite=not args.no_overwrite,
        use_videos=not args.images,
    )

    log.info("LeRobot repo_id: %s", metadata["repo_id"])
    log.info("LeRobot dataset_dir: %s", metadata["dataset_dir"])
    log.info("Vision cameras: %s", ", ".join(metadata["camera_names"]))


if __name__ == "__main__":
    main()
