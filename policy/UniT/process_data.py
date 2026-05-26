import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from _base_data_preprocessor import BaseDataPreprocessor


class UniTDataPreprocessor(BaseDataPreprocessor):
    def visual_transform(self, images: np.ndarray) -> np.ndarray:
        return np.stack(
            [cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR) for img in images],
            axis=0,
            dtype=np.uint8,
        )

    def tactile_transform(self, images: np.ndarray) -> np.ndarray:
        return np.stack(
            [cv2.resize(img, (160, 128), interpolation=cv2.INTER_LINEAR) for img in images],
            axis=0,
            dtype=np.uint8,
        )

    def joint_transform(self, joints: np.ndarray) -> np.ndarray:
        return joints[:, :8]


def main(task_name, task_config, expert_data_num):
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"

    task_settings_path = Path(__file__).parent.parent / "task_settings.json"
    if task_settings_path.exists():
        with open(task_settings_path, "r") as f:
            task_settings = json.load(f)
    else:
        task_settings = {}

    camera_type = task_settings.get(task_name, {}).get("camera_type", "head")
    visual_cameras = ["head", "wrist"] if camera_type == "all" else [camera_type]
    tactile_cameras = ["left", "right"]
    downsample_factor = task_settings.get(task_name, {}).get(
        "downsample_factor", task_settings.get(task_name, {}).get("downsample", 1)
    )

    processor = UniTDataPreprocessor(task_name, task_config)
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=visual_cameras,
        tactile_cameras=tactile_cameras,
        downsample_factor=downsample_factor,
        episode_num=expert_data_num,
        random_select=False,
    )

    sim_task_configs_path = Path(__file__).parent / "SIM_TASK_CONFIGS.json"
    try:
        with open(sim_task_configs_path, "r") as f:
            sim_task_configs = json.load(f)
    except Exception:
        sim_task_configs = {}

    sim_task_configs[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": metadata["dataset_dir"],
        "num_episodes": metadata["num_episodes"],
        "episode_len": metadata["episode_len"],
        "camera_names": metadata["camera_names"],
    }

    with open(sim_task_configs_path, "w") as f:
        json.dump(sim_task_configs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for UniT training.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    args = parser.parse_args()
    main(args.task_name, args.task_config, args.expert_data_num)
