import hydra
import os
import pathlib
from omegaconf import OmegaConf
import numpy as np
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import sys
import rerun as rr
from tqdm import tqdm

EPISODES = os.getenv("EPISODES")

JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
]

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("UniT", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    rr.init("unit_demonstration_data_vis", spawn=False)
    save_path = pathlib.Path(cfg.dataset_path) / "debug.rrd"
    if save_path.exists():
        save_path.unlink()
    rr.save(str(save_path))

    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    episode_idxs = [int(EPISODE.strip()) for EPISODE in EPISODES.split(",")]

    for episode_idx in tqdm(episode_idxs, "Loading episodes"):
        vis_episode = dataset.replay_buffer.get_episode(episode_idx)
        action_buffer = vis_episode["action"]
        qpos_buffer = vis_episode["qpos"]
        size = action_buffer.shape[0]

        action_dim = action_buffer.shape[1]
        for i in range(size):
            for j in range(action_buffer.shape[1]):
                if action_dim == 14:
                    if j < len(JOINT_NAMES):
                        name = "left_"
                    else:
                        name = "right_"
                    name += JOINT_NAMES[j % len(JOINT_NAMES)]
                    rr.log(f"action/{name}", rr.Scalar(action_buffer[i, j]))
                    rr.log(f"qpos/{name}", rr.Scalar(qpos_buffer[i, j]))
                elif action_dim == 7:
                    name = JOINT_NAMES[j]
                    rr.log(f"action/{name}", rr.Scalar(action_buffer[i, j]))
                    rr.log(f"qpos/{name}", rr.Scalar(qpos_buffer[i, j]))

            for name in [
                "cam_low",
                "cam_high",
                "cam_right_wrist",
                "cam_left_wrist",
                "tactile_right_image",
                "tactile_left_image",
            ]:
                if name not in vis_episode:
                    continue
                # for tactile iamge, convert color channels
                if name in ["tactile_right_image", "tactile_left_image"]:
                    rr.log(f"image/{name}", rr.Image(vis_episode[name][i][:, :, [2, 1, 0]]))
                else:
                    rr.log(f"image/{name}", rr.Image(vis_episode[name][i]))
            rr.log("episode", rr.Scalar(episode_idx))

    rr.disconnect()
    print(f"Saved at {save_path}!")

# %%
if __name__ == "__main__":
    main()