import json
import os
import sys
import types
from collections import deque
from pathlib import Path

import cv2
import dill
import torch
from omegaconf import OmegaConf

UNIT_ROOT = Path(__file__).parent
sys.path.insert(0, str(UNIT_ROOT))
sys.path.append(str(UNIT_ROOT.parent))
from _base_policy import BasePolicy


class Policy(BasePolicy):
    def __init__(self, args):
        self.task_name = args["task_name"]
        self.task_config = args["task_config"]
        self.train_config_name = os.environ.get("TRAIN_CONFIG", args.get("train_config_name", "train_config"))
        self.ep_num = str(os.environ.get("EP_NUM", args.get("expert_data_num", "50")))

        with open(Path(__file__).parent.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        self.camera_type = task_settings.get(self.task_name, {}).get("camera_type", "head")

        ckpt_path = args.get("ckpt_path")
        if ckpt_path is None:
            ckpt_path = (
                Path(__file__).parent
                / "unit_ckpt"
                / f"unit-{self.task_name}"
                / f"{self.task_config}-{self.ep_num}"
                / self.train_config_name
                / "checkpoints"
                / "latest.ckpt"
            )
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"UniT checkpoint not found: {self.ckpt_path}")

        payload = torch.load(self.ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        self.device = torch.device(args.get("device", cfg.training.device))
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.action_queue = deque()
        self.obs_history = deque(maxlen=self.n_obs_steps)

        import hydra
        unit_pkg = types.ModuleType("UniT")
        unit_pkg.__path__ = [str(UNIT_ROOT / "UniT")]
        sys.modules["UniT"] = unit_pkg

        self.model = hydra.utils.instantiate(cfg.policy)
        state_key = "ema_model" if cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
        self.model.load_state_dict(payload["state_dicts"][state_key])
        self.model.to(self.device)
        self.model.eval()

    def _image(self, image: torch.Tensor, hw):
        arr = image.detach().cpu().numpy()
        arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        arr = arr.astype("float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def encode_obs(self, observation):
        visual_hw = (224, 224)
        tactile_hw = (128, 160)
        if self.camera_type == "all":
            cam = observation["observation"]["head"]["rgb"]
        else:
            cam = observation["observation"][self.camera_type]["rgb"]
        return {
            "cam_high": self._image(cam, visual_hw),
            "tactile_left_image": self._image(observation["tactile"]["left_tactile"]["rgb_marker"], tactile_hw),
            "tactile_right_image": self._image(observation["tactile"]["right_tactile"]["rgb_marker"], tactile_hw),
            "qpos": observation["embodiment"]["joint"][:8].detach().cpu().float(),
        }

    def _stack_history(self):
        obs_dict = {}
        for key in self.obs_history[0]:
            obs_dict[key] = torch.stack([obs[key] for obs in self.obs_history], dim=0).unsqueeze(0).to(self.device)
        return obs_dict

    def eval(self, task, observation):
        obs = self.encode_obs(observation)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.append(obs)
        self.obs_history.append(obs)

        if not self.action_queue:
            with torch.no_grad():
                result = self.model.predict_action(self._stack_history())
            actions = result["action"][0, : self.n_action_steps].detach().cpu()
            self.action_queue.extend(actions)

        action = self.action_queue.popleft().to(task.device).float()
        task.take_action(action, action_type="qpos")

    def reset(self):
        self.action_queue.clear()
        self.obs_history.clear()
