"""UniT deployment policy for UniVTAC evaluation.

The public ``Policy`` class is a lightweight FastAPI client used inside the
IsaacLab evaluation process. The heavy UniT / diffusion-policy stack is loaded
in ``unit_server.py`` through ``_UniTLocalPolicy`` so both environments can stay
separate, following the SmolVLA deployment pattern.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

from .._base_policy import BasePolicy


UNIT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = UNIT_ROOT.parents[1]


def _setup_unit_paths() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(UNIT_ROOT))
    sys.path.insert(0, str(UNIT_ROOT / "third_party" / "diffusion_policy"))


def _to_wire_payload(value: Any):
    """Convert tensors/arrays to JSON-safe containers for HTTP transport."""
    import numpy as np

    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(k): _to_wire_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return _to_wire_payload(np.asarray(value))


def _from_wire_payload(value: Any):
    """Restore ndarray payloads serialized by ``_to_wire_payload``."""
    import numpy as np

    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            array = np.frombuffer(data, dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"]).copy()
        return {k: _from_wire_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire_payload(v) for v in value]
    return value


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shape_meta_obs(cfg) -> dict:
    if "task" in cfg and "dataset" in cfg.task and "shape_meta" in cfg.task.dataset:
        return cfg.task.dataset.shape_meta.obs
    return cfg.policy.shape_meta.obs


def _resolve_checkpoint(args: dict) -> Path:
    if args.get("ckpt_path"):
        ckpt_path = Path(args["ckpt_path"]).expanduser()
        if ckpt_path.exists():
            return ckpt_path
        raise FileNotFoundError(f"UniT checkpoint not found: {ckpt_path}")

    if args.get("ckpt_dir"):
        ckpt_dir = Path(args["ckpt_dir"]).expanduser()
        candidates = [
            ckpt_dir / "checkpoints" / "last.ckpt",
            ckpt_dir / "checkpoints" / "latest.ckpt",
            ckpt_dir / "last.ckpt",
            ckpt_dir / "latest.ckpt",
        ]
    else:
        task_name = args["task_name"]
        task_config = args["task_config"]
        ep_num = str(os.environ.get("EP_NUM", args.get("expert_data_num", "50")))
        train_config_name = os.environ.get(
            "TRAIN_CONFIG", args.get("train_config_name", "train_config")
        )
        ckpt_roots = [
            UNIT_ROOT / "unit_ckpt" / task_name / f"{task_config}-{ep_num}" / train_config_name,
            UNIT_ROOT / "unit_ckpt" / f"unit-{task_name}" / f"{task_config}-{ep_num}" / train_config_name,
            UNIT_ROOT / "unit_ckpt" / task_name / f"demo-{ep_num}" / train_config_name,
            UNIT_ROOT / "unit_ckpt" / f"unit-{task_name}" / f"demo-{ep_num}" / train_config_name,
        ]
        candidates = []
        for root in ckpt_roots:
            candidates.extend(
                [
                    root / "checkpoints" / "last.ckpt",
                    root / "checkpoints" / "latest.ckpt",
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"UniT checkpoint not found. Checked:\n{checked}")


def _resize_hwc_to_chw_float(image, shape: tuple[int, int, int]):
    import cv2
    import numpy as np
    import torch

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {array.shape}")

    _, target_h, target_w = shape
    if array.shape[:2] != (target_h, target_w):
        array = cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    array = array.astype(np.float32)
    if array.size and array.max() > 1.5:
        array = array / 255.0
    return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


class _UniTLocalPolicy(BasePolicy):
    """Heavy UniT model runner instantiated only by ``unit_server.py``."""

    _IMAGE_SOURCES = {
        "cam_high": ("observation", "head", "rgb"),
        "cam_low": ("observation", "head", "rgb"),
        "cam_wrist": ("observation", "wrist", "rgb"),
        "cam_right_wrist": ("observation", "wrist", "rgb"),
        "cam_left_wrist": ("observation", "wrist", "rgb"),
        "tactile_left_image": ("tactile", "left_tactile", "rgb_marker"),
        "tactile_right_image": ("tactile", "right_tactile", "rgb_marker"),
    }

    def __init__(self, args: dict):
        _setup_unit_paths()

        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)

        self.task_name = args["task_name"]
        self.ckpt_path = _resolve_checkpoint(args)
        print(f"[unit] loading checkpoint from {self.ckpt_path}")

        payload = torch.load(
            self.ckpt_path.open("rb"),
            pickle_module=dill,
            map_location="cpu",
        )
        cfg = payload["cfg"]
        self.cfg = cfg
        self.device = torch.device(
            args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.action_queue = deque()
        self.obs_history = deque(maxlen=self.n_obs_steps)

        self.obs_shape_meta = _shape_meta_obs(cfg)
        self.action_dim = int(cfg.policy.shape_meta.action.shape[0])
        self.rgb_keys = [
            key for key, attr in self.obs_shape_meta.items() if attr.get("type", "low_dim") == "rgb"
        ]
        self.lowdim_keys = [
            key for key, attr in self.obs_shape_meta.items() if attr.get("type", "low_dim") == "low_dim"
        ]

        self.model = hydra.utils.instantiate(cfg.policy)
        state_dicts = payload["state_dicts"]
        state_key = "ema_model" if cfg.training.use_ema and "ema_model" in state_dicts else "model"
        self.model.load_state_dict(state_dicts[state_key])
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[unit] ready; state={state_key}, obs={list(self.obs_shape_meta.keys())}, "
            f"action_dim={self.action_dim}, n_obs_steps={self.n_obs_steps}, "
            f"n_action_steps={self.n_action_steps}"
        )

    def _pick_image(self, observation: dict, key: str):
        root, cam, image_key = self._IMAGE_SOURCES[key]
        try:
            return observation[root][cam][image_key]
        except KeyError:
            if root == "tactile" and image_key == "rgb_marker":
                return observation[root][cam]["rgb"]
            raise KeyError(f"observation missing {root}.{cam}.{image_key} for UniT key {key}")

    def encode_obs(self, observation: dict) -> dict:
        import torch

        obs = {}
        for key in self.rgb_keys:
            if key not in self._IMAGE_SOURCES:
                raise KeyError(f"no UniVTAC observation mapping for UniT rgb key {key}")
            shape = tuple(int(x) for x in self.obs_shape_meta[key]["shape"])
            obs[key] = _resize_hwc_to_chw_float(self._pick_image(observation, key), shape)

        for key in self.lowdim_keys:
            shape = tuple(int(x) for x in self.obs_shape_meta[key]["shape"])
            if key != "qpos":
                raise KeyError(f"no UniVTAC observation mapping for UniT low_dim key {key}")
            qpos = torch.as_tensor(observation["embodiment"]["joint"]).detach().cpu().float().view(-1)
            obs[key] = qpos[: shape[0]]

        return obs

    def _stack_history(self):
        import torch

        obs_dict = {}
        for key in self.obs_history[0]:
            obs_dict[key] = torch.stack(
                [obs[key] for obs in self.obs_history], dim=0
            ).unsqueeze(0).to(self.device)
        return obs_dict

    def get_action(self, observation: dict):
        import numpy as np
        import torch

        obs = self.encode_obs(observation)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.append(obs)
        self.obs_history.append(obs)

        if not self.action_queue:
            with torch.no_grad():
                result = self.model.predict_action(self._stack_history())
            actions = result["action"][0, : self.n_action_steps].detach().cpu()
            self.action_queue.extend(actions)

        return self.action_queue.popleft().numpy().astype(np.float32)

    def eval(self, task, observation):
        import torch

        action = torch.from_numpy(self.get_action(observation)).to(task.device).float()
        return task.take_action(action, action_type="qpos")

    def reset(self):
        self.action_queue.clear()
        self.obs_history.clear()


class _UniTFastAPIClient:
    def __init__(self, args: dict):
        self.host = str(args.get("unit_host", os.environ.get("UNIT_HOST", "127.0.0.1")))
        self.port = int(args.get("unit_port", os.environ.get("UNIT_PORT", 0)) or 0)
        self.python = Path(
            args.get(
                "unit_python",
                os.environ.get(
                    "UNIT_PYTHON",
                    str(UNIT_ROOT / ".venv" / "bin" / "python")
                    if (UNIT_ROOT / ".venv" / "bin" / "python").exists()
                    else sys.executable,
                ),
            )
        )
        self.startup_timeout = float(args.get("unit_startup_timeout", 300))
        self.request_timeout = float(args.get("unit_request_timeout", 300))
        self.authkey = str(args.get("unit_authkey", os.environ.get("UNIT_AUTHKEY", "")))
        self._external = self.port != 0
        self._process: subprocess.Popen | None = None
        self._closed = False

        if not self._external:
            self.port = _find_free_port()
            self.authkey = secrets.token_hex(16)
            self._start_server()

        self.base_url = f"http://{self.host}:{self.port}"
        self._wait_until_ready()
        self._post_json("/init", {"authkey": self.authkey, "args": _to_wire_payload(args)})

    def _start_server(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(f"UniT python not found: {self.python}")
        self._process = subprocess.Popen(
            [
                str(self.python),
                str(UNIT_ROOT / "unit_server.py"),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--authkey",
                self.authkey,
            ],
            cwd=str(REPO_ROOT),
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                response = self._request("GET", "/health")
                if response.get("status") == "ok":
                    return
            except BaseException as exc:
                last_error = exc
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"UniT server exited with code {self._process.returncode}") from exc
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for UniT server: {last_error}")

    def _request(self, method: str, path: str, data: bytes | None = None, content_type: str | None = None):
        headers = {"X-UniT-Auth": self.authkey}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"UniT server HTTP {exc.code}: {detail}") from exc

    def _post_json(self, path: str, payload: Any):
        response = self._request(
            "POST",
            path,
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(
                "UniT server error:\n"
                f"{response.get('error')}\n"
                f"{response.get('traceback')}"
            )
        return response

    def get_action(self, observation: dict):
        import numpy as np

        response = self._post_json(
            "/act",
            {
                "authkey": self.authkey,
                "observation": _to_wire_payload(observation),
            },
        )
        return np.asarray(_from_wire_payload(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post_json("/reset", {"authkey": self.authkey})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._post_json("/shutdown" if not self._external else "/reset", {"authkey": self.authkey})
        except Exception:
            pass
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()


class Policy(BasePolicy):
    """Deploy-time UniT wrapper for UniVTAC via a FastAPI inference server."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.model = _UniTFastAPIClient(args)

    def encode_obs(self, observation: dict) -> dict:
        return observation

    def eval(self, task, observation):
        import torch

        action_np = self.model.get_action(observation).reshape(-1)
        action_t = torch.from_numpy(action_np).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def reset(self):
        self.model.reset()

    def close(self):
        self.model.close()
