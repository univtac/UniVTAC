#!/usr/bin/env python
# Copyright 2026 The vitac_smolvla team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Convert raw visuo-tactile HDF5 episodes into a LeRobot v3.0 dataset.

Expected input layout (see ``/data/lift_bottle/clean/*.hdf5``)::

    actor/{...}                    (unused)
    embodiment/ee                  (T, 7) float32
    embodiment/joint               (T, 9) float32
    observation/head/rgb           (T,)   bytes   -- JPEG-encoded RGB
    observation/wrist/rgb          (T,)   bytes
    tactile/left_gsmini/rgb        (T,)   bytes
    tactile/left_gsmini/rgb_marker (T,)   bytes
    tactile/right_gsmini/rgb       (T,)   bytes
    tactile/right_gsmini/rgb_marker(T,)   bytes
    step                           (T,)

Output: a LeRobot v3.0 dataset under
``<root>/data/<task_name>_lerobot`` with the following feature schema:

    observation.images.head            video
    observation.images.wrist           video
    observation.images.tactile_ll      video  (left arm, left gripper)
    observation.images.tactile_lr      video  (left arm, right gripper)
    observation.images.tactile_ll_marker  video
    observation.images.tactile_lr_marker  video
    observation.state                  float32 (16,)  -- concat(ee[7], joint[9])
    action                             float32 (9,)   -- next-step joint pose

The "ll" / "lr" naming reserves space for a future dual-arm setting where
the second letter denotes the arm and the third the gripper.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# Make src/ importable when this script is invoked from the repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from UniVTAC.policy.smolvla.src.lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocess_hdf5")


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def _decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode a single JPEG byte-string to an HWC uint8 RGB array."""
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _probe_image_shape(dataset: h5py.Dataset) -> tuple[int, int]:
    """Return (H, W) of the first JPEG-encoded frame."""
    sample = bytes(dataset[0])
    arr = _decode_jpeg(sample)
    return arr.shape[0], arr.shape[1]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

VIDEO_KEYS = {
    "observation.images.head":               ("observation", "head", "rgb"),
    "observation.images.wrist":              ("observation", "wrist", "rgb"),
    "observation.images.tactile_ll":         ("tactile", "left_gsmini", "rgb"),
    "observation.images.tactile_lr":         ("tactile", "right_gsmini", "rgb"),
    "observation.images.tactile_ll_marker":  ("tactile", "left_gsmini", "rgb_marker"),
    "observation.images.tactile_lr_marker":  ("tactile", "right_gsmini", "rgb_marker"),
}


def _build_feature_schema(sample_shapes: dict[str, tuple[int, int]],
                          state_dim: int, action_dim: int) -> dict:
    features: dict = {}
    for key, (h, w) in sample_shapes.items():
        features[key] = {
            "dtype": "video",
            "shape": (3, h, w),
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


def _task_string_from_name(task_name: str) -> str:
    """Convert ``lift_bottle`` -> ``lift the bottle``."""
    words = task_name.replace("-", "_").split("_")
    if len(words) >= 2:
        return f"{words[0]} the {' '.join(words[1:])}"
    return task_name.replace("_", " ")


def convert_hdf5_dir(
    src_dir: Path,
    repo_id: str,
    out_root: Path,
    fps: int = 30,
    task_text: str | None = None,
) -> None:
    """Convert all ``*.hdf5`` files in ``src_dir`` into a single LeRobot dataset."""
    hdf5_files = sorted(src_dir.glob("**/*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files under {src_dir}")

    task_text = task_text or _task_string_from_name(src_dir.name)
    log.info("Found %d hdf5 file(s) in %s (task: %r)", len(hdf5_files), src_dir, task_text)

    # Probe the first file for image shapes and vector dims.
    with h5py.File(hdf5_files[0], "r") as f:
        sample_shapes = {}
        for k, path in VIDEO_KEYS.items():
            ds = f
            for p in path:
                ds = ds[p]
            sample_shapes[k] = _probe_image_shape(ds)
        state_dim = f["embodiment/ee"].shape[1] + f["embodiment/joint"].shape[1]
        action_dim = f["embodiment/joint"].shape[1]
        log.info("Image shapes: %s", sample_shapes)
        log.info("state_dim=%d action_dim=%d", state_dim, action_dim)

    features = _build_feature_schema(sample_shapes, state_dim, action_dim)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type="vitac_arm_left",
        use_videos=True,
    )

    for ep_idx, hdf5_path in enumerate(hdf5_files):
        log.info("[ep %02d] %s", ep_idx, hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            ee = f["embodiment/ee"][:].astype(np.float32)     # (T, 7)
            joint = f["embodiment/joint"][:].astype(np.float32)  # (T, 9)
            T = ee.shape[0]
            # Pre-fetch image byte arrays as lists.
            jpeg_streams: dict[str, np.ndarray] = {}
            for key, path in VIDEO_KEYS.items():
                node = f
                for p in path:
                    node = node[p]
                jpeg_streams[key] = node[:]

            # Action target = next-step joint pose; last frame repeats.
            next_joint = np.concatenate([joint[1:], joint[-1:]], axis=0)

            for t in range(T):
                frame = {
                    "observation.state": np.concatenate([ee[t], joint[t]]).astype(np.float32),
                    "action": next_joint[t].astype(np.float32),
                    "task": task_text,
                }
                for key in VIDEO_KEYS:
                    frame[key] = _decode_jpeg(bytes(jpeg_streams[key][t]))
                dataset.add_frame(frame)

            dataset.save_episode()

    dataset.finalize()
    log.info("Done. Wrote dataset to %s", out_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Source folder containing .hdf5 episodes (recursive).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output LeRobot v3.0 dataset root directory.")
    p.add_argument("--repo-id", type=str, default=None,
                   help="Dataset repo_id, e.g. 'local/lift_bottle_vitac'. Defaults to <src.name>_lerobot.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", type=str, default=None,
                   help="Task description string. Defaults inferred from folder name.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_id = args.repo_id or f"local/{args.src.name}_vitac"
    convert_hdf5_dir(
        src_dir=args.src,
        repo_id=repo_id,
        out_root=args.out,
        fps=args.fps,
        task_text=args.task,
    )


if __name__ == "__main__":
    main()
