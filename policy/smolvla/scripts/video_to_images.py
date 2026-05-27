#!/usr/bin/env python
"""Convert a video-backed LeRobot dataset to an image-backed dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import shutil
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

POLICY_ROOT = Path(__file__).resolve().parents[1]
SMOLVLA_ROOT = POLICY_ROOT

import sys

sys.path.insert(0, str(SMOLVLA_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.compute_stats import (  # noqa: E402
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.feature_utils import validate_episode_buffer  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames_torchcodec  # noqa: E402

LOG = logging.getLogger(__name__)

AUTO_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index"}


def _to_plain_value(value):
    if isinstance(value, torch.Tensor):
        value = value.cpu()
        if value.ndim == 0:
            return value.item()
        return value.numpy()
    return value


def _to_pil_image(image: torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return Image.fromarray(image)


def _image_features_from_video(features: dict) -> dict:
    converted = {}
    for key, feature in features.items():
        converted[key] = dict(feature)
        if feature.get("dtype") == "video":
            converted[key]["dtype"] = "image"
            converted[key]["info"] = None
    return converted


def _new_episode_buffer(features: dict, episode_index: int) -> dict:
    buffer = {"size": 0, "task": []}
    for key in features:
        buffer[key] = episode_index if key == "episode_index" else []
    return buffer


def _decode_video_segment_pyav(
    video_path: Path,
    start_timestamp: float,
    num_frames: int,
    fps: int,
    tolerance_s: float,
) -> torch.Tensor:
    target_timestamps = start_timestamp + np.arange(num_frames, dtype=np.float64) / fps
    first_ts = float(target_timestamps[0])
    last_ts = float(target_timestamps[-1])
    decoded_frames: list[torch.Tensor] = []
    decoded_timestamps: list[float] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts * av.time_base), backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            arr = frame.to_ndarray(format="rgb24")
            decoded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            decoded_timestamps.append(timestamp)
            if timestamp >= last_ts:
                break

    if not decoded_frames:
        raise RuntimeError(f"No frames decoded from {video_path} around timestamp {first_ts:.6f}")

    decoded_ts = np.asarray(decoded_timestamps, dtype=np.float64)
    nearest_indices = np.searchsorted(decoded_ts, target_timestamps)
    selected_indices = []
    max_error = 0.0

    for target, right_idx in zip(target_timestamps, nearest_indices, strict=True):
        candidates = []
        if right_idx < len(decoded_ts):
            candidates.append(int(right_idx))
        if right_idx > 0:
            candidates.append(int(right_idx - 1))
        best_idx = min(candidates, key=lambda idx: abs(decoded_ts[idx] - target))
        error = abs(decoded_ts[best_idx] - target)
        max_error = max(max_error, error)
        selected_indices.append(best_idx)

    if max_error > tolerance_s:
        raise RuntimeError(
            f"Decoded frame timestamp error {max_error:.6f}s exceeds tolerance {tolerance_s:.6f}s "
            f"for {video_path}"
        )

    return torch.stack([decoded_frames[idx] for idx in selected_indices])


def _decode_video_segment(
    video_path: Path,
    start_timestamp: float,
    num_frames: int,
    fps: int,
    tolerance_s: float,
    backend: str,
) -> torch.Tensor:
    if backend == "torchcodec":
        timestamps = (start_timestamp + np.arange(num_frames, dtype=np.float64) / fps).tolist()
        return decode_video_frames_torchcodec(
            video_path,
            timestamps,
            tolerance_s,
            return_uint8=True,
        )
    if backend == "pyav":
        return _decode_video_segment_pyav(
            video_path=video_path,
            start_timestamp=start_timestamp,
            num_frames=num_frames,
            fps=fps,
            tolerance_s=tolerance_s,
        )
    raise ValueError(f"Unsupported video backend for batch decoding: {backend}")


def _decode_episode_chunk(
    src: LeRobotDataset,
    ep_idx: int,
    chunk_start: int,
    chunk_size: int,
    decode_workers: int,
    video_backend: str,
) -> dict[str, torch.Tensor]:
    def decode_one(video_key: str) -> tuple[str, torch.Tensor]:
        ep = src.meta.episodes[ep_idx]
        from_timestamp = float(ep[f"videos/{video_key}/from_timestamp"])
        video_path = src.root / src.meta.get_video_file_path(ep_idx, video_key)
        frames = _decode_video_segment(
            video_path=video_path,
            start_timestamp=from_timestamp + chunk_start / src.fps,
            num_frames=chunk_size,
            fps=int(src.fps),
            tolerance_s=src.tolerance_s,
            backend=video_backend,
        )
        return video_key, frames

    video_keys = list(src.meta.video_keys)
    if decode_workers <= 1 or len(video_keys) <= 1:
        return dict(decode_one(video_key) for video_key in video_keys)

    max_workers = min(decode_workers, len(video_keys))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(decode_one, video_key) for video_key in video_keys]
        return dict(future.result() for future in concurrent.futures.as_completed(futures))


def _append_direct_chunk(
    src: LeRobotDataset,
    features: dict,
    episode_buffer: dict,
    from_idx: int,
    chunk_start: int,
    decoded_images: dict[str, torch.Tensor],
) -> None:
    chunk_size = next(iter(decoded_images.values())).shape[0]
    for local_idx in range(chunk_size):
        frame_idx = from_idx + chunk_start + local_idx
        item = src.get_raw_item(frame_idx)
        task_idx = int(_to_plain_value(item["task_index"]))

        episode_buffer["task"].append(src.meta.tasks.iloc[task_idx].name)
        episode_buffer["frame_index"].append(local_idx + chunk_start)
        episode_buffer["timestamp"].append((local_idx + chunk_start) / src.fps)

        for key in features:
            if key in AUTO_KEYS:
                continue
            if key in decoded_images:
                episode_buffer[key].append(_to_pil_image(decoded_images[key][local_idx]))
            elif key in item:
                episode_buffer[key].append(_to_plain_value(item[key]))

        episode_buffer["size"] += 1


def _append_staged_chunk(
    src: LeRobotDataset,
    dst: LeRobotDataset,
    features: dict,
    from_idx: int,
    chunk_start: int,
    decoded_images: dict[str, torch.Tensor],
) -> None:
    chunk_size = next(iter(decoded_images.values())).shape[0]
    for local_idx in range(chunk_size):
        frame_idx = from_idx + chunk_start + local_idx
        item = src.get_raw_item(frame_idx)
        task_idx = int(_to_plain_value(item["task_index"]))
        frame = {"task": src.meta.tasks.iloc[task_idx].name}
        for key in features:
            if key in AUTO_KEYS:
                continue
            if key in decoded_images:
                frame[key] = decoded_images[key][local_idx]
            elif key in item:
                frame[key] = _to_plain_value(item[key])
        dst.add_frame(frame)


def _compute_direct_episode_stats(episode_buffer: dict, features: dict) -> dict:
    stats = {}
    for key, feature in features.items():
        if key in {"index", "episode_index", "task_index"}:
            data = episode_buffer[key]
        else:
            data = episode_buffer.get(key)
        if data is None or feature["dtype"] in {"string", "language"}:
            continue

        if feature["dtype"] in {"image", "video"}:
            sampled = []
            for idx in sample_indices(len(data)):
                image = np.asarray(data[idx], dtype=np.uint8).transpose(2, 0, 1)
                sampled.append(auto_downsample_height_width(image))
            array = np.stack(sampled)
            feature_stats = get_feature_stats(array, axis=(0, 2, 3), keepdims=True)
            stats[key] = {
                stat_key: stat_value if stat_key == "count" else np.squeeze(stat_value / 255.0, axis=0)
                for stat_key, stat_value in feature_stats.items()
            }
            continue

        if not isinstance(data, np.ndarray):
            data = np.stack(data)
        if tuple(feature["shape"]) == (1,) and feature["dtype"] != "string":
            data = data.reshape(-1)
        stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return stats


def _save_direct_episode(dst: LeRobotDataset, episode_buffer: dict, features: dict) -> None:
    validate_episode_buffer(episode_buffer, dst.meta.total_episodes, features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(dst.meta.total_frames, dst.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    dst.meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array([dst.meta.get_task_index(task) for task in tasks])

    for key, feature in features.items():
        if key in {"index", "episode_index", "task_index"} or feature["dtype"] in {"image", "video"}:
            continue
        stacked_values = np.stack(episode_buffer[key])
        if tuple(feature["shape"]) == (1,) and feature["dtype"] != "string":
            stacked_values = stacked_values.reshape(episode_length)
        episode_buffer[key] = stacked_values

    episode_stats = _compute_direct_episode_stats(episode_buffer, features)
    episode_metadata = dst.writer._save_episode_data(episode_buffer)
    dst.meta.save_episode(episode_index, episode_length, episode_tasks, episode_stats, episode_metadata)


def convert_video_dataset_to_images(
    src_root: Path,
    dst_root: Path,
    repo_id: str,
    video_backend: str | None = None,
    overwrite: bool = False,
    image_writer_processes: int = 0,
    image_writer_threads: int = 8,
    frames_per_chunk: int = 512,
    decode_workers: int = 0,
    storage_mode: str = "direct",
) -> Path:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)

    src = LeRobotDataset(
        repo_id=repo_id,
        root=src_root,
        video_backend=video_backend or "pyav",
        return_uint8=True,
    )
    if not src.meta.video_keys:
        raise ValueError(f"Source dataset has no video features: {src_root}")
    if frames_per_chunk <= 0:
        raise ValueError(f"frames_per_chunk must be positive, got {frames_per_chunk}")
    if storage_mode not in {"direct", "staged"}:
        raise ValueError(f"storage_mode must be 'direct' or 'staged', got {storage_mode!r}")

    if decode_workers <= 0:
        decode_workers = len(src.meta.video_keys)
    decode_backend = video_backend or "pyav"

    features = _image_features_from_video(src.meta.features)
    dst = LeRobotDataset.create(
        repo_id=f"{repo_id}_images",
        fps=src.fps,
        features=features,
        root=dst_root,
        robot_type=src.meta.robot_type,
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        data_files_size_in_mb=src.meta.data_files_size_in_mb,
        video_backend="pyav",
    )

    try:
        for ep_idx in tqdm(range(src.meta.total_episodes), desc="Converting episodes"):
            from_idx = int(src.meta.episodes["dataset_from_index"][ep_idx])
            to_idx = int(src.meta.episodes["dataset_to_index"][ep_idx])
            episode_length = to_idx - from_idx
            episode_buffer = _new_episode_buffer(features, dst.meta.total_episodes)
            for chunk_start in tqdm(
                range(0, episode_length, frames_per_chunk),
                desc=f"episode {ep_idx}",
                leave=False,
            ):
                chunk_size = min(frames_per_chunk, episode_length - chunk_start)
                decoded_images = _decode_episode_chunk(
                    src,
                    ep_idx,
                    chunk_start,
                    chunk_size,
                    decode_workers,
                    decode_backend,
                )
                if storage_mode == "direct":
                    _append_direct_chunk(src, features, episode_buffer, from_idx, chunk_start, decoded_images)
                else:
                    _append_staged_chunk(src, dst, features, from_idx, chunk_start, decoded_images)

            if storage_mode == "direct":
                _save_direct_episode(dst, episode_buffer, features)
            else:
                dst.save_episode()
    finally:
        dst.finalize()

    source_metadata = src_root / "source_metadata.json"
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text())
        metadata["converted_from"] = str(src_root)
        metadata["storage"] = "images"
        (dst_root / "source_metadata.json").write_text(json.dumps(metadata, indent=4))

    LOG.info("Wrote image-backed dataset to %s", dst_root)
    return dst_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True, help="Existing LeRobot dataset root.")
    parser.add_argument("--dst-root", type=Path, required=True, help="Output image-backed dataset root.")
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo_id.")
    parser.add_argument("--video-backend", type=str, default=None, help="pyav or torchcodec.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dst-root if it exists.")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=512,
        help="Frames decoded per episode/camera chunk. Increase for speed, decrease to reduce RAM.",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help="Parallel camera decoders per chunk. 0 means one worker per video key.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=("direct", "staged"),
        default="direct",
        help="direct embeds images to parquet from memory; staged writes temporary PNGs first.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    convert_video_dataset_to_images(
        src_root=args.src_root,
        dst_root=args.dst_root,
        repo_id=args.repo_id,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        frames_per_chunk=args.frames_per_chunk,
        decode_workers=args.decode_workers,
        storage_mode=args.storage_mode,
    )


if __name__ == "__main__":
    main()
