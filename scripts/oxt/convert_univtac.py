#!/usr/bin/env python3
"""Convert UniVTAC HDF5 trajectories to the FTP-1/OXT Zarr contract.

Conversion is parallel at the HDF5 episode level.  Each worker writes an
independent staging Zarr; the parent process then merges episodes in a stable
order into one Zarr per task.  No two processes ever mutate the same store.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np


IMAGE_SIZE = 224
INSTRUCTION_PAD_LEN = 100
DEFAULT_GRIPPER_IDX = 28
TACTILE_SENSOR_NAME = "GelSightMini"
EXPECTED_FULL_DATASET_EPISODES = 800
REVIEWED_SOURCE_REVISION = "3d4646a7"
REVIEWED_SOURCE_STATUS_COUNTS = {"fail": 19, "missing": 18, "success": 763}
CONVERSION_SCHEMA_VERSION = "1.1.0"

INSTRUCTION_MAP = {
    "grasp_classify": "use the grasped tool for tactile sensing and move it to the target surface.",
    "insert_HDMI": "insert the HDMI connector into the fixed slot.",
    "insert_hole": "insert the peg into the hole.",
    "insert_tube": "insert the tube into the fixed slot.",
    "lift_can": "grasp the can and lift it vertically without slippage.",
    "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base near the wall.",
    "pull_out_key": "pull the key out of the slot.",
    "put_bottle_in_shelf": "grasp the bottle and place it into the shelf cavity.",
}

TACTILE_PATHS = {
    "left": (
        "tactile/left_gsmini/rgb_marker",
        "tactile/left_tactile/rgb_marker",
    ),
    "right": (
        "tactile/right_gsmini/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ),
}


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    task: str
    source: str
    source_relpath: str
    episode_id: str
    source_result: str | None
    source_seed: Any
    staging_path: str
    image_size: int
    downsample: int
    decode_batch_size: int
    resume: bool


def _require_zarr() -> tuple[Any, Any]:
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise SystemExit(
            "缺少转换依赖 zarr/numcodecs。请先运行：bash data/download.sh setup"
        ) from exc
    major = int(zarr.__version__.split(".", 1)[0])
    if major >= 3:
        raise SystemExit(
            f"当前 zarr={zarr.__version__}，转换器要求 zarr<3；"
            "请运行：python -m pip install 'zarr<3'"
        )
    return zarr, numcodecs


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    )


def _find_hdf5_dir(task_dir: Path) -> Path | None:
    for relative in (Path("clean"), Path("demo/hdf5")):
        candidate = task_dir / relative
        if candidate.is_dir() and any(candidate.glob("*.hdf5")):
            return candidate
    return None


def _read_task_metadata(hdf5_dir: Path) -> dict[str, Any]:
    candidates = (hdf5_dir / "metadata.json", hdf5_dir.parent / "metadata.json")
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError(f"metadata 必须是 JSON object: {path}")
            return data
    return {}


def discover_jobs(
    raw_dir: Path,
    output_dir: Path,
    selected_tasks: set[str] | None,
    episodes_per_task: int | None,
    image_size: int,
    downsample: int,
    decode_batch_size: int,
    resume: bool,
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    for task_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        task = task_dir.name
        if selected_tasks is not None and task not in selected_tasks:
            continue
        hdf5_dir = _find_hdf5_dir(task_dir)
        if hdf5_dir is None:
            continue
        metadata = _read_task_metadata(hdf5_dir)
        files = sorted(hdf5_dir.glob("*.hdf5"), key=_natural_key)
        if episodes_per_task is not None:
            files = files[:episodes_per_task]
        for source in files:
            entry = metadata.get(source.stem, {})
            if not isinstance(entry, dict):
                entry = {}
            raw_result = entry.get("result")
            source_result = str(raw_result) if raw_result is not None else None
            staging = output_dir / ".staging" / task / f"{source.stem}.zarr"
            jobs.append(
                EpisodeJob(
                    task=task,
                    source=str(source.resolve()),
                    source_relpath=str(source.relative_to(raw_dir)),
                    episode_id=source.stem,
                    source_result=source_result,
                    source_seed=entry.get("seed"),
                    staging_path=str(staging),
                    image_size=image_size,
                    downsample=downsample,
                    decode_batch_size=decode_batch_size,
                    resume=resume,
                )
            )

    if selected_tasks:
        found = {job.task for job in jobs}
        missing = sorted(selected_tasks - found)
        if missing:
            raise FileNotFoundError(f"没有找到这些 task 的 HDF5: {', '.join(missing)}")
    if not jobs:
        raise FileNotFoundError(
            f"{raw_dir} 下未找到 <task>/clean/*.hdf5 或 <task>/demo/hdf5/*.hdf5"
        )
    return jobs


def _pick_hdf5_path(handle: h5py.File, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in handle:
            return candidate
    raise KeyError(f"缺少 HDF5 key，尝试过: {', '.join(candidates)}")


def _decode_jpeg(value: Any, image_size: int) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        encoded = value
    else:
        encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG 解码失败")
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # UniVTAC collected Isaac RGB arrays with cv2.imencode directly.  Therefore
    # cv2.imdecode returns the original numeric RGB channel order.  The upstream
    # FTP-1 parser's BGR->RGB followed by [..., ::-1] has the same net result.
    return image.astype(np.uint8, copy=False)


def _decode_batch(dataset: h5py.Dataset, indices: np.ndarray, image_size: int) -> np.ndarray:
    return np.stack([_decode_jpeg(dataset[int(index)], image_size) for index in indices])


def _chunks(shape: tuple[int, ...], dtype: np.dtype[Any], target_bytes: int = 8 << 20) -> tuple[int, ...]:
    if not shape:
        return shape
    bytes_per_frame = max(1, int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize)
    time_chunk = max(1, min(shape[0], target_bytes // bytes_per_frame))
    return (time_chunk, *shape[1:])


def _write_array(group: Any, name: str, value: np.ndarray, compressor: Any) -> None:
    group.create_dataset(
        name,
        data=value,
        shape=value.shape,
        chunks=_chunks(value.shape, value.dtype),
        dtype=value.dtype,
        compressor=compressor,
        overwrite=True,
    )


def _write_stream(
    group: Any,
    name: str,
    dataset: h5py.Dataset,
    indices: np.ndarray,
    image_size: int,
    batch_size: int,
    compressor: Any,
) -> None:
    shape = (len(indices), image_size, image_size, 3)
    destination = group.create_dataset(
        name,
        shape=shape,
        chunks=_chunks(shape, np.dtype(np.uint8)),
        dtype=np.uint8,
        compressor=compressor,
        overwrite=True,
    )
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        destination[start:stop] = _decode_batch(dataset, indices[start:stop], image_size)


def _write_tactile_stream(
    group: Any,
    left: h5py.Dataset,
    right: h5py.Dataset,
    indices: np.ndarray,
    image_size: int,
    batch_size: int,
    compressor: Any,
) -> None:
    shape = (len(indices), 2, image_size, image_size, 3)
    destination = group.create_dataset(
        "right_tactile_data_gripper",
        shape=shape,
        chunks=_chunks(shape, np.dtype(np.uint8)),
        dtype=np.uint8,
        compressor=compressor,
        overwrite=True,
    )
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch_indices = indices[start:stop]
        left_batch = _decode_batch(left, batch_indices, image_size)
        right_batch = _decode_batch(right, batch_indices, image_size)
        destination[start:stop] = np.stack((left_batch, right_batch), axis=1)


def _completion_path(staging_path: Path) -> Path:
    return staging_path / ".univtac_complete.json"


def _read_completion(staging_path: Path) -> dict[str, Any] | None:
    marker = _completion_path(staging_path)
    if not marker.is_file():
        return None
    try:
        with marker.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def convert_episode(job: EpisodeJob) -> dict[str, Any]:
    zarr, numcodecs = _require_zarr()
    source = Path(job.source)
    staging = Path(job.staging_path)
    source_stat = source.stat()
    previous = _read_completion(staging) if job.resume else None
    if (
        previous
        and previous.get("conversion_schema_version") == CONVERSION_SCHEMA_VERSION
        and previous.get("source_size") == source_stat.st_size
        and previous.get("source_mtime_ns") == source_stat.st_mtime_ns
    ):
        return previous

    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )

    try:
        with h5py.File(source, "r") as handle:
            required = (
                "embodiment/joint",
                "observation/head/rgb",
                "observation/wrist/rgb",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise KeyError(f"缺少 HDF5 key: {', '.join(missing)}")

            joint = np.asarray(handle["embodiment/joint"], dtype=np.float32)
            if joint.ndim != 2 or joint.shape[1] < 8:
                raise ValueError(f"embodiment/joint 预期 (T, >=8)，实际 {joint.shape}")
            full_length = joint.shape[0]
            if full_length < 2:
                raise ValueError(f"轨迹至少需要 2 帧，实际 {full_length}")
            indices = np.arange(0, full_length - 1, job.downsample, dtype=np.int64)
            length = len(indices)

            left_path = _pick_hdf5_path(handle, TACTILE_PATHS["left"])
            right_path = _pick_hdf5_path(handle, TACTILE_PATHS["right"])
            stream_paths = (
                "observation/head/rgb",
                "observation/wrist/rgb",
                left_path,
                right_path,
            )
            bad_lengths = {
                name: len(handle[name])
                for name in stream_paths
                if len(handle[name]) != full_length
            }
            if bad_lengths:
                raise ValueError(
                    f"图像流长度与 joint({full_length}) 不一致: {bad_lengths}"
                )

            root = zarr.open_group(str(staging), mode="w")
            data = root.create_group("data", overwrite=True)
            meta = root.create_group("meta", overwrite=True)

            selected_joint = joint[indices, :8]
            _write_array(data, "timestamps", np.arange(length, dtype=np.int64), compressor)
            _write_array(data, "right_arm_joints", selected_joint[:, :7], compressor)
            _write_array(data, "right_hand_joints", selected_joint[:, 7:8], compressor)
            _write_array(
                data,
                "right_hand_joints_idx",
                np.full((length, 1), DEFAULT_GRIPPER_IDX, dtype=np.int32),
                compressor,
            )
            _write_array(
                data,
                "right_tactile_area_gripper",
                np.tile(np.asarray([[0, 1]], dtype=np.int64), (length, 1)),
                compressor,
            )
            _write_array(
                data,
                "right_tactile_sensor_gripper",
                np.asarray([TACTILE_SENSOR_NAME] * length),
                compressor,
            )
            _write_array(
                data,
                "right_tactile_type_gripper",
                np.asarray(["image"] * length),
                compressor,
            )
            instruction = INSTRUCTION_MAP.get(
                job.task, f"complete the manipulation task {job.task}."
            ).ljust(INSTRUCTION_PAD_LEN)
            _write_array(
                data,
                "sub_task_instruction",
                np.asarray([instruction] * length),
                compressor,
            )
            _write_stream(
                data,
                "camera_main_rgb",
                handle["observation/head/rgb"],
                indices,
                job.image_size,
                job.decode_batch_size,
                compressor,
            )
            _write_stream(
                data,
                "right_wrist_camera_rgb",
                handle["observation/wrist/rgb"],
                indices,
                job.image_size,
                job.decode_batch_size,
                compressor,
            )
            _write_tactile_stream(
                data,
                handle[left_path],
                handle[right_path],
                indices,
                job.image_size,
                job.decode_batch_size,
                compressor,
            )
            _write_array(meta, "episode_ends", np.asarray([length], dtype=np.int64), compressor)
            _write_array(meta, "episode_lengths", np.asarray([length], dtype=np.int64), compressor)
            seed = int(job.source_seed) if job.source_seed is not None else -1
            _write_array(meta, "episode_seeds", np.asarray([seed], dtype=np.int64), compressor)

        result = {
            "task": job.task,
            "episode_id": job.episode_id,
            "source": job.source_relpath,
            "seed": job.source_seed,
            "source_frames": full_length,
            "frames": length,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "conversion_schema_version": CONVERSION_SCHEMA_VERSION,
            "staging_path": str(staging),
        }
        with _completion_path(staging).open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _copy_array(source: Any, destination: Any, offset: int, batch_size: int = 64) -> None:
    for start in range(0, source.shape[0], batch_size):
        stop = min(start + batch_size, source.shape[0])
        destination[offset + start : offset + stop] = source[start:stop]


def aggregate_task(
    task: str,
    episodes: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> Path:
    zarr, numcodecs = _require_zarr()
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    episodes = sorted(episodes, key=lambda item: _natural_key(Path(item["episode_id"])))
    total_frames = sum(int(item["frames"]) for item in episodes)
    target = output_dir / "zarr" / f"{task}_head_wrist.zarr"
    partial = output_dir / ".aggregate" / f"{task}_head_wrist.zarr.partial"
    if target.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；确认重建时加 --overwrite: {target}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)

    first = zarr.open_group(episodes[0]["staging_path"], mode="r")["data"]
    keys = sorted(first.array_keys())
    root = zarr.open_group(str(partial), mode="w")
    destination_data = root.create_group("data", overwrite=True)
    destination_meta = root.create_group("meta", overwrite=True)
    destinations: dict[str, Any] = {}
    for key in keys:
        source_array = first[key]
        shape = (total_frames, *source_array.shape[1:])
        destinations[key] = destination_data.create_dataset(
            key,
            shape=shape,
            chunks=_chunks(shape, np.dtype(source_array.dtype)),
            dtype=source_array.dtype,
            compressor=compressor,
            overwrite=True,
        )

    offset = 0
    episode_ends: list[int] = []
    episode_lengths: list[int] = []
    episode_seeds: list[int] = []
    for episode in episodes:
        source_data = zarr.open_group(episode["staging_path"], mode="r")["data"]
        if sorted(source_data.array_keys()) != keys:
            raise ValueError(f"episode keys 不一致: {episode['source']}")
        length = int(episode["frames"])
        for key in keys:
            source_array = source_data[key]
            destination = destinations[key]
            if source_array.shape[1:] != destination.shape[1:] or source_array.dtype != destination.dtype:
                raise ValueError(f"{episode['source']} 的 {key} shape/dtype 不一致")
            _copy_array(source_array, destination, offset)
        offset += length
        episode_ends.append(offset)
        episode_lengths.append(length)
        episode_seeds.append(int(episode["seed"]) if episode["seed"] is not None else -1)
    _write_array(
        destination_meta,
        "episode_ends",
        np.asarray(episode_ends, dtype=np.int64),
        compressor,
    )
    _write_array(
        destination_meta,
        "episode_lengths",
        np.asarray(episode_lengths, dtype=np.int64),
        compressor,
    )
    _write_array(
        destination_meta,
        "episode_seeds",
        np.asarray(episode_seeds, dtype=np.int64),
        compressor,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)
    return target


def _status_counts(jobs: list[EpisodeJob]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        key = job.source_result if job.source_result is not None else "missing"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_manifest(
    output_dir: Path,
    jobs: list[EpisodeJob],
    results: list[dict[str, Any]],
    source_dataset: str,
    source_revision: str,
    image_size: int,
    downsample: int,
) -> Path:
    task_summary: dict[str, dict[str, int]] = {}
    for result in results:
        item = task_summary.setdefault(result["task"], {"trajectory": 0, "frame": 0})
        item["trajectory"] += 1
        item["frame"] += int(result["frames"])
    manifest = {
        "format": "UniVTAC-to-FTP1/OXT",
        "format_version": CONVERSION_SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "selection": (
            "All trajectories from the pinned, reviewed 800-episode release are retained. "
            "Legacy result fields are intentionally not propagated."
        ),
        "mapping": {
            "observation/head/rgb": "camera_main_rgb",
            "observation/wrist/rgb": "right_wrist_camera_rgb",
            "embodiment/joint[:7]": "right_arm_joints",
            "embodiment/joint[7:8]": "right_hand_joints",
            "left/right tactile rgb_marker": "right_tactile_data_gripper[:, 0/1]",
            "omitted": [
                "embodiment/ee (pose convention is not verified against FTP-1)",
                "actor/*",
                "raw tactile depth/marker/pose/rgb",
            ],
        },
        "parameters": {"image_size": image_size, "downsample": downsample},
        "summary": {
            "task": len(task_summary),
            "trajectory": len(results),
            "frame": sum(int(item["frames"]) for item in results),
            "tasks": dict(sorted(task_summary.items())),
        },
        "episodes": sorted(
            [
                {
                    key: result[key]
                    for key in ("task", "episode_id", "source", "seed", "source_frames", "frames")
                }
                for result in results
            ],
            key=lambda item: (item["task"], _natural_key(Path(item["episode_id"]))),
        ),
    }
    path = output_dir / "conversion_manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="并行地将 UniVTAC HDF5 转换为 FTP-1/OXT Zarr。"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--tasks", help="逗号分隔的 task 子集；用于调试")
    parser.add_argument("--episodes-per-task", type=int, help="每 task 上限；用于调试")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=EXPECTED_FULL_DATASET_EPISODES,
        help="全量转换的预期轨迹数；0 表示不检查",
    )
    parser.add_argument("--source-dataset", default="byml2024/UniVTAC")
    parser.add_argument("--source-revision", default=REVIEWED_SOURCE_REVISION)
    parser.add_argument(
        "--accept-unreviewed-revision",
        action="store_true",
        help="明确接受未复核的 source revision；其 result 归一化需由操作者负责",
    )
    parser.add_argument("--overwrite", action="store_true", help="重建已存在的最终 task Zarr")
    parser.add_argument("--no-resume", action="store_true", help="不复用已完成的 episode staging")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只发现和校验输入，不转换")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1 or args.decode_batch_size < 1 or args.downsample < 1:
        raise SystemExit("workers、decode-batch-size、downsample 必须大于 0")
    selected_tasks = (
        {item.strip() for item in args.tasks.split(",") if item.strip()}
        if args.tasks
        else None
    )
    output_dir = args.output_dir.resolve()
    jobs = discover_jobs(
        args.raw_dir.resolve(),
        output_dir,
        selected_tasks,
        args.episodes_per_task,
        args.image_size,
        args.downsample,
        args.decode_batch_size,
        not args.no_resume,
    )
    full_conversion = selected_tasks is None and args.episodes_per_task is None
    if full_conversion:
        if args.source_revision != REVIEWED_SOURCE_REVISION and not args.accept_unreviewed_revision:
            raise SystemExit(
                f"source revision {args.source_revision!r} 未经过本次 800 条轨迹复核。"
                f"请使用 {REVIEWED_SOURCE_REVISION!r}；确已重新复核时加 "
                "--accept-unreviewed-revision。"
            )
        if args.expected_episodes and len(jobs) != args.expected_episodes:
            raise SystemExit(
                f"全量数据预期 {args.expected_episodes} 个 HDF5，实际 {len(jobs)}。"
                "这可能是下载不完整或上游版本已变化；复核后可用 --expected-episodes 0。"
            )
        counts = _status_counts(jobs)
        if (
            args.source_revision == REVIEWED_SOURCE_REVISION
            and counts != REVIEWED_SOURCE_STATUS_COUNTS
        ):
            raise SystemExit(
                f"revision {REVIEWED_SOURCE_REVISION} 的 metadata 分布应为 "
                f"{REVIEWED_SOURCE_STATUS_COUNTS}，实际 {counts}。请检查下载完整性和 metadata。"
            )

    print(
        json.dumps(
            {
                "raw_dir": str(args.raw_dir.resolve()),
                "output_dir": str(output_dir),
                "tasks": sorted({job.task for job in jobs}),
                "episodes": len(jobs),
                "workers": args.workers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    _require_zarr()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(convert_episode, job): job for job in jobs}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] {job.task}/{job.episode_id}: "
                    f"{result['frames']} frames",
                    flush=True,
                )
            except Exception as exc:  # keep other independent jobs running
                failures.append(
                    {
                        "source": job.source_relpath,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"FAILED {job.source_relpath}: {exc}", file=sys.stderr, flush=True)

    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "conversion_failures.json"
        with failure_path.open("w", encoding="utf-8") as file:
            json.dump(failures, file, ensure_ascii=False, indent=2)
            file.write("\n")
        raise SystemExit(
            f"{len(failures)} 个 HDF5 转换失败；staging 已保留，详情见 {failure_path}"
        )

    by_task: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_task.setdefault(result["task"], []).append(result)
    for task, task_results in sorted(by_task.items()):
        path = aggregate_task(task, task_results, output_dir, args.overwrite)
        print(f"merged {len(task_results)} episodes -> {path}", flush=True)

    manifest = _write_manifest(
        output_dir,
        jobs,
        results,
        args.source_dataset,
        args.source_revision,
        args.image_size,
        args.downsample,
    )
    if not args.keep_staging:
        shutil.rmtree(output_dir / ".staging", ignore_errors=True)
        shutil.rmtree(output_dir / ".aggregate", ignore_errors=True)
    print(f"manifest -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
