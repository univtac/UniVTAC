#!/usr/bin/env python
# Copyright 2026 The UniVTAC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export raw UniVTAC visuo-tactile HDF5 episodes to the TLabel format.

TLabel (https://github.com/liesliy/tlabel, ``pip install tlabel``) is a unified
annotation schema for robot tactile data -- the tactile counterpart of COCO in
vision.  This preprocessor subclasses :class:`BaseDataPreprocessor`
(``policy/_base_data_preprocessor.py``), reuses its data loading pipeline
(episode discovery under ``data/{task}/{config}/``, episode selection,
down-sampling and the new/old tactile key fallback) and replaces the ACT-style
HDF5 export with TLabel JSON export.

One ``.tlabel.json`` file is produced per tactile sensor per episode.  The
marker count is *always* inferred dynamically from the ``marker`` dataset
shape ``(T, 2, marker_size, 2)`` -- it is never hard-coded (63 for the new
GelSight Mini collector format, 1200 for the old ModelScope releases, 81/220
for GF225/XenseWS, ...).

tlabel is an *optional* dependency: it is imported lazily, inside the export
methods, so that merely importing this module does not require it.  Install it
with::

    pip install tlabel        # h5py/numpy/opencv are already UniVTAC deps

Example::

    python -m policy.tlabel_data_preprocessor insert_hole demo 50 \\
        --output ./data/tlabel/insert_hole-demo

See ``docs/TLabelExport.md`` for the supported input fields, the output schema
and the HDF5 compatibility notes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make the repo root / policy dir importable, exactly like the other
# preprocessors under policy/ (e.g. policy/smolvla/scripts/process_data.py).
# The base class lives in policy/_base_data_preprocessor.py and is imported as
# a top-level module from the policy/ directory.
# ---------------------------------------------------------------------------
POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tlabel_preprocess")

# New (current collector) vs old (ModelScope releases) tactile group names.
TACTILE_GROUP_CANDIDATES = ("{side}_tactile", "{side}_gsmini")

# Human-readable profiles for the tactile groups that may appear in UniVTAC
# HDF5 files.  marker_count is deliberately NOT stored here: it is inferred at
# runtime from the marker array shape (see _read_marker_count), because the
# count differs across sensors/versions (GelSight Mini 7x9=63, GF225 9x9=81,
# XenseWS 11x20=220, old ModelScope UniVTAC data 1200).
_TLABEL_SENSOR_PROFILES = {
    "left_tactile": {"name": "GelSight Mini Left", "manufacturer": "GelSight Inc.", "frame_rate": 30.0},
    "right_tactile": {"name": "GelSight Mini Right", "manufacturer": "GelSight Inc.", "frame_rate": 30.0},
    "left_gsmini": {"name": "GelSight Mini Left", "manufacturer": "GelSight Inc.", "frame_rate": 30.0},
    "right_gsmini": {"name": "GelSight Mini Right", "manufacturer": "GelSight Inc.", "frame_rate": 30.0},
}


def _import_base_preprocessor():
    """Import BaseDataPreprocessor, with an actionable error for light envs.

    Importing ``policy/_base_data_preprocessor.py`` pulls in
    ``envs.utils.data``, which imports torch (and, in a full Isaac checkout,
    heavier stacks).  In a normal UniVTAC installation these are present; when
    the preprocessor is used in a minimal environment (tlabel + h5py + numpy)
    we surface a clear message instead of an opaque traceback.
    """
    try:
        from _base_data_preprocessor import BaseDataPreprocessor
        return BaseDataPreprocessor
    except ImportError as exc:
        raise ImportError(
            "Failed to import policy/_base_data_preprocessor.py. In a full "
            "UniVTAC checkout this requires torch and opencv "
            "(see docs/Installation.md). To run TLabel export in a minimal "
            "environment, install torch/opencv or use the base pipeline from "
            "the UniVTAC repo. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _require_tlabel():
    """Import tlabel lazily; raise an actionable error if it is missing."""
    try:
        import tlabel  # noqa: F401
        from tlabel.adapters.univtac import UniVTACAdapter  # noqa: F401
        return tlabel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TLabel export requires the optional 'tlabel' package. "
            "Install it with `pip install tlabel` and re-run. "
            f"(import error: {exc})"
        ) from exc


BaseDataPreprocessor = _import_base_preprocessor()


class TLabelDataPreprocessor(BaseDataPreprocessor):
    """Export UniVTAC raw HDF5 episodes to the TLabel annotation format.

    Inherits :class:`BaseDataPreprocessor` and reuses its data loading
    interface unchanged (``load_data`` -> ``HDF5Handler.batch_gather_hdf5``,
    including the automatic ``*_tactile`` -> ``*_gsmini`` key fallback).  The
    tactile feature extraction (contact / deformation / force / slip / shear /
    optical flow / ...) and the TLabel schema construction are delegated to the
    upstream ``tlabel.adapters.univtac.UniVTACAdapter`` shipped with the
    tlabel package, so the conversion logic stays canonical and maintained in
    one place.
    """

    def __init__(self, task_name: str, collect_config_name: str):
        super().__init__(task_name, collect_config_name)
        # {side: tactile group name actually present}, e.g. {'left': 'left_tactile'}
        self.tactile_group_map: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Data loading -- reuse the base pipeline verbatim
    # ------------------------------------------------------------------ #
    def load_data(
        self,
        visual_cameras=("head",),
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        episode_num=50,
        random_select=False,
    ):
        """Load episodes via ``BaseDataPreprocessor.load_data``.

        Reuses the base implementation as-is (episode discovery, selection,
        down-sampling, and the new/old tactile key fallback) and additionally
        records which tactile group exists in the data for the export step.
        """
        data = super().load_data(
            visual_cameras=list(visual_cameras),
            tactile_cameras=list(tactile_cameras),
            downsample_factor=downsample_factor,
            episode_num=episode_num,
            random_select=random_select,
        )
        self.tactile_group_map = self._resolve_tactile_groups(
            self.selected_raw_hdf5_paths[0], list(tactile_cameras)
        )
        return data

    @staticmethod
    def _resolve_tactile_groups(hdf5_path, tactile_cameras) -> dict[str, str]:
        """Return ``{side: tactile_group_name}`` present in the HDF5 file.

        Mirrors the probe in ``BaseDataPreprocessor.load_data``: prefer
        ``{side}_tactile``, fall back to ``{side}_gsmini``.
        """
        groups: dict[str, str] = {}
        with h5py.File(str(hdf5_path), "r") as f:
            for side in tactile_cameras:
                for candidate in TACTILE_GROUP_CANDIDATES:
                    group = candidate.format(side=side)
                    if f"tactile/{group}/depth" in f and f"tactile/{group}/marker" in f:
                        groups[side] = group
                        break
                if side not in groups:
                    raise KeyError(
                        f"Could not find tactile depth/marker for '{side}' in "
                        f"{hdf5_path}. Tried groups: "
                        f"{[c.format(side=side) for c in TACTILE_GROUP_CANDIDATES]}"
                    )
        return groups

    # ------------------------------------------------------------------ #
    # TLabel export
    # ------------------------------------------------------------------ #
    def export_to_tlabel(
        self,
        save_root_path,
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        sensor_names: dict[str, str] | None = None,
    ) -> dict:
        """Convert each selected episode to per-sensor TLabel JSON files.

        Args:
            save_root_path: Directory that receives the ``.tlabel.json`` files.
            tactile_cameras: Tactile sides to export (``left``/``right``).
            downsample_factor: Frame stride, matching the base pipeline
                convention (frames ``arange(0, T - 1, factor)`` are exported).
            sensor_names: Optional ``{side: human_readable_sensor_name}``
                override for the TLabel sensor metadata.

        Returns:
            metadata dict, also written to ``<save_root_path>/tlabel_metadata.json``.
        """
        tlabel = _require_tlabel()
        from tlabel.adapters.univtac import UniVTACAdapter

        assert self._data is not None, (
            "Data not loaded. Please call load_data() before export_to_tlabel()."
        )

        self.save_root_path = Path(save_root_path)
        self.save_root_path.mkdir(parents=True, exist_ok=True)

        if not self.tactile_group_map:
            self.tactile_group_map = self._resolve_tactile_groups(
                self.selected_raw_hdf5_paths[0], list(tactile_cameras)
            )

        adapter = self._configure_adapter(UniVTACAdapter(), sensor_names or {})

        log.info(
            "Exporting %d episodes x %d tactile sensor(s) to TLabel JSON (stride=%d)",
            len(self.selected_raw_hdf5_paths), len(tactile_cameras), downsample_factor,
        )

        exported_files: list[dict] = []
        total_frames = 0
        for ep_idx, hdf5_path in enumerate(
            tqdm(self.selected_raw_hdf5_paths, desc="TLabel export", unit="episode")
        ):
            frame_count = self._episode_frame_count(hdf5_path)
            keep_indices = np.arange(0, frame_count - 1, downsample_factor)

            for side in tactile_cameras:
                group = self.tactile_group_map[side]
                # The tlabel adapter reads the full episode; the frame stride
                # is applied afterwards to match the base pipeline convention
                # (export frames [0, stride, 2*stride, ...]).
                tlabel_data = adapter.load(str(hdf5_path), sensor_id=group)
                tlabel_data = self._stride_tlabel_data(tlabel_data, keep_indices)

                marker_count = self._read_marker_count(hdf5_path, group)
                tlabel_data.sensor_info = self._annotate_sensor_info(
                    tlabel_data.sensor_info, group, marker_count
                )
                tlabel_data.episode_info.setdefault("univtac_task", self.task_name)
                tlabel_data.episode_info.setdefault("univtac_config", self.collect_config_name)
                tlabel_data.episode_info["source_episode"] = Path(hdf5_path).stem
                tlabel_data.episode_info["downsample_factor"] = downsample_factor
                # The released tlabel adapter builds sensor_ids with a
                # full-path membership check on the HDF5 root group, which
                # h5py does not resolve; populate it from the groups we
                # actually probed (works for both new *_tactile and old
                # *_gsmini formats).
                with h5py.File(str(hdf5_path), "r") as hf:
                    tlabel_data.episode_info["sensor_ids"] = [
                        g for g in ("left_tactile", "right_tactile",
                                    "left_gsmini", "right_gsmini")
                        if f"tactile/{g}" in hf
                    ]

                out_name = f"episode_{ep_idx:04d}_{side}.tlabel.json"
                out_file = self.save_root_path / out_name
                tlabel_data.export(str(out_file), format="json")

                exported_files.append({
                    "episode": ep_idx,
                    "side": side,
                    "sensor_id": group,
                    "marker_count": marker_count,
                    "file": out_name,
                    "frames": tlabel_data.num_frames,
                    "source": str(hdf5_path),
                })
                total_frames += tlabel_data.num_frames

        metadata = {
            "format": "tlabel",
            "tlabel_schema": "v2",
            "task_name": self.task_name,
            "collect_config_name": self.collect_config_name,
            "num_episodes": len(self.selected_raw_hdf5_paths),
            "tactile_sensors": list(tactile_cameras),
            "tactile_groups": self.tactile_group_map,
            "downsample_factor": downsample_factor,
            "total_tlabel_frames": total_frames,
            "output_dir": str(self.save_root_path.resolve()),
            "files": exported_files,
        }
        with open(self.save_root_path / "tlabel_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        log.info("TLabel export finished: %d files, %d frames -> %s",
                 len(exported_files), total_frames, self.save_root_path)
        return metadata

    # ------------------------------------------------------------------ #
    # Adapter configuration / helpers
    # ------------------------------------------------------------------ #
    def _configure_adapter(self, adapter, sensor_names: dict[str, str]):
        """Register all tactile groups we may encounter with the adapter.

        The released tlabel UniVTACAdapter ships sensor profiles for the old
        ``*_gsmini`` groups.  The current UniVTAC collector writes
        ``*_tactile`` groups, so equivalent profiles are registered here at
        runtime -- without modifying the installed tlabel package.
        Marker counts are never hard-coded: they come from the data.
        """
        from tlabel.adapters.univtac import SENSOR_CONFIG as shipped_config

        for group, profile in _TLABEL_SENSOR_PROFILES.items():
            if group in shipped_config:
                continue
            side = "left" if group.startswith("left") else "right"
            shipped_config[group] = {
                "name": sensor_names.get(side, profile["name"]),
                "type": "vision_based",
                "description": f"{profile['name']} ({group}); "
                               "marker_count inferred dynamically from data",
                # Default resolution for gsmini-class sensors; replaced per
                # episode from the actual depth array by _annotate_sensor_info.
                "resolution": [240, 320],
                "frame_rate": profile["frame_rate"],
                "marker_count": None,
            }
        return adapter

    @staticmethod
    def _read_marker_count(hdf5_path, group: str) -> int:
        """Read marker_size dynamically from ``tactile/{group}/marker``.

        Shape convention: ``(T, 2, marker_size, 2)`` (reference / current
        marker positions, xy coordinates).  Never hard-coded.
        """
        with h5py.File(str(hdf5_path), "r") as f:
            marker_ds = f[f"tactile/{group}/marker"]
            if marker_ds.ndim != 4 or marker_ds.shape[1] != 2 or marker_ds.shape[3] != 2:
                raise ValueError(
                    f"Unexpected marker shape {marker_ds.shape} for "
                    f"tactile/{group}/marker in {hdf5_path}; "
                    "expected (T, 2, marker_size, 2)"
                )
            return int(marker_ds.shape[2])

    @staticmethod
    def _episode_frame_count(hdf5_path) -> int:
        with h5py.File(str(hdf5_path), "r") as f:
            if "step" in f:
                return len(f["step"])
            return len(f["embodiment/joint"])

    @staticmethod
    def _stride_tlabel_data(tlabel_data, keep_indices: np.ndarray):
        """Apply the base-pipeline frame stride to a loaded TLabelData."""
        n = tlabel_data.num_frames
        keep = keep_indices[keep_indices < n]
        tlabel_data.frames = [tlabel_data.frames[i] for i in keep]
        tlabel_data.episode_info["original_frames"] = n
        tlabel_data.episode_info["exported_frames"] = len(keep)
        return tlabel_data

    @staticmethod
    def _annotate_sensor_info(sensor_info: dict, group: str, marker_count: int) -> dict:
        """Fill sensor metadata with dynamically inferred values."""
        sensor_info = dict(sensor_info or {})
        layout = dict(sensor_info.get("layout") or {})
        layout["sensor_id"] = group
        layout["marker_count"] = marker_count
        sensor_info["layout"] = layout
        sensor_info["sensor_id"] = group
        return sensor_info

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_output(save_root_path) -> dict:
        """Re-load every ``.tlabel.json`` under ``save_root_path`` with tlabel.

        Returns a summary dict; raises if any file fails to parse.
        """
        tlabel = _require_tlabel()
        out_root = Path(save_root_path)
        files = sorted(out_root.glob("*.tlabel.json"))
        if not files:
            raise FileNotFoundError(f"No .tlabel.json files found under {out_root}")
        summary = {"files": [], "total_frames": 0}
        for fp in files:
            data = tlabel.load(str(fp))
            summary["files"].append({
                "file": fp.name,
                "frames": data.num_frames,
                "sensor_id": data.sensor_id,
            })
            summary["total_frames"] += data.num_frames
        log.info("Validated %d TLabel files, %d frames total",
                 len(files), summary["total_frames"])
        return summary

    # ------------------------------------------------------------------ #
    # End-to-end entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        save_root_path,
        visual_cameras=("head",),
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        episode_num=50,
        random_select=False,
        sensor_names: dict[str, str] | None = None,
    ) -> dict:
        self.load_data(
            visual_cameras=visual_cameras,
            tactile_cameras=tactile_cameras,
            downsample_factor=downsample_factor,
            episode_num=episode_num,
            random_select=random_select,
        )
        return self.export_to_tlabel(
            save_root_path=save_root_path,
            tactile_cameras=tactile_cameras,
            downsample_factor=downsample_factor,
            sensor_names=sensor_names,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export UniVTAC raw HDF5 episodes to the TLabel tactile annotation format.",
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name, e.g. insert_hole (data lives under data/<task_name>/<task_config>/)",
    )
    parser.add_argument(
        "task_config",
        type=str,
        help="Collect config name, e.g. demo",
    )
    parser.add_argument(
        "episode_num",
        type=int,
        help="Number of episodes to convert",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for .tlabel.json files "
             "(default: ./data/tlabel/<task_name>-<task_config>-<episode_num>)",
    )
    parser.add_argument(
        "--tactile-cameras",
        type=str,
        nargs="+",
        default=["left", "right"],
        choices=["left", "right"],
        help="Tactile sides to export (default: left right)",
    )
    parser.add_argument(
        "--visual-cameras",
        type=str,
        nargs="+",
        default=["head"],
        choices=["head", "wrist"],
        help="Visual cameras (kept for CLI parity with the base pipeline; "
             "TLabel is tactile-only, visual streams are not exported)",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Frame stride (default: 1, every frame)",
    )
    parser.add_argument(
        "--random-select",
        action="store_true",
        help="Randomly select episodes instead of taking the first N",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Re-load every exported file with tlabel.load() to verify it",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path = args.output or (
        REPO_ROOT / "data" / "tlabel"
        / f"{args.task_name}-{args.task_config}-{args.episode_num}"
    )

    processor = TLabelDataPreprocessor(args.task_name, args.task_config)
    metadata = processor.run(
        save_root_path=output_path,
        visual_cameras=tuple(args.visual_cameras),
        tactile_cameras=tuple(args.tactile_cameras),
        downsample_factor=args.downsample_factor,
        episode_num=args.episode_num,
        random_select=args.random_select,
    )

    if args.validate:
        metadata["validation"] = processor.validate_output(output_path)

    log.info("Done. Output directory: %s", output_path)


if __name__ == "__main__":
    main()
