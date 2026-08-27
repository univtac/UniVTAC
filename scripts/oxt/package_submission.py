#!/usr/bin/env python3
"""Package converted UniVTAC Zarrs and generate OXT submission metadata."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


OFFICIAL_SJTU_LOGO_URL = "https://vi.sjtu.edu.cn/index.php/downloads/attachments/133"
INSTITUTION = "ScaleLab@SJTU"
INSTITUTION_ZH = "上海交通大学 ScaleLab"
CONTACT = "baijunchen2004@gmail.com"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value


def _zip_zarr(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"归档已存在；确认重建时加 --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.write(path, Path(source.name) / path.relative_to(source))
    os.replace(temporary, destination)


def _install_logo(destination: Path, local_logo: Path | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    if local_logo is not None:
        shutil.copyfile(local_logo, temporary)
    else:
        with urllib.request.urlopen(OFFICIAL_SJTU_LOGO_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
    if temporary.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        temporary.unlink(missing_ok=True)
        raise ValueError("下载/提供的上海交通大学 Logo 不是 PNG")
    os.replace(temporary, destination)


def _dataset_readme(summary: dict[str, Any], dataset_url: str) -> str:
    return f"""---
license: mit
task_categories:
- robotics
---

# UniVTAC-OXT

UniVTAC-OXT is the FTP-1/OXT-compatible release of UniVTAC, a simulated
visuo-tactile manipulation dataset. It contains {summary['trajectory']} trajectories,
{summary['frame']} frames, and {summary['task']} tasks. Every trajectory contains a
fixed main-camera stream, a wrist-camera stream, Franka arm/gripper joint states,
and two GelSight Mini tactile image streams.

The original HDF5 release is available at
https://modelscope.cn/datasets/byml2024/UniVTAC. The converted dataset is hosted at
{dataset_url}.

Conversion provenance and the original per-trajectory result fields are retained
in `conversion_manifest.json`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="打包 UniVTAC-OXT 数据集和 OXT 提交 metadata。")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="byml2024/UniVTAC-OXT")
    parser.add_argument("--local-logo", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    manifest_path = work_dir / "conversion_manifest.json"
    zarr_dir = work_dir / "zarr"
    if not manifest_path.is_file() or not zarr_dir.is_dir():
        raise SystemExit(f"缺少 conversion_manifest.json 或 zarr/: {work_dir}")
    manifest = _read_json(manifest_path)
    summary = manifest["summary"]
    dataset_url = f"https://modelscope.cn/datasets/{args.dataset_id}"

    publish_dir = work_dir / "publish"
    for zarr_path in sorted(zarr_dir.glob("*.zarr")):
        task = zarr_path.name.removesuffix("_head_wrist.zarr")
        archive = publish_dir / task / f"{zarr_path.name}.zip"
        _zip_zarr(zarr_path, archive, args.overwrite)
        print(f"packed {zarr_path} -> {archive}", flush=True)
    publish_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, publish_dir / manifest_path.name)
    (publish_dir / "README.md").write_text(
        _dataset_readme(summary, dataset_url), encoding="utf-8"
    )

    submission_dir = work_dir / "submission" / "OXT_Submission_UniVTAC"
    submission_dir.mkdir(parents=True, exist_ok=True)
    logo_name = "institution_ScaleLab@SJTU.png"
    _install_logo(submission_dir / logo_name, args.local_logo)
    metadata = {
        "name": "UniVTAC",
        "url": [dataset_url],
        "version": "1.0.0",
        "description": (
            "Open-source simulated visuo-tactile manipulation data from UniVTAC, "
            "converted to the FTP-1/OXT format."
        ),
        "type": "gripper",
        "sensor_image": ["GelSightMini"],
        "sensor_array": [],
        "sensor_state": [],
        "label": [],
        "task": summary["task"],
        "trajectory": summary["trajectory"],
        "frame": summary["frame"],
        "contact": [CONTACT],
        "reference": {
            "paper": (
                "UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation "
                "Data Generation, Learning, and Benchmarking"
            ),
            "website": "https://univtac.github.io/",
            "authors": (
                "Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, "
                "Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al."
            ),
        },
        "code_trigger": "FTP_1_Source",
    }
    with (submission_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    profile = {
        "institution": INSTITUTION,
        "institution_zh": INSTITUTION_ZH,
        "institution_logo": logo_name,
        "institution_logo_source": OFFICIAL_SJTU_LOGO_URL,
        "contact": CONTACT,
        "dataset_id": args.dataset_id,
    }
    with (submission_dir / "submission_profile.json").open("w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"publish folder -> {publish_dir}", flush=True)
    print(f"OXT submission metadata -> {submission_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
