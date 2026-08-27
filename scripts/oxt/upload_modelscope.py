#!/usr/bin/env python3
"""Create a ModelScope dataset if needed and upload a complete folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="创建 ModelScope 数据集并整体上传目录。")
    parser.add_argument("--repo-id", default="byml2024/UniVTAC-OXT")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--token",
        default=(
            os.environ.get("MODELSCOPE_API_TOKEN")
            or os.environ.get("MODELSCOPE_TOKEN")
        ),
    )
    parser.add_argument("--visibility", choices=("public", "private", "internal"), default="public")
    parser.add_argument("--license", default="MIT")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"待上传目录不存在: {folder}")
    files = [path for path in folder.rglob("*") if path.is_file()]
    if not files:
        raise SystemExit(f"待上传目录为空: {folder}")
    print(
        f"repo={args.repo_id} folder={folder} files={len(files)} workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        return 0

    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise SystemExit("缺少 modelscope。请先运行：bash data/download.sh setup") from exc

    api = HubApi()
    repo_url = api.create_repo(
        args.repo_id,
        token=args.token,
        visibility=args.visibility,
        repo_type="dataset",
        chinese_name="UniVTAC OXT 转换数据集",
        license=args.license,
        exist_ok=True,
    )
    print(f"dataset -> {repo_url}", flush=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=folder,
        repo_type="dataset",
        token=args.token,
        max_workers=args.workers,
        commit_message="Publish FTP-1/OXT-compatible UniVTAC dataset",
        commit_description=(
            "Converted and contributed by ScaleLab@SJTU; provenance is recorded "
            "in conversion_manifest.json."
        ),
    )
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
