# UniVTAC → OXT 数据迁移

入口统一为 `data/download.sh`。默认目录都位于被 Git 忽略的 `data/` 下，可用环境变量覆盖：

```bash
bash data/download.sh setup
bash data/download.sh download
bash data/download.sh convert
bash data/download.sh upload
bash data/download.sh submit --dry-run
bash data/download.sh submit
```

其中 `upload` 是本地准备步骤；如果希望使用更准确的命令名，也可以执行与它等价的
`bash data/download.sh package`。两者不要连续执行，否则第二次打包会因归档已存在而停止；
需要主动重建时可添加 `--overwrite`。

建议在独立 Python/Conda 环境中执行 `setup`，避免数据工具依赖影响 Isaac Sim 环境。

也可以依次完成下载、转换和打包：

```bash
bash data/download.sh all
```

常用环境变量：

- `UNIVTAC_RAW_DIR`：ModelScope 原始 HDF5 目录。
- `OXT_WORK_DIR`：转换工作目录。
- `SOURCE_DATASET_ID`、`SOURCE_REVISION`：原数据集及版本。默认固定到已复核的
  ModelScope revision `3d4646a7`，不要无审查地改回 `master`。
- `OXT_DATASET_ID`：新建的转换后数据集，默认 `byml2024/UniVTAC-OXT`。
- `DOWNLOAD_WORKERS`、`CONVERT_WORKERS`、`UPLOAD_WORKERS`：三阶段并行度。
- `MODELSCOPE_API_TOKEN`：提交 token；也可先执行 `modelscope login`。
- `MODELSCOPE_LICENSE`：新数据集许可证，默认 `MIT`，正式发布前应与原数据授权再次核对。

`download` 使用 ModelScope 的文件级 worker，只下载每个 task 的 HDF5 和 metadata；
`convert` 以单个 HDF5 为 worker 粒度。worker 先写独立 staging Zarr，主进程再按 task 和
episode id 稳定合并，避免并发追加同一个 Zarr。转换完成后生成
`conversion_manifest.json`。每条轨迹只保留 source 定位信息、seed、原始帧数和转换后帧数。

`package` 将每个 task Zarr 打包到 `publish/<task>/`，生成数据集 README、OXT metadata
和 `submission/OXT_Submission_UniVTAC/institution_ScaleLab@SJTU.png`。Logo 从上海交通大学
官方视觉形象识别系统下载。完整数据生成后，还需要使用 OXT-QC 当前版本执行 metadata
precheck 和 VLM/QC，并将其输出加入 submission 目录，再向 OXT-QC 提交 PR。

`upload` 只在当前设备上执行 `package`，并把待上传内容保存在
`data/oxt/UniVTAC-OXT/publish/`，不会创建远程数据集或提交文件。检查本地结果后，只有显式
执行 `submit` 才会连接 ModelScope；可先运行 `submit --dry-run` 核对目录和文件数量。

## 已确定的转换口径

- 当前发布版预期为 8 个 task、800 个 HDF5。默认严格检查总数，防止上游数据变化后仍沿用
  本次对 800 条数据作出的结论；同时检查原 metadata 分布必须为 763 success、19 fail、
  18 missing。非 `3d4646a7` revision 默认拒绝全量转换。
- 800 条全部纳入贡献集。旧 metadata 的 result 字段只用于转换前版本校验，不复制到 Zarr
  或 manifest，也不额外写入恒为 success 的字段。该决策只适用于已经回放/终态核验过的
  当前发布版。
- Zarr `meta/` 只保存 `episode_ends`、`episode_lengths` 和 `episode_seeds`；其中
  `episode_ends` 是 FTP-1 episode 切分所必需，后两项分别是每条轨迹的转换后帧数和 seed。
- `head` 相机固定在 world/environment 下，因此映射为 `camera_main_rgb`，不标为 ego；
  `wrist` 对所有 task 都保留为 `right_wrist_camera_rgb`。
- 左右 GelSight Mini 的 `rgb_marker` 按 `[left, right]` 合并为
  `right_tactile_data_gripper`，功能区索引为 `[0, 1]`。
- 关节导出 7 维右臂和第 8 维夹爪，夹爪 UAS 索引为 28。最后一帧只作为未来状态，不作为
  当前 observation，与 FTP-1 的 UniVTAC 参考解析器一致。
- `embodiment/ee` 暂不导出：UniVTAC 的四元数/末端坐标系尚未验证到 FTP-1 的 gripper pose
  约定。保留关节状态可避免写入看似完整但坐标语义错误的 pose。
- JPEG 解码保持 UniVTAC 原始数值通道。采集端对 Isaac RGB 数组直接调用 OpenCV 编码；本
  转换的最终结果与 FTP-1 参考解析器的两次通道翻转等价。

## 全量服务器流程

```bash
DOWNLOAD_WORKERS=16 CONVERT_WORKERS=8 bash data/download.sh all
bash data/download.sh submit --dry-run
bash data/download.sh submit
```

`all` 和 `upload` 都不会进行远程提交，避免转换完成后未经人工检查就产生外部发布。正式提交前至少核对
`conversion_manifest.json` 中的 task/trajectory/frame 总数和许可证。
