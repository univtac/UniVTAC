# XenseWS 标定与 UniVTAC 接入教程

本文说明如何为 XenseWS 触觉传感器生成 Taxim 光学标定文件、将标定结果接入 UniVTAC，并校准仿真中的 marker 布局。开始前请先按 [安装文档](docs/Installation.md) 完成 UniVTAC 环境配置。

参考资料：

- [Taxim 仓库](https://github.com/Robo-Touch/Taxim)
- [Taxim 论文](https://arxiv.org/abs/2109.04027)
- [UniVTAC 数据采集说明](docs/Collection.md)

## 1. 先区分两类标定

XenseWS 的仿真包含两条相互独立的链路：

1. **Taxim 光学标定**：根据真实传感器的背景图和球压数据，生成接触区域的颜色与光照响应。结果主要由 `dataPack.npz`、`polycalib.npz` 和 `params.json` 决定。
2. **Marker 布局与运动标定**：在 FEM 胶体表面布置 marker，并把三维运动投影到触觉图像。参数位于 [`create_xensews_cfg`](envs/sensors/tactile.py)；它不读取 Taxim 的标定结果。

因此，光学图像正常并不代表 marker 一定正常，反之亦然。两部分需要分别验证。

## 2. 仓库当前状态

XenseWS 的标定目录已经存在：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Sensors/XenseWS/
└── calibs/640x480/
    ├── 0.png
    ├── dataPack.npz
    ├── gelmap.npy
    ├── params.json
    ├── polycalib.npz
    └── shadowTable.npz
```

当前目录只能作为接入占位基线：`polycalib.npz`、`gelmap.npy`、`params.json` 和 `shadowTable.npz` 与 GelSight Mini 版本完全相同，尚不是完整的 XenseWS 标定结果。现有 XenseWS `dataPack.npz` 包含真实背景和 24 张球压图，但接触中心集中在图像中央，不能充分拟合 Taxim 使用的空间多项式响应。正式标定时应让接触点覆盖整个有效成像区域。

XenseWS 当前只接入了 Taxim 光学后端。任务配置应使用：

```yaml
sensor_type: xensews
optical_backend: taxim
```

## 3. 准备真实标定数据

TacEx 内置的是 Taxim 的 GPU 渲染实现，不包含上游的数据采集和人工标注工具，因此需要另外取得 Taxim 仓库：

```bash
git clone https://github.com/Robo-Touch/Taxim.git
cd Taxim
pip install -r requirements.txt
```

人工标注程序还依赖 NanoGUI 的 Python 绑定，但该依赖没有包含在 `requirements.txt` 中，需要按 [Taxim README](https://github.com/Robo-Touch/Taxim#installation-and-prerequisites) 单独安装。

> Taxim 上游的数据采集代码较旧：`DataCollection/record_Gel.py` 使用 ROS、`raw_input` 和 Python 2 语法，不能直接在 UniVTAC 的 Python 3.11 环境中运行。可以在兼容的旧环境中使用它，也可以使用 XenseWS SDK 或 OpenCV 自行采图。后续标定只要求图像目录满足下面的格式。建议为上游工具使用独立环境，不要为兼容旧依赖而降级 UniVTAC 环境中的 NumPy、OpenCV 或 SciPy。

### 3.1 球压数据要求

使用已知直径的刚性球头采集图像。目录中的文件应按数字编号：

```text
calib_ball/
├── frame_0.png       # 无接触背景图
├── frame_1.png       # 球压图
├── frame_2.png
└── ...
```

采集时满足以下条件：

- 所有图像使用相同的曝光、白平衡、增益、裁剪、缩放和颜色通道顺序。
- `frame_0` 必须是胶体完全无接触时的背景图。
- 当前标定目录以原生 `640 × 480` 为基准，因此数组形状应为 `(480, 640, 3)`。
- 建议采集至少 30～50 个有效接触，并覆盖中心、四角和边缘；不要只在中心反复按压。
- 使用不同按压深度，但避免球头压穿、严重饱和或接触圆超出图像边界。
- 记录球头的真实半径。Taxim 的 `ball_radius` 单位是毫米，填写的是**半径而不是直径**。

Taxim 标定脚本通过 OpenCV 读取 BGR 图像，TacEx 加载时会完成 BGR 到 RGB 的转换。不要在中间步骤额外交换颜色通道。

### 3.2 设置传感器参数

编辑 Taxim 仓库中的 `Basics/sensorParams.py`：

```python
ball_radius = 2.0  # 示例：直径 4 mm 的球头
pixmm = 0.0295     # 每像素对应的毫米数，需替换为实测值
numBins = 125
h = 480
w = 640
```

`pixmm` 是毫米/像素。应根据经过裁剪和畸变校正后的有效成像区域实测，而不是直接沿用 GelSight Mini 的值。Taxim 只支持一个标量 `pixmm`；如果横纵方向的毫米/像素明显不同，应先对真实图像做固定的裁剪或几何校正，使像素在两个方向上具有一致的物理尺度，再采集整套标定数据。

一旦确定图像预处理方式，背景图、球压图和以后用于对比的真实触觉图像必须使用完全相同的处理流程。

## 4. 生成 Taxim 标定文件

以下命令应从 Taxim 仓库的 `Calibration/` 目录执行，因为上游脚本通过相对路径导入 `Basics`：

```bash
cd Taxim/Calibration
python generateDataPack.py -data_path /absolute/path/to/calib_ball
```

在标注窗口中：

- 使用方向键移动圆心；
- 使用 `m` / `p` 缩小或放大圆；
- 使用 `f` / `c` 减小或增大移动步长；
- 让圆与真实球头接触边界重合，然后点击 **Calibrate**；
- 无法可靠标注的帧点击 **Skip**；
- 全部处理完后点击 **Save Params**，生成 `dataPack.npz`。直接关闭窗口不会自动保存。

随后生成光学响应查找表：

```bash
python polyTableCalib.py -data_path /absolute/path/to/calib_ball
```

输出文件为：

```text
calib_ball/dataPack.npz
calib_ball/polycalib.npz
```

### 4.1 阴影标定

当前 XenseWS 配置设置了 `with_shadow=False`，实际渲染不会应用阴影，但 TacEx 初始化时仍会读取 `shadowTable.npz`。如果暂时不标定阴影，应保留仓库中现有且格式有效的占位文件，不能删除。

如需启用阴影，请按 Taxim 上游说明单独采集阴影标定图并生成对应的 `dataPack.npz`，然后执行：

```bash
python generateShadowMasks.py -data_path /absolute/path/to/calib_shadow/
```

路径末尾保留 `/`；上游脚本直接拼接输出文件名，否则可能把文件写到错误的位置。生成后还需要在 [`XenseWSCfg`](third_party/TacEx/source/tacex_assets/tacex_assets/sensors/xensews/xensews_cfg.py) 中将 `with_shadow` 改为 `True`，再重新验证渲染效果。

### 4.2 `gelmap.npy` 与 `params.json`

Taxim 上游脚本不会自动生成 TacEx 所需的这两个文件：

- `gelmap.npy`：形状必须为 `(h, w)`，表示无接触胶体表面的高度图；数值按像素高度存储，TacEx 会再乘以 `pixmm`。对于近似平面的胶体，可以先保留常量平面作为基线；若胶体有明显曲率，应使用实测或重建的表面高度图。
- `params.json`：包含 Taxim 渲染参数和传感器参数。可以复制当前文件作为起点，但至少要把 `sensor.pixmm`、`sensor.w`、`sensor.h` 改成与本次标定一致的值。

建议首次接入时保持 `params.json` 的 `simulator` 段不变，先确认背景、颜色方向和接触位置正确，再调节软胶扩散和阴影参数。不要同时改动光学标定、胶体形变和 marker 参数，否则出现偏差时很难定位原因。

## 5. 部署到 UniVTAC

将生成或确认过的文件放入：

```text
third_party/TacEx/source/tacex_assets/tacex_assets/data/Sensors/XenseWS/calibs/640x480/
```

Taxim 运行时实际需要以下五个文件：

| 文件 | 作用 | 必须为 XenseWS 专属 |
|---|---|---|
| `dataPack.npz` | 提供无接触背景 `f0` | 是 |
| `polycalib.npz` | 表面梯度到 RGB 响应的空间多项式 | 是 |
| `params.json` | 图像尺寸、`pixmm` 和渲染参数 | 是 |
| `gelmap.npy` | 无接触胶体表面高度 | 建议 |
| `shadowTable.npz` | 阴影响应；关闭阴影时仍须存在 | 启用阴影时是 |

`0.png` 不会被 XenseWS 的 Taxim 后端读取；真正的背景来自 `dataPack.npz` 中的 `f0`。可以保留 `0.png` 作为人工预览，但不要用它代替 `dataPack.npz`。

替换文件前建议把旧目录复制到仓库外备份。不要覆盖 GelSight Mini 的标定目录。

### 5.1 静态检查

在 UniVTAC 根目录运行：

```bash
python - <<'PY'
import json
from pathlib import Path

import numpy as np

root = Path(
    "third_party/TacEx/source/tacex_assets/tacex_assets/"
    "data/Sensors/XenseWS/calibs/640x480"
)
required = {
    "dataPack.npz", "polycalib.npz", "params.json",
    "gelmap.npy", "shadowTable.npz",
}
missing = sorted(name for name in required if not (root / name).is_file())
assert not missing, f"missing files: {missing}"

params = json.loads((root / "params.json").read_text())
h = params["sensor"]["h"]
w = params["sensor"]["w"]
bins = params["sensor"]["num_bins"]

data = np.load(root / "dataPack.npz", allow_pickle=True)
poly = np.load(root / "polycalib.npz")
shadow = np.load(root / "shadowTable.npz", allow_pickle=True)
gelmap = np.load(root / "gelmap.npy")

assert "f0" in data.files and data["f0"].shape == (h, w, 3)
assert {"bins", "grad_r", "grad_g", "grad_b"} <= set(poly.files)
assert int(poly["bins"]) == bins
for key in ("grad_r", "grad_g", "grad_b"):
    assert poly[key].shape == (bins, bins, 6), (key, poly[key].shape)
assert gelmap.shape == (h, w), gelmap.shape
assert {"shadowDirections", "shadowTable"} <= set(shadow.files)

print(f"XenseWS calibration OK: {w}x{h}, pixmm={params['sensor']['pixmm']}")
PY
```

这个检查只能发现缺文件、键名和尺寸错误，不能判断标定质量。

## 6. 校准 marker 布局

XenseWS marker 参数位于 [`envs/sensors/tactile.py`](envs/sensors/tactile.py) 的 `create_xensews_cfg`：

```python
resolution = (320, 240)
marker_shape = (20, 11)
marker_interval = (1.15, 1.2)
sub_marker_num = 0
marker_radius = 2
camera_to_surface = 0.0245
real_size = (0.0293, 0.0175)
num_markers = 220
```

各参数含义如下：

| 参数 | 含义 |
|---|---|
| `resolution` | 最终触觉图像的 `(宽, 高)`，当前为 `320 × 240` |
| `marker_shape` | 横向、纵向 marker 数量；当前共 `20 × 11 = 220` 个 |
| `marker_interval` | 横向、纵向 marker 中心间距，单位毫米 |
| `num_markers` | 输出 marker 数量，必须与 `marker_shape` 的乘积一致 |
| `marker_radius` | 当前渲染中 marker 圆点的视觉半径，主要按像素调节 |
| `camera_to_surface` | 相机到胶体表面的距离，单位米 |
| `real_size` | 图像覆盖的真实区域 `(宽, 高)`，单位米 |

这部分参数是已经调整过的，可以先固定，待到触觉背景确定后，再视情况进行微调。

## 7. 采集 1～2 条接触数据进行验收

可以修改通用的 `task_config/contact.yml`，设置：

```yaml
episode_num: 2
sensor_type: xensews
optical_backend: taxim

observations:
  camera: ['rgb']
  tactile: ['rgb', 'rgb_marker', 'marker', 'depth', 'press_depth', 'pose']
  embodiment: ['joint', 'ee']
  actor: true
```

在 UniVTAC 根目录启动单环境采集（环境变量可传入形状，形状可见 `assets/objects/Bar_<形状名>`）：

```bash
PRISM_NAME=Hemisphere python scripts/collect_contact.py \
    collect contact_xensews --headless --device cuda:0
```

结果默认保存在：

```text
data/contact_xensews/Hemisphere/
├── hdf5/
├── video/
├── metadata.json
└── suc_map.txt
```

视频右侧上下两个触觉面板分别是左右传感器的 `rgb_marker`。需要分别检查纯光学结果时，可从 HDF5 中读取 `tactile/left_tactile/rgb` 和 `tactile/right_tactile/rgb`；marker 叠加结果位于对应的 `rgb_marker`。

## 8. 验收标准

标定完成后应同时满足：

- 程序启动时没有缺文件、NPZ 键名、图像尺寸或 CUDA 设备错误。
- 无接触时，仿真背景的整体颜色、亮度分布和真实 XenseWS 接近，没有虚假接触区域。
- 随按压深度增加，接触区域连续扩大，颜色变化方向正确且不过早饱和。
- 左右传感器的背景、接触方向和 marker 朝向一致。
- 无接触 marker 数量正确、分布居中、没有越界；运动过程没有整体漂移、瞬移或每帧重新居中。
- `marker_shape` 的乘积与 `num_markers` 一致，HDF5 中 marker 数组形状为 `(2, 220, 2)`。
- 至少检查 1～2 条完整采集视频，并抽查 HDF5 中的 `rgb`、`rgb_marker`、`marker` 和 `press_depth`。

## 9. 常见问题

| 现象 | 优先检查 |
|---|---|
| 启动时报 `FileNotFoundError` | 五个运行时标定文件是否都位于 `calibs/640x480/`；即使关闭阴影也要保留 `shadowTable.npz` |
| 背景正确但接触颜色像 GelSight Mini | 是否只替换了 `dataPack.npz`，却仍在使用 GS Mini 的 `polycalib.npz` |
| 图像旋转、镜像或宽高颠倒 | 采集端裁剪/旋转、`params.json` 的 `w/h`、`dataPack.f0` 形状是否一致 |
| 红蓝通道互换 | 是否在 OpenCV BGR 流程之外又手工交换了一次通道 |
| 接触范围或强度明显不对 | `ball_radius`、`pixmm`、`gelmap.npy` 和真实球头尺寸是否一致 |
| marker 数量不对 | `marker_shape` 的乘积是否等于 `marker_params.num_markers` |
| 调整 `real_size` 后 marker 不变 | `mani_skill_sim.py` 是否已把 `real_size` 和 `camera_to_surface` 传到底层 |
| marker 整体偏移或越界 | 先检查相机投影和有效成像区域，再检查间距；不要先用平移随机化掩盖系统误差 |

建议按“静态文件检查 → 无接触背景 → 单点轻压 → 多位置多深度 → marker 运动 → 完整采集”的顺序逐级验证。