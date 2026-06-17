# 🏗️ 点云导航项目 (PointCloud-Navigation)

基于 3D Gaussian Splatting (3DGS) 的场景渲染与 VLN 数据采集工具。

在变电柜等 3DGS 重建场景中插入机器人轨迹，渲染第一视角行走视频，并批量采集 VLN (Vision-Language Navigation) 训练数据（RGB 图像 + 深度图 + 6DoF 位姿）。

---

## 📋 功能

| 功能 | 说明 |
|------|------|
| **PLY 信息查看** | 解析 3DGS 格式 PLY，显示场景边界、高斯数量、地面平面 |
| **轨迹生成** | 支持矩形巡逻、之字形扫描、自定义 waypoint 三种模式 |
| **3D 可视化** | 交互式查看点云 + 轨迹线 + 相机视锥体 |
| **视频渲染** | 沿轨迹逐帧渲染第一视角 RGB，合成 MP4 视频 |
| **VLN 数据采集** | 批量输出 (image, depth, pose) 数据，支持 R2R / Habitat 格式转换 |

---

## 🛠️ 环境配置

### 基本要求

- Python 3.10+
- pip

### 安装依赖

```bash
cd ~/.openclaw/workspace-coder/pointcloud-nav

# 安装基础依赖
pip install numpy scipy Pillow PyYAML opencv-python

# 渲染引擎（二选一）

# 方案 A：gsplat（推荐，3DGS 原生渲染，画质最好）
# ⚠️ 需要 CUDA + PyTorch
pip install torch gsplat

# 方案 B：Open3D（Fallback，CPU 可用，点云渲染）
pip install open3d
```

> **提示**：如果机器没有 NVIDIA GPU，请使用 Open3D 方案。渲染效果会略差（点云 vs 高斯泼溅），但功能完整。

---

## 🚀 快速开始

### 1. 查看 PLY 文件信息

```bash
python main.py info
```

输出示例：
```
==================================================
  3DGS PLY 文件信息
==================================================
  高斯数量: 306,764
  场景边界:
    X: [-2.35, 5.12] (跨度 7.47m)
    Y: [-1.80, 3.91] (跨度 5.71m)
    Z: [-0.50, 2.80] (跨度 3.30m)
  地面平面:
    法向量: [0.001, 0.002, 1.000]
    距离: 0.50
==================================================
```

### 2. 可视化场景

```bash
# 完整可视化（点云 + 轨迹 + 视锥体）
python main.py visualize

# 仅查看轨迹（不加载点云，速度快）
python main.py visualize --trajectory-only

# 之字形轨迹
python main.py visualize -t zigzag
```

> 可视化窗口支持鼠标旋转/缩放，绿色球体 = 起点，红色球体 = 终点。

### 3. 渲染视频

```bash
# 默认轨迹渲染
python main.py render-video

# 之字形轨迹，自定义输出路径
python main.py render-video -t zigzag -o output/zigzag_walk.mp4

# 自定义分辨率
python main.py render-video --width 1920 --height 1080
```

### 4. 采集 VLN 数据

```bash
# 采集 RGB + 深度图 + 位姿
python main.py collect-vln

# 不采集深度图（更快）
python main.py collect-vln --no-depth

# 自定义输出目录
python main.py collect-vln -o output/my_vln_data
```

---

## 📖 详细使用说明

### 子命令

| 命令 | 说明 |
|------|------|
| `python main.py info` | 查看 PLY 文件信息 |
| `python main.py visualize` | 3D 可视化 |
| `python main.py render-video` | 渲染视频 |
| `python main.py collect-vln` | 采集 VLN 数据 |

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c, --config` | 配置文件路径 | `configs/default.yaml` |

### visualize 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --trajectory-type` | 轨迹类型：`default`(矩形巡逻) / `zigzag`(之字形) | `default` |
| `--trajectory-only` | 仅显示轨迹，不加载点云 | `False` |
| `--point-size` | 点云渲染大小 | `2.0` |
| `--frustum-interval` | 视锥体显示间隔（每隔几帧显示一个） | `30` |

### render-video 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --trajectory-type` | 轨迹类型 | `default` |
| `-o, --output` | 输出视频路径 | `output/video.mp4` |
| `--width` | 图像宽度 | 配置文件中的值 |
| `--height` | 图像高度 | 配置文件中的值 |

### collect-vln 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --trajectory-type` | 轨迹类型 | `default` |
| `-o, --output` | 输出目录 | `output/vln_data` |
| `--no-depth` | 不采集深度图 | `False` |

---

## 🎯 如何自定义轨迹

### 方式 1：修改配置文件

编辑 `configs/default.yaml`，在 `trajectory.waypoints` 中添加坐标：

```yaml
trajectory:
  waypoints:
    - [1.0, 2.0, 0]    # 起点 (x, y, z)
    - [3.0, 2.0, 0]    # 第2个点
    - [3.0, 5.0, 0]    # 第3个点
    - [1.0, 5.0, 0]    # 终点
```

> **注意**：z 坐标会被自动调整为 `地面高度 + robot.height`，所以填 0 即可。

### 方式 2：创建新配置文件

```bash
cp configs/default.yaml configs/my_trajectory.yaml
# 编辑 my_trajectory.yaml
python main.py visualize -c configs/my_trajectory.yaml
```

### 如何确定 waypoint 坐标？

1. 先运行 `python main.py info` 查看场景边界
2. 运行 `python main.py visualize` 交互式查看场景，确定行走路线
3. 根据场景边界和观察结果，在配置文件中填写 waypoint 坐标

### 轨迹参数调整

| 参数 | 位置 | 说明 |
|------|------|------|
| `robot.height` | 配置文件 | 机器人相机高度（默认 1.5m） |
| `robot.fov` | 配置文件 | 相机视场角（默认 90°） |
| `trajectory.speed` | 配置文件 | 行走速度 m/s（默认 1.0） |
| `trajectory.fps` | 配置文件 | 输出帧率（默认 30） |

---

## 📁 输出文件说明

### 视频输出

```
output/
└── video.mp4          # MP4 视频，H.264 编码
```

### VLN 数据输出

```
output/vln_data/
├── vln_data.json      # 元数据（位姿、帧信息）
└── frames/
    ├── 000000.jpg     # RGB 图像
    ├── 000000_depth.png  # 深度图（16-bit PNG）
    ├── 000001.jpg
    ├── 000001_depth.png
    └── ...
```

### vln_data.json 格式

```json
{
  "scene_id": "AHOLO",
  "total_frames": 120,
  "trajectory": [
    {
      "frame_id": 0,
      "image_path": "frames/000000.jpg",
      "depth_path": "frames/000000_depth.png",
      "pose": {
        "position": [1.23, 2.45, 1.50],
        "rotation": [0.998, 0.001, 0.002, 0.060]
      },
      "fov": 90,
      "resolution": [640, 480]
    }
  ]
}
```

> `rotation` 格式为四元数 `[qw, qx, qy, qz]`

---

## 🏗️ 项目结构

```
pointcloud-nav/
├── src/
│   ├── __init__.py        # 模块初始化
│   ├── ply_loader.py      # 3DGS PLY 加载器
│   ├── trajectory.py      # 轨迹定义与插值
│   ├── renderer.py        # 渲染器（gsplat / Open3D）
│   ├── video_builder.py   # 视频合成
│   ├── vln_collector.py   # VLN 数据采集
│   └── visualize.py       # 3D 可视化
├── configs/
│   └── default.yaml       # 默认配置
├── output/                # 输出目录
├── main.py                # CLI 主入口
├── requirements.txt       # Python 依赖
└── README.md              # 本文档
```

---

## ❓ 常见问题

### Q: 运行报错 "No module named 'open3d'" 或 "No module named 'gsplat'"

需要安装至少一个渲染引擎：

```bash
# 方案 A（推荐，需要 GPU）
pip install gsplat

# 方案 B（CPU 可用）
pip install open3d
```

### Q: gsplat 安装失败

gsplat 需要 CUDA 和 PyTorch。如果机器没有 NVIDIA GPU，请使用 Open3D 作为 fallback。

### Q: 渲染效果不理想（点云有空洞）

如果使用 Open3D 渲染，点云在远处会有空洞。解决方案：
1. 使用 gsplat 渲染（3DGS 原生渲染，效果最好）
2. 增加点云密度（在重建时提高采样率）
3. 减小相机 FOV，避免看到太远的区域

### Q: 如何查看场景中的坐标？

运行 `python main.py visualize`，在 Open3D 窗口中可以看到坐标轴。也可以先运行 `python main.py info` 查看场景边界范围。

### Q: 自定义轨迹时 z 坐标填什么？

填 0 即可，程序会自动调整为 `地面高度 + robot.height`。如果你知道确切高度，也可以手动填写。

### Q: PLY 文件格式不对，报错怎么办？

本项目针对 **3D Gaussian Splatting 格式** 的 PLY 文件（包含 scale, rot, f_dc, opacity, f_rest 等属性）。如果是普通点云 PLY（只有 x, y, z, r, g, b），需要修改 `ply_loader.py` 中的属性提取逻辑。

### Q: 内存不够怎么办？

306K 个高斯体大约需要 ~200MB 内存。如果场景更大，可以：
1. 在 `ply_loader.py` 中添加降采样逻辑
2. 减小渲染分辨率（`camera.resolution`）
3. 关闭深度图采集（`--no-depth`）

---

## 📄 许可

MIT License
