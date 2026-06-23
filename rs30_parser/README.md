# RS30 数据解析器

解析 RS30 移动扫描系统的专有数据格式，提取图像和相机参数，转换为 COLMAP 标准格式。

## 支持的格式

| 文件 | 模块 | 说明 |
|------|------|------|
| `.cam` | `extract_images.py` | 相机图像流（JPEG 帧序列） |
| `.CP` | `parse_camera_params.py` | 相机内参（Protobuf 编码） |
| `gnss_valid_trajectory.txt` | `trajectory_to_colmap.py` | GNSS/INS 轨迹 |
| `utm_origin_pose.txt` | `trajectory_to_colmap.py` | UTM 原点位姿 |
| `time.diff` | `trajectory_to_colmap.py` | 设备-GPS 时间偏移 |

## 快速开始

### 安装

```bash
pip install opencv-python-headless numpy pillow
```

### 一键转换（推荐）

将 RS30 站点数据转换为 COLMAP 格式，可直接用于 SuGaR / 3DGS 训练：

```bash
python -m rs30_parser to_colmap /path/to/site_data -o /path/to/output
```

**关键特性**：
- ✅ 自动提取 `.cam` 中的 JPEG 图像
- ✅ 自动解析 `.CP` 中的相机内参
- ✅ **自动去畸变**（RS30 镜头畸变显著 k1≈0.39，SuGaR 只支持 PINHOLE 模型）
- ✅ 输出 PINHOLE 模型的 COLMAP 项目（SuGaR 兼容）
- ✅ 生成 flat `images/` 目录（3DGS 标准输入）

### 命令行参数

```bash
python -m rs30_parser to_colmap SITE_DIR [OPTIONS]

位置参数:
  site_dir              RS30 站点数据目录

可选参数:
  -o, --output          输出 COLMAP 项目目录（默认: site_dir_colmap）
  --cameras             指定相机（如 Camera1 Camera2 Camera3）
  --max-frames N        每相机最大帧数（0=全部）
  --with-poses          写入 SLAM 位姿到 images.bin
  --no-undistort        不去畸变（保留 OPENCV 模型，SuGaR 不支持）
  --force-pinhole       忽略畸变，强制 PINHOLE（不推荐，精度有损）
```

### 其他子命令

```bash
# 仅提取图像
python -m rs30_parser extract /path/to/site_data -o ./images

# 查看相机参数
python -m rs30_parser info /path/to/site_data --colmap
```

## 输出结构

```
output_dir/
├── images/                  # 去畸变后的图像（flat，SuGaR/3DGS 直接用）
│   ├── Camera1_xxxxxx.jpg
│   ├── Camera2_xxxxxx.jpg
│   └── Camera3_xxxxxx.jpg
├── Camera1/
│   ├── images/              # 去畸变后的图像（按相机分目录）
│   └── raw/                 # 原始图像备份
├── Camera2/
│   ├── images/
│   └── raw/
├── Camera3/
│   ├── images/
│   └── raw/
├── sparse/0/
│   ├── cameras.bin          # PINHOLE 模型（去畸变后）
│   ├── images.bin           # 空（让 COLMAP SfM 计算）
│   └── points3D.bin         # 空
├── cameras.json             # NeRF 格式辅助文件
└── conversion_meta.json     # 转换元数据
```

## SuGaR 训练流程

转换完成后，在 Ubuntu 上运行：

```bash
# 1. COLMAP SfM
colmap feature_extractor --database_path db.db --image_path images/ \
    --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path db.db
colmap mapper --database_path db.db --image_path images/ --output_path sparse/

# 2. SuGaR 全流程
python train_full_pipeline.py -s . -r dn_consistency --high_poly True
```

详见 [SuGaR Ubuntu 部署指南](../docs/sugar-ubuntu-deploy.md)。

## 已知限制

- **Camera4 (Insta360)**：`.insv` 格式需要官方 Stitcher 软件，暂不支持
- **`.KXY` 文件**：关键点数据格式尚未完全解析
- **`.TP` 文件**：时间参数文件（404 bytes），格式待分析
