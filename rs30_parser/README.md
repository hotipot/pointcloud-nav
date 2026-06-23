# RS30 Data Parser

逆向解析 RS30 移动扫描系统的私有数据格式，提取标准格式数据用于 3DGS/NeRF 重建。

## 支持的格式

| 文件 | 格式 | 解析状态 |
|------|------|----------|
| `.cam` | 相机图像帧（内嵌 JPEG） | ✅ 已解析 |
| `.CP` | 相机参数（Protobuf） | ✅ 已解析 |
| `.TP` | 时间参数 | ✅ 已解析 |
| `.KXY` | 关键点数据 | ⏳ 待解析 |
| `.pcap` | 禾赛激光雷达数据 | 📦 需禾赛 SDK |
| `.tra` | SLAM 轨迹 | ✅ 已解析 |
| `gnss_valid_trajectory.txt` | GNSS 轨迹 | ✅ 已解析 |
| `utm_origin_pose.txt` | UTM 原点位姿 | ✅ 已解析 |
| `.gga` | NMEA GGA | ✅ 标准格式 |

## 快速使用

```bash
# 提取某站点的全部图像帧
python -m rs30_parser.extract_images /path/to/2026-06-13-BD-GLKGQ

# 解析相机内参
python -m rs30_parser.parse_camera_params /path/to/2026-06-13-BD-GLKGQ

# 转换 SLAM 轨迹为 COLMAP 格式
python -m rs30_parser.trajectory_to_colmap /path/to/2026-06-13-BD-GLKGQ

# 一键转换为 COLMAP 项目
python -m rs30_parser.to_colmap /path/to/2026-06-13-BD-GLKGQ -o /path/to/colmap_output
```
