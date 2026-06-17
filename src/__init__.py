"""
点云导航项目 - 3D Gaussian Splatting 场景渲染与 VLN 数据采集

功能模块:
- ply_loader: 3DGS 格式 PLY 文件加载
- trajectory: 轨迹定义与插值
- renderer: 3DGS 渲染器
- video_builder: 视频生成
- vln_collector: VLN 数据采集
- visualize: 3D 可视化工具
"""

__version__ = "0.1.0"
__author__ = "Chandler"

from .ply_loader import load_3dgs_ply, GaussianData
from .trajectory import Trajectory, Waypoint, create_interior_patrol_trajectory
from .renderer import render_view, render_depth
from .video_builder import build_video
from .vln_collector import collect_vln_data
from .visualize import visualize_scene, render_scene_screenshot

__all__ = [
    "load_3dgs_ply",
    "GaussianData",
    "Trajectory",
    "Waypoint",
    "render_view",
    "render_depth",
    "build_video",
    "collect_vln_data",
    "visualize_scene",
    "render_scene_screenshot",
]
