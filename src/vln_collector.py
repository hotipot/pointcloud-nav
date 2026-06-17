"""
VLN 数据采集模块

批量采集 (image, depth, pose) 数据，输出标准 VLN 数据格式。
用于 Vision-and-Language Navigation 模型研发。
"""

import os
import json
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def collect_vln_data(
    gaussian_data,
    camera_poses,
    output_dir: str = "output/vln_data",
    scene_id: str = "AHOLO",
    width: int = 640,
    height: int = 480,
    fov: Optional[float] = None,
    render_depth: bool = True,
    backend: Optional[str] = None,
    save_depth_png: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    批量采集 VLN 数据

    Args:
        gaussian_data: GaussianData 对象
        camera_poses: CameraPose 列表
        output_dir: 输出目录
        scene_id: 场景 ID
        width: 图像宽度
        height: 图像高度
        fov: 视场角（None 使用各 pose 自带的 fov）
        render_depth: 是否渲染深度图
        backend: 渲染后端
        save_depth_png: 是否将深度图保存为 PNG
        verbose: 是否打印进度

    Returns:
        VLN 数据字典（与保存的 JSON 相同）
    """
    from .renderer import render_view

    output_dir = os.path.expanduser(output_dir)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    total = len(camera_poses)
    trajectory_data = []

    logger.info(f"开始采集 VLN 数据: {total} 帧 -> {output_dir}")

    for i, pose in enumerate(camera_poses):
        actual_fov = fov if fov is not None else pose.fov

        # 渲染
        rgb, depth = render_view(
            gaussian_data, pose, width, height, actual_fov,
            render_depth=render_depth, backend=backend,
        )

        # 保存 RGB 图像
        img_filename = f"{i:06d}.jpg"
        img_path = os.path.join(frames_dir, img_filename)
        Image.fromarray(rgb).save(img_path, quality=95)

        # 保存深度图
        depth_filename = None
        if depth is not None:
            depth_filename = f"{i:06d}_depth.png"
            depth_path = os.path.join(frames_dir, depth_filename)

            if save_depth_png:
                # 归一化深度到 16-bit PNG
                depth_vis = depth.copy()
                valid = depth_vis > 0
                if valid.any():
                    depth_vis[valid] = (depth_vis[valid] - depth_vis[valid].min()) / \
                                       (depth_vis[valid].max() - depth_vis[valid].min() + 1e-8)
                    depth_vis[valid] *= 65535
                depth_vis = depth_vis.astype(np.uint16)
                Image.fromarray(depth_vis).save(depth_path)
            else:
                # 保存为 numpy 数组
                depth_filename = f"{i:06d}_depth.npy"
                np.save(depth_path, depth)

        # 构建位姿数据
        frame_data = {
            "frame_id": i,
            "image_path": f"frames/{img_filename}",
            "pose": {
                "position": pose.position.tolist(),
                "rotation": pose.rotation.tolist(),  # [qw, qx, qy, qz]
            },
            "fov": actual_fov,
            "resolution": [width, height],
        }

        if depth is not None and depth_filename:
            frame_data["depth_path"] = f"frames/{depth_filename}"

        trajectory_data.append(frame_data)

        if verbose and (i + 1) % 50 == 0:
            logger.info(f"采集进度: {i + 1}/{total}")

    # 构建完整 VLN 数据
    vln_data = {
        "scene_id": scene_id,
        "total_frames": total,
        "trajectory": trajectory_data,
    }

    # 保存 JSON
    json_path = os.path.join(output_dir, "vln_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vln_data, f, indent=2, ensure_ascii=False)

    if verbose:
        logger.info(f"VLN 数据已保存: {json_path}")
        logger.info(f"  场景: {scene_id}")
        logger.info(f"  总帧数: {total}")
        logger.info(f"  分辨率: {width}x{height}")
        logger.info(f"  深度图: {'是' if render_depth else '否'}")

    return vln_data


def load_vln_data(json_path: str) -> Dict[str, Any]:
    """
    加载 VLN 数据

    Args:
        json_path: vln_data.json 路径

    Returns:
        VLN 数据字典
    """
    json_path = os.path.expanduser(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def vln_to_r2r_format(vln_data: Dict[str, Any], output_path: str) -> str:
    """
    转换为 R2R 格式（Room-to-Room），兼容常见 VLN 基准

    Args:
        vln_data: VLN 数据字典
        output_path: 输出 JSON 路径

    Returns:
        输出文件路径
    """
    r2r_data = []

    for frame in vln_data["trajectory"]:
        r2r_item = {
            "pathId": vln_data["scene_id"],
            "path": [
                {
                    "x": frame["pose"]["position"][0],
                    "y": frame["pose"]["position"][1],
                    "z": frame["pose"]["position"][2],
                }
            ],
            "heading": frame["pose"]["rotation"],
            "image": frame["image_path"],
        }
        r2r_data.append(r2r_item)

    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(r2r_data, f, indent=2)

    logger.info(f"R2R 格式数据已保存: {output_path}")
    return output_path


def vln_to_habitat_format(vln_data: Dict[str, Any], output_path: str) -> str:
    """
    转换为 Habitat 格式的 episode 数据

    Args:
        vln_data: VLN 数据字典
        output_path: 输出 JSON 路径

    Returns:
        输出文件路径
    """
    episodes = []

    for i, frame in enumerate(vln_data["trajectory"]):
        episode = {
            "episode_id": i,
            "scene_id": vln_data["scene_id"],
            "start_position": frame["pose"]["position"],
            "start_rotation": frame["pose"]["rotation"],
            "info": {
                "fov": frame["fov"],
                "resolution": frame["resolution"],
            },
        }
        if "depth_path" in frame:
            episode["info"]["depth_path"] = frame["depth_path"]
        episodes.append(episode)

    habitat_data = {"episodes": episodes}

    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(habitat_data, f, indent=2)

    logger.info(f"Habitat 格式数据已保存: {output_path}")
    return output_path
