"""
RS30 站点数据一键转换 → COLMAP 项目

将 RS30 移动扫描数据转换为 COLMAP SfM 格式，可直接用于：
  - COLMAP 特征提取 + 匹配 + SfM
  - 3DGS / SuGaR 训练
  - NeRF (nerfacto) 训练

输出结构：
  colmap_output/
  ├── images/          # 提取的 JPG 图像帧
  │   ├── Camera1/
  │   ├── Camera2/
  │   └── Camera3/
  ├── sparse/
  │   ├── 0/
  │   │   ├── cameras.bin   # 相机内参
  │   │   ├── images.bin    # 图像外参（初始值，可选）
  │   │   └── points3D.bin  # 空点云（让 COLMAP 自己算）
  ├── cameras.json     # 相机参数（NeRF 格式）
  └── transforms.json  # 位姿数据（NeRF 格式，可选）
"""

import os
import struct
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

from .extract_images import extract_site_images, parse_cam_file, extract_jpeg
from .parse_camera_params import (
    parse_all_cameras, intrinsics_to_colmap_dict, CameraIntrinsics,
)
from .trajectory_to_colmap import (
    parse_gnss_trajectory, parse_utm_origin, align_poses_to_local,
    parse_time_diff, build_colmap_images_dict,
)

logger = logging.getLogger(__name__)


# COLMAP 二进制格式常量
COLMAP_MAGIC_CAMERAS = 634527751
COLMAP_MAGIC_IMAGES = 634527752
COLMAP_MAGIC_POINTS3D = 634527753


def write_colmap_cameras_bin(cameras: dict, output_path: str):
    """写入 COLMAP cameras.bin
    
    Args:
        cameras: {camera_id: {model, width, height, params}}
        output_path: 输出文件路径
    """
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", COLMAP_MAGIC_CAMERAS))
        f.write(struct.pack("<Q", len(cameras)))
        
        for cam_id, cam in cameras.items():
            model_name = cam["model"]
            width = cam["width"]
            height = cam["height"]
            params = cam["params"]
            
            # 模型 ID 映射
            model_id = {
                "PINHOLE": 0,
                "OPENCV": 2,
                "SIMPLE_PINHOLE": 1,
            }.get(model_name, 0)
            
            f.write(struct.pack("<IQQ", cam_id, model_id, width))
            f.write(struct.pack("<Q", height))
            f.write(struct.pack("<Q", len(params)))
            for p in params:
                f.write(struct.pack("<d", p))


def write_colmap_images_bin(images: dict, output_path: str):
    """写入 COLMAP images.bin
    
    Args:
        images: {image_id: {name, qvec, tvec, xys, point3D_ids}}
        output_path: 输出文件路径
    """
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", COLMAP_MAGIC_IMAGES))
        f.write(struct.pack("<Q", len(images)))
        
        for img_id, img in images.items():
            # image_id
            f.write(struct.pack("<I", img_id))
            
            # qvec (qw, qx, qy, qz)
            qvec = img.get("qvec", np.array([1, 0, 0, 0]))
            for q in qvec:
                f.write(struct.pack("<d", q))
            
            # tvec (tx, ty, tz)
            tvec = img.get("tvec", np.array([0, 0, 0]))
            for t in tvec:
                f.write(struct.pack("<d", t))
            
            # camera_id
            f.write(struct.pack("<I", img.get("camera_id", 1)))
            
            # name (null-terminated string)
            name = img.get("name", f"{img_id:06d}.jpg")
            f.write(name.encode("utf-8") + b"\x00")
            
            # num_points2D (0 for initial)
            num_points = len(img.get("xys", []))
            f.write(struct.pack("<Q", num_points))
            
            # xys and point3D_ids (empty for initial)
            for xy, p3d in zip(img.get("xys", []), img.get("point3D_ids", [])):
                f.write(struct.pack("<dd", xy[0], xy[1]))
                f.write(struct.pack("<Q", p3d))


def write_colmap_points3d_bin(output_path: str):
    """写入空的 COLMAP points3D.bin
    
    让 COLMAP SfM 自己从图像中提取 3D 点
    """
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", COLMAP_MAGIC_POINTS3D))
        f.write(struct.pack("<Q", 0))  # 0 个点


def write_cameras_json(cameras: dict, output_path: str):
    """写入 NeRF 格式的 cameras.json"""
    out = []
    for cam_id, cam in cameras.items():
        out.append({
            "id": cam_id,
            "model": cam["model"],
            "width": cam["width"],
            "height": cam["height"],
            "params": cam["params"],
        })
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


def convert_site_to_colmap(
    site_dir: str,
    output_dir: str,
    cameras: Optional[List[str]] = None,
    max_frames_per_camera: int = 0,
    use_slam_poses: bool = True,
    write_initial_poses: bool = False,
) -> str:
    """一键转换站点数据为 COLMAP 项目
    
    Args:
        site_dir: RS30 站点数据目录
        output_dir: 输出 COLMAP 项目目录
        cameras: 指定相机列表（None = 全部 .cam 相机）
        max_frames_per_camera: 每相机最大帧数（0 = 全部）
        use_slam_poses: 是否使用 SLAM 位姿（True 用 SLAM，False 让 COLMAP 自己算）
        write_initial_poses: 是否写入初始位姿到 images.bin
        
    Returns:
        输出目录路径
    """
    site_path = Path(site_dir)
    out_path = Path(output_dir)
    
    logger.info(f"转换站点: {site_path.name}")
    logger.info(f"输出目录: {out_path}")
    
    # Step 1: 提取图像
    logger.info("=== Step 1: 提取图像 ===")
    image_results = extract_site_images(
        str(site_path),
        str(out_path),
        cameras=cameras,
        max_frames=max_frames_per_camera,
    )
    
    # Step 2: 解析相机内参
    logger.info("=== Step 2: 解析相机内参 ===")
    intrinsics = parse_all_cameras(str(site_path))
    
    # Step 3: 构建 COLMAP cameras.bin
    logger.info("=== Step 3: 构建 COLMAP cameras ===")
    colmap_cameras = {}
    cam_id = 1
    for cam_name, intr in intrinsics.items():
        colmap_cam = intrinsics_to_colmap_dict(intr)
        colmap_cameras[cam_id] = colmap_cam
        logger.info(f"  {cam_name} → camera_id={cam_id}, model={colmap_cam['model']}")
        cam_id += 1
    
    # 如果没有在 intrinsics 中发现相机，添加默认相机
    for cam_name in image_results:
        if cam_name not in intrinsics:
            # 使用默认参数（基于已知的 1944x2592 分辨率）
            default_intr = CameraIntrinsics(
                fx=1018.09, fy=1018.09,
                cx=972.0, cy=1296.0,  # 图像中心
                image_width=1944, image_height=2592,
            )
            colmap_cameras[cam_id] = intrinsics_to_colmap_dict(default_intr)
            logger.info(f"  {cam_name} → camera_id={cam_id} (默认内参)")
            cam_id += 1
    
    # Step 4: 写入 COLMAP 二进制文件
    logger.info("=== Step 4: 写入 COLMAP 文件 ===")
    sparse_dir = out_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    write_colmap_cameras_bin(colmap_cameras, str(sparse_dir / "cameras.bin"))
    write_colmap_points3d_bin(str(sparse_dir / "points3D.bin"))
    
    if write_initial_poses and use_slam_poses:
        # Step 5: 使用 SLAM 位姿
        logger.info("=== Step 5: 构建 SLAM 位姿 ===")
        gnss_poses = parse_gnss_trajectory(
            str(site_path / "SLAM_DATA" / "gnss_valid_trajectory.txt")
        )
        origin_pos, origin_q = parse_utm_origin(
            str(site_path / "SLAM_DATA" / "utm_origin_pose.txt")
        )
        aligned_poses = align_poses_to_local(gnss_poses, origin_pos)
        time_offset = parse_time_diff(
            str(site_path / "GPS" / "Time" / "time.diff")
        )
        
        # 为每帧图像构建位姿
        colmap_images = {}
        img_id = 1
        
        for cam_name, files in image_results.items():
            # 找到对应的 camera_id
            cam_id_for_this = 1  # 默认
            
            # 从 .cam 文件中获取每帧时间戳
            cam_files = list((site_path / "IMG" / cam_name).glob("*.cam"))
            if cam_files:
                cam_data = parse_cam_file(str(cam_files[0]))
                with open(str(cam_files[0]), "rb") as f:
                    raw_data = f.read()
                
                for frame in cam_data.frames:
                    pose_dict = build_colmap_images_dict(
                        [frame.timestamp],
                        aligned_poses,
                        time_offset,
                    )
                    if pose_dict:
                        for _, pose_info in pose_dict.items():
                            colmap_images[img_id] = {
                                "name": frame.filename if frame.filename else f"{frame.index:06d}.jpg",
                                "qvec": pose_info["qvec"],
                                "tvec": pose_info["tvec"],
                                "camera_id": cam_id_for_this,
                            }
                            img_id += 1
        
        write_colmap_images_bin(colmap_images, str(sparse_dir / "images.bin"))
    else:
        # 不写入初始位姿，让 COLMAP SfM 自己算
        write_colmap_images_bin({}, str(sparse_dir / "images.bin"))
        logger.info("不写入初始位姿，COLMAP SfM 将自动计算")
    
    # Step 6: 写入 NeRF 格式的辅助文件
    logger.info("=== Step 6: 写入辅助文件 ===")
    write_cameras_json(colmap_cameras, str(out_path / "cameras.json"))
    
    # 写入转换元数据
    meta = {
        "site": site_path.name,
        "cameras": list(image_results.keys()),
        "total_images": sum(len(f) for f in image_results.values()),
        "colmap_cameras": {k: v for k, v in colmap_cameras.items()},
        "use_slam_poses": use_slam_poses and write_initial_poses,
    }
    with open(out_path / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"转换完成! 输出目录: {out_path}")
    logger.info(f"总图像数: {meta['total_images']}")
    logger.info(f"下一步:")
    logger.info(f"  1. cd {out_path}")
    logger.info(f"  2. colmap feature_extractor --database_path database.db --image_path images")
    logger.info(f"  3. colmap exhaustive_matcher --database_path database.db")
    logger.info(f"  4. colmap mapper --database_path database.db --image_path images --output_path sparse")
    
    return str(out_path)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="RS30 站点数据 → COLMAP 格式一键转换")
    parser.add_argument("site_dir", help="RS30 站点数据目录")
    parser.add_argument("-o", "--output", default=None, help="输出 COLMAP 项目目录")
    parser.add_argument("--cameras", nargs="*", help="指定相机")
    parser.add_argument("--max-frames", type=int, default=0, help="每相机最大帧数")
    parser.add_argument("--with-poses", action="store_true", help="写入 SLAM 位姿（否则 COLMAP 自己算）")
    
    args = parser.parse_args()
    
    site_dir = args.site_dir
    if args.output is None:
        args.output = str(Path(site_dir).parent / (Path(site_dir).name + "_colmap"))
    
    convert_site_to_colmap(
        site_dir=site_dir,
        output_dir=args.output,
        cameras=args.cameras,
        max_frames_per_camera=args.max_frames,
        write_initial_poses=args.with_poses,
    )
