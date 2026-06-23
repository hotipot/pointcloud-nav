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

SuGaR 兼容性说明：
  SuGaR (3DGS) 只接受 SIMPLE_PINHOLE / PINHOLE 模型（不支持 OPENCV 畸变）。
  因此，当相机有畸变时，需要：
    方案 A（推荐）：--undistort，用 OpenCV 去畸变图像，输出 PINHOLE 模型
    方案 B：先输出 OPENCV 模型，然后手动运行 colmap image_undistorter
    方案 C：忽略畸变，直接用 PINHOLE 模型（精度有损，不推荐）
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
                "SIMPLE_PINHOLE": 1,
                "PINHOLE": 0,
                "OPENCV": 2,
            }.get(model_name, 0)
            
            f.write(struct.pack("<IQQ", cam_id, model_id, width))
            f.write(struct.pack("<Q", height))
            f.write(struct.pack("<Q", len(params)))
            for p in params:
                f.write(struct.pack("<d", p))


def write_colmap_images_bin(images: dict, output_path: str):
    """写入 COLMAP images.bin
    
    Args:
        images: {image_id: {name, qvec, tvec, camera_id, xys, point3D_ids}}
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


def undistort_images(
    image_dir: str,
    output_dir: str,
    intrinsics: CameraIntrinsics,
    camera_name: str = "",
) -> dict:
    """使用 OpenCV 对图像去畸变，输出 PINHOLE 模型
    
    Args:
        image_dir: 原始图像目录
        output_dir: 去畸变后图像输出目录
        intrinsics: 相机内参（含畸变系数）
        camera_name: 相机名称
        
    Returns:
        去畸变后的内参 dict（PINHOLE 模型）
    """
    import cv2
    from PIL import Image
    
    src_dir = Path(image_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建相机矩阵和畸变系数
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ], dtype=np.float64)
    
    dist_coeffs = np.array([
        intrinsics.k1, intrinsics.k2,
        intrinsics.p1, intrinsics.p2,
        intrinsics.k3,  # k3
    ], dtype=np.float64)
    
    # 计算去畸变后的新相机矩阵
    # alpha=0 裁剪到无黑边区域，alpha=1 保留所有像素
    img_size = (intrinsics.image_width, intrinsics.image_height)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, img_size, alpha=0)
    
    # 计算去畸变映射（只需一次）
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, img_size, cv2.CV_16SC2,
    )
    
    # 裁剪后的有效区域
    x, y, w, h = roi
    logger.info(f"  去畸变后有效区域: x={x}, y={y}, w={w}, h={h}")
    logger.info(f"  原始内参: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
                f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")
    logger.info(f"  去畸变内参: fx={new_K[0,0]:.2f}, fy={new_K[1,1]:.2f}, "
                f"cx={new_K[0,2]:.2f}, cy={new_K[1,2]:.2f}")
    
    # 处理每张图像
    images = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")))
    logger.info(f"  去畸变处理 {len(images)} 张图像...")
    
    processed = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"  无法读取: {img_path.name}")
            continue
        
        # 应用去畸变
        dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        # 裁剪到有效区域（去除黑边）
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        # 保存
        out_path = dst_dir / img_path.name
        cv2.imwrite(str(out_path), dst, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed += 1
    
    logger.info(f"  去畸变完成: {processed}/{len(images)} 张")
    
    # 返回去畸变后的 PINHOLE 内参
    # 如果裁剪了，需要调整内参和尺寸
    final_w = w if (w > 0 and h > 0) else intrinsics.image_width
    final_h = h if (w > 0 and h > 0) else intrinsics.image_height
    final_fx = new_K[0, 0]
    final_fy = new_K[1, 1]
    final_cx = new_K[0, 2] - x  # 裁剪后调整主点
    final_cy = new_K[1, 2] - y
    
    return {
        "model": "PINHOLE",
        "width": final_w,
        "height": final_h,
        "params": [final_fx, final_fy, final_cx, final_cy],
    }


def convert_site_to_colmap(
    site_dir: str,
    output_dir: str,
    cameras: Optional[List[str]] = None,
    max_frames_per_camera: int = 0,
    use_slam_poses: bool = True,
    write_initial_poses: bool = False,
    undistort: bool = True,
    force_pinhole: bool = False,
) -> str:
    """一键转换站点数据为 COLMAP 项目
    
    Args:
        site_dir: RS30 站点数据目录
        output_dir: 输出 COLMAP 项目目录
        cameras: 指定相机列表（None = 全部 .cam 相机）
        max_frames_per_camera: 每相机最大帧数（0 = 全部）
        use_slam_poses: 是否使用 SLAM 位姿（True 用 SLAM，False 让 COLMAP 自己算）
        write_initial_poses: 是否写入初始位姿到 images.bin
        undistort: 是否对图像去畸变（推荐 True，SuGaR 需要 PINHOLE 模型）
        force_pinhole: 忽略畸变，强制使用 PINHOLE 模型（不推荐，精度有损）
        
    Returns:
        输出目录路径
    """
    site_path = Path(site_dir)
    out_path = Path(output_dir)
    
    logger.info(f"转换站点: {site_path.name}")
    logger.info(f"输出目录: {out_path}")
    logger.info(f"去畸变模式: {undistort}")
    
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
    cam_name_to_id = {}
    
    for cam_name, intr in intrinsics.items():
        cam_name_to_id[cam_name] = cam_id
        
        if undistort and (abs(intr.k1) > 1e-6 or abs(intr.k2) > 1e-6):
            # 有畸变 + 需要去畸变：先占位，去畸变后更新
            colmap_cameras[cam_id] = intrinsics_to_colmap_dict(intr)  # 临时
            logger.info(f"  {cam_name} → camera_id={cam_id}, model=OPENCV (将去畸变为 PINHOLE)")
        elif force_pinhole:
            # 强制 PINHOLE：忽略畸变
            colmap_cameras[cam_id] = {
                "model": "PINHOLE",
                "width": intr.image_width,
                "height": intr.image_height,
                "params": [intr.fx, intr.fy, intr.cx, intr.cy],
            }
            logger.info(f"  {cam_name} → camera_id={cam_id}, model=PINHOLE (强制，忽略畸变)")
        else:
            colmap_cameras[cam_id] = intrinsics_to_colmap_dict(intr)
            logger.info(f"  {cam_name} → camera_id={cam_id}, model={colmap_cameras[cam_id]['model']}")
        cam_id += 1
    
    # 为没有 .CP 内参的相机添加默认参数
    for cam_name in image_results:
        if cam_name not in intrinsics:
            cam_name_to_id[cam_name] = cam_id
            default_intr = CameraIntrinsics(
                fx=1018.09, fy=1018.09,
                cx=972.0, cy=1296.0,
                image_width=1944, image_height=2592,
            )
            colmap_cameras[cam_id] = intrinsics_to_colmap_dict(default_intr)
            logger.info(f"  {cam_name} → camera_id={cam_id} (默认内参)")
            cam_id += 1
    
    # Step 4: 去畸变（如果需要）
    if undistort:
        logger.info("=== Step 4: 图像去畸变 ===")
        for cam_name, intr in intrinsics.items():
            if abs(intr.k1) > 1e-6 or abs(intr.k2) > 1e-6:
                cam_image_dir = out_path / cam_name / "images"
                if not cam_image_dir.exists():
                    logger.warning(f"  {cam_name} 图像目录不存在，跳过去畸变")
                    continue
                
                # 去畸变输出到同目录（覆盖原图）
                # 先备份原图到 raw/ 目录
                raw_dir = out_path / cam_name / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                
                import shutil
                for img_file in cam_image_dir.glob("*.jpg"):
                    shutil.move(str(img_file), str(raw_dir / img_file.name))
                for img_file in cam_image_dir.glob("*.JPG"):
                    shutil.move(str(img_file), str(raw_dir / img_file.name))
                
                logger.info(f"  {cam_name}: 原图已备份到 {raw_dir}")
                
                # 去畸变处理
                new_cam = undistort_images(
                    image_dir=str(raw_dir),
                    output_dir=str(cam_image_dir),
                    intrinsics=intr,
                    camera_name=cam_name,
                )
                
                # 更新 COLMAP 相机参数为 PINHOLE
                cid = cam_name_to_id[cam_name]
                colmap_cameras[cid] = new_cam
                logger.info(f"  {cam_name} 更新为 PINHOLE: fx={new_cam['params'][0]:.2f}, "
                            f"fy={new_cam['params'][1]:.2f}, "
                            f"cx={new_cam['params'][2]:.2f}, cy={new_cam['params'][3]:.2f}")
                logger.info(f"  {cam_name} 新尺寸: {new_cam['width']}x{new_cam['height']}")
    
    # Step 5: 写入 COLMAP 二进制文件
    logger.info("=== Step 5: 写入 COLMAP 文件 ===")
    sparse_dir = out_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    write_colmap_cameras_bin(colmap_cameras, str(sparse_dir / "cameras.bin"))
    write_colmap_points3d_bin(str(sparse_dir / "points3D.bin"))
    
    if write_initial_poses and use_slam_poses:
        # Step 6: 使用 SLAM 位姿
        logger.info("=== Step 6: 构建 SLAM 位姿 ===")
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
            cam_id_for_this = cam_name_to_id.get(cam_name, 1)
            
            cam_files = list((site_path / "IMG" / cam_name).glob("*.cam"))
            if cam_files:
                cam_data = parse_cam_file(str(cam_files[0]))
                
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
    
    # Step 7: 写入 NeRF 格式的辅助文件
    logger.info("=== Step 7: 写入辅助文件 ===")
    write_cameras_json(colmap_cameras, str(out_path / "cameras.json"))
    
    # 写入转换元数据
    meta = {
        "site": site_path.name,
        "cameras": list(image_results.keys()),
        "total_images": sum(len(f) for f in image_results.values()),
        "colmap_cameras": {str(k): v for k, v in colmap_cameras.items()},
        "use_slam_poses": use_slam_poses and write_initial_poses,
        "undistorted": undistort,
        "force_pinhole": force_pinhole,
    }
    with open(out_path / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # 写入 SuGaR 友好的 images/ 目录（flat 结构）
    # SuGaR/3DGS 期望 images/ 下直接是图像文件
    logger.info("=== Step 8: 构建 SuGaR 友好的 images 目录 ===")
    sugar_images_dir = out_path / "images"
    sugar_images_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_name in image_results:
        cam_img_dir = out_path / cam_name / "images"
        if not cam_img_dir.exists():
            continue
        
        import shutil
        for img_file in sorted(cam_img_dir.glob("*.jpg")) + sorted(cam_img_dir.glob("*.JPG")):
            # 重命名为 Camera1_xxxxxx.jpg 避免冲突
            new_name = f"{cam_name}_{img_file.name}"
            dst = sugar_images_dir / new_name
            if not dst.exists():
                shutil.copy2(str(img_file), str(dst))
    
    logger.info(f"转换完成! 输出目录: {out_path}")
    logger.info(f"总图像数: {meta['total_images']}")
    
    # 输出下一步指引
    logger.info("=" * 60)
    logger.info("下一步操作:")
    logger.info("=" * 60)
    logger.info(f"  方案 A: 使用 COLMAP SfM 从头计算（推荐）")
    logger.info(f"  1. colmap feature_extractor \\")
    logger.info(f"       --database_path {out_path}/database.db \\")
    logger.info(f"       --image_path {out_path}/images \\")
    logger.info(f"       --ImageReader.camera_model PINHOLE \\")
    logger.info(f"       --ImageReader.single_camera_per_folder 1")
    logger.info(f"  2. colmap exhaustive_matcher \\")
    logger.info(f"       --database_path {out_path}/database.db")
    logger.info(f"  3. colmap mapper \\")
    logger.info(f"       --database_path {out_path}/database.db \\")
    logger.info(f"       --image_path {out_path}/images \\")
    logger.info(f"       --output_path {out_path}/sparse")
    logger.info(f"")
    logger.info(f"  方案 B: 直接使用 SuGaR 全流程")
    logger.info(f"  python train_full_pipeline.py -s {out_path} -r dn_consistency --high_poly True")
    
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
    parser.add_argument("--no-undistort", action="store_true", 
                        help="不去畸变图像（保留 OPENCV 模型，SuGaR 不支持）")
    parser.add_argument("--force-pinhole", action="store_true",
                        help="忽略畸变，强制 PINHOLE 模型（不推荐，精度有损）")
    
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
        undistort=not args.no_undistort,
        force_pinhole=args.force_pinhole,
    )
