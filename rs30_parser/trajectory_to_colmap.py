"""
RS30 SLAM 轨迹 → COLMAP 格式转换器

将 RS30 系统的 SLAM 轨迹和 GNSS 轨迹转换为 COLMAP 格式的相机外参。

输入文件：
  - SLAM_DATA/trajectory3d.tra: SLAM 解算的 3D 轨迹（二进制）
  - SLAM_DATA/gnss_valid_trajectory.txt: GNSS 验证的轨迹（文本）
  - SLAM_DATA/utm_origin_pose.txt: UTM 原点位姿
  - GPS/Time/time.diff: 设备时间与 GPS 时间偏移

输出格式：COLMAP images.bin / images.txt
"""

import struct
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """6DoF 位姿"""
    timestamp: float  # Unix 时间戳
    position: np.ndarray  # [x, y, z] 世界坐标系
    quaternion: np.ndarray  # [qw, qx, qy, qz] 世界坐标系
    valid: bool = True  # GNSS 有效性标志


def parse_gnss_trajectory(filepath: str) -> List[Pose]:
    """解析 GNSS 验证轨迹文件
    
    格式: 帧号 时间戳 经度 纬度 高度 标志位
    10Hz 采样
    
    Args:
        filepath: gnss_valid_trajectory.txt 路径
        
    Returns:
        Pose 列表
    """
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            timestamp = float(parts[1])
            lon = float(parts[2])
            lat = float(parts[3])
            alt = float(parts[4])
            valid = int(parts[5]) == 1 if len(parts) > 5 else True
            
            poses.append(Pose(
                timestamp=timestamp,
                position=np.array([lon, lat, alt]),
                quaternion=np.array([1, 0, 0, 0]),  # GNSS 没有姿态
                valid=valid,
            ))
    
    logger.info(f"解析 GNSS 轨迹: {len(poses)} 帧")
    return poses


def parse_utm_origin(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """解析 UTM 原点位姿
    
    格式: 帧号 时间戳 经度 纬度 高度 qx qy qz qw
    
    Args:
        filepath: utm_origin_pose.txt 路径
        
    Returns:
        (position, quaternion): 原点的位置和姿态
    """
    with open(filepath, "r") as f:
        line = f.readline().strip()
        parts = line.split()
        
    position = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    quaternion = np.array([float(parts[8]), float(parts[5]), float(parts[6]), float(parts[7])])
    
    logger.info(f"UTM 原点: lon={position[0]:.8f}, lat={position[1]:.8f}, alt={position[2]:.2f}")
    return position, quaternion


def parse_time_diff(filepath: str) -> float:
    """解析时间偏移文件
    
    格式:
      realTime ready
      realTime: <seq> <gps_time>
      fakeTime: <seq> <device_time>
    
    Args:
        filepath: time.diff 路径
        
    Returns:
        设备时间与 GPS 时间的平均偏移（秒）
    """
    offsets = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("realTime:") and "fakeTime:" not in line:
                parts = line.split()
                if len(parts) >= 3:
                    real_time = float(parts[2])
            elif line.startswith("fakeTime:"):
                parts = line.split()
                if len(parts) >= 3:
                    fake_time = float(parts[2])
                    offsets.append(fake_time - real_time)
    
    if offsets:
        avg_offset = np.mean(offsets)
        logger.info(f"时间偏移: 平均 {avg_offset:.3f}s, 样本数 {len(offsets)}")
        return avg_offset
    return 0.0


def lonlat_to_utm(lon: float, lat: float) -> Tuple[float, float]:
    """经纬度转 UTM 坐标（简化版，不依赖 pyproj）
    
    使用中国区域 UTM zone 50N (适用于东经 102°-108°) 或 zone 49N
    保定在东经 115°，使用 zone 50N
    
    简化公式：近似将经纬度差转为米
    1° 经度 ≈ 111320 * cos(lat) 米
    1° 纬度 ≈ 110540 米
    """
    # 保定纬度约 38.8°
    lat_rad = np.radians(lat)
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    meters_per_deg_lat = 110540.0
    
    # 相对于参考点（0,0 经纬度）的偏移
    utm_x = lon * meters_per_deg_lon
    utm_y = lat * meters_per_deg_lat
    
    return utm_x, utm_y


def align_poses_to_local(
    gnss_poses: List[Pose],
    origin_position: np.ndarray,
) -> List[Pose]:
    """将 GNSS 轨迹从经纬度对齐到局部坐标系
    
    以 UTM 原点为局部坐标系原点，X 朝东，Y 朝北，Z 朝上
    
    Args:
        gnss_poses: GNSS 轨迹（经纬度）
        origin_position: UTM 原点位置（经纬度）
        
    Returns:
        局部坐标系的 Pose 列表
    """
    # 原点的 UTM 坐标
    origin_x, origin_y = lonlat_to_utm(origin_position[0], origin_position[1])
    origin_z = origin_position[2]
    
    aligned_poses = []
    for pose in gnss_poses:
        px, py = lonlat_to_utm(pose.position[0], pose.position[1])
        pz = pose.position[2]
        
        local_pos = np.array([
            px - origin_x,
            py - origin_y,
            pz - origin_z,
        ])
        
        aligned_poses.append(Pose(
            timestamp=pose.timestamp,
            position=local_pos,
            quaternion=pose.quaternion,
            valid=pose.valid,
        ))
    
    logger.info(f"对齐到局部坐标系: {len(aligned_poses)} 帧")
    return aligned_poses


def interpolate_pose_for_timestamp(
    poses: List[Pose],
    target_timestamp: float,
) -> Optional[Pose]:
    """为给定时间戳插值位姿
    
    Args:
        poses: 已排序的 Pose 列表（按时间戳递增）
        target_timestamp: 目标时间戳
        
    Returns:
        插值后的 Pose，如果超出范围则返回 None
    """
    if not poses:
        return None
    
    # 二分查找
    lo, hi = 0, len(poses) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if poses[mid].timestamp < target_timestamp:
            lo = mid + 1
        else:
            hi = mid
    
    # 边界情况
    if lo == 0 and poses[0].timestamp > target_timestamp:
        return None
    if lo == len(poses) - 1 and poses[-1].timestamp < target_timestamp:
        return None
    
    # 插值
    if poses[lo].timestamp > target_timestamp:
        # 在 lo-1 和 lo 之间
        i0, i1 = lo - 1, lo
    else:
        # 精确匹配 lo
        return poses[lo]
    
    t0 = poses[i0].timestamp
    t1 = poses[i1].timestamp
    alpha = (target_timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0
    
    interp_pos = poses[i0].position * (1 - alpha) + poses[i1].position * alpha
    
    # 四元数球面插值
    q0 = poses[i0].quaternion
    q1 = poses[i1].quaternion
    # 确保 q0 · q1 > 0（最短路径）
    if np.dot(q0, q1) < 0:
        q1 = -q1
    interp_q = slerp(q0, q1, alpha)
    
    return Pose(
        timestamp=target_timestamp,
        position=interp_pos,
        quaternion=interp_q,
        valid=poses[i0].valid and poses[i1].valid,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元数球面线性插值"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    if dot > 0.9995:
        # 非常接近时用线性插值
        result = q0 * (1 - t) + q1 * t
        return result / np.linalg.norm(result)
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta
    
    return s0 * q0 + s1 * q1


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 [qw, qx, qy, qz] → 3x3 旋转矩阵"""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    return R


def pose_to_colmap_extrinsic(pose: Pose) -> np.ndarray:
    """Pose → COLMAP 外参 4x4 矩阵
    
    COLMAP 外参: 世界到相机变换 (world-to-camera)
    即 P_cam = R * (P_world - t)，其中 R 是旋转，t 是平移
    
    Returns:
        4x4 变换矩阵
    """
    R = quaternion_to_rotation_matrix(pose.quaternion)
    
    # COLMAP convention: T = [R | -R*t; 0 0 0 1]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ pose.position
    
    return T


def build_colmap_images_dict(
    image_timestamps: List[float],  # 每帧图像的 Unix 时间戳
    aligned_poses: List[Pose],       # 已对齐到局部坐标系的位姿序列
    time_offset: float = 0.0,       # 设备时间偏移
) -> dict:
    """构建 COLMAP images 格式数据
    
    Args:
        image_timestamps: 图像帧的时间戳列表
        aligned_poses: 对齐后的位姿序列
        time_offset: 设备时间相对于 GPS 时间的偏移
        
    Returns:
        {image_id: {name, qvec, tvec}} 字典
    """
    # 修正时间偏移
    corrected_timestamps = [ts + time_offset for ts in image_timestamps]
    
    images = {}
    for i, ts in enumerate(corrected_timestamps):
        pose = interpolate_pose_for_timestamp(aligned_poses, ts)
        
        if pose is None:
            logger.warning(f"图像 {i} 时间戳 {ts:.3f} 无对应位姿，跳过")
            continue
        
        # COLMAP convention: qvec = [qw, qx, qy, qz], tvec = [tx, ty, tz]
        # 但 COLMAP 存储 world-to-camera 变换
        R = quaternion_to_rotation_matrix(pose.quaternion)
        tvec = -R @ pose.position
        
        images[i + 1] = {
            "name": f"{i:06d}.jpg",
            "qvec": pose.quaternion,  # [qw, qx, qy, qz]
            "tvec": tvec,
        }
    
    return images


if __name__ == "__main__":
    import argparse
    import json
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="RS30 SLAM 轨迹 → COLMAP 格式转换")
    parser.add_argument("site_dir", help="站点数据目录")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    site_path = Path(args.site_dir)
    
    # 解析 GNSS 轨迹
    gnss_poses = parse_gnss_trajectory(str(site_path / "SLAM_DATA" / "gnss_valid_trajectory.txt"))
    
    # 解析 UTM 原点
    origin_pos, origin_q = parse_utm_origin(str(site_path / "SLAM_DATA" / "utm_origin_pose.txt"))
    
    # 对齐到局部坐标系
    aligned_poses = align_poses_to_local(gnss_poses, origin_pos)
    
    # 解析时间偏移
    time_offset = parse_time_diff(str(site_path / "GPS" / "Time" / "time.diff"))
    
    # 输出信息
    result = {
        "total_gnss_poses": len(gnss_poses),
        "time_offset": time_offset,
        "origin": {
            "lon": origin_pos[0],
            "lat": origin_pos[1],
            "alt": origin_pos[2],
            "quaternion": origin_q.tolist(),
        },
        "aligned_poses_range": {
            "x_min": min(p.position[0] for p in aligned_poses),
            "x_max": max(p.position[0] for p in aligned_poses),
            "y_min": min(p.position[1] for p in aligned_poses),
            "y_max": max(p.position[1] for p in aligned_poses),
            "z_min": min(p.position[2] for p in aligned_poses),
            "z_max": max(p.position[2] for p in aligned_poses),
        },
    }
    
    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(out_json)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(out_json)
        logger.info(f"已保存到 {args.output}")
