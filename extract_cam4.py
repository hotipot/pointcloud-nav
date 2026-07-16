#!/usr/bin/env python3
"""从 Camera4 全景视频提取帧，生成 COLMAP 兼容的等距柱状投影图像

使用已拼接的 .mp4（1920x1080 equirectangular），按指定间隔提取帧。
COLMAP EQUIRECTANGULAR 模型不需要焦距参数，只需要图像宽高。
"""

import argparse
import subprocess
import json
import os
from pathlib import Path


def extract_panoramic_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    prefix: str = "Camera4",
):
    """从全景视频提取帧
    
    Args:
        video_path: 已拼接的全景 MP4 路径
        output_dir: 输出图像目录
        fps: 提取帧率（1.0 = 每秒1帧）
        prefix: 文件名前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 用 ffmpeg 提取帧
    output_pattern = str(output_dir / f"{prefix}_%06d.jpg")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # 高质量 JPEG
        "-y",
        output_pattern,
    ]
    
    print(f"[extract] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed:\n{result.stderr}")
        return 0
    
    # 统计提取的帧数
    frames = sorted(output_dir.glob(f"{prefix}_*.jpg"))
    print(f"[extract] Extracted {len(frames)} frames at {fps} fps")
    
    # 重命名为时间戳格式（与其他 camera 一致）
    # 从视频元数据获取起始时间
    video_start_time = get_video_start_time(video_path)
    
    renamed = 0
    for i, frame_path in enumerate(frames):
        # 计算时间戳（秒）
        timestamp_s = i / fps
        # 使用毫秒级时间戳作为文件名（与 Camera1/2/3 格式一致）
        ts_ms = int(video_start_time + timestamp_s * 1000)
        new_name = f"{prefix}_{ts_ms}.jpg"
        new_path = frame_path.parent / new_name
        frame_path.rename(new_path)
        renamed += 1
    
    print(f"[extract] Renamed {renamed} frames with timestamp format")
    return renamed


def get_video_start_time(video_path: str) -> int:
    """从视频元数据获取起始时间戳（毫秒）
    
    如果无法获取，返回一个默认值（与 Camera1 起始时间接近）
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            # 尝试从 creation_time 获取
            tags = info.get("format", {}).get("tags", {})
            creation_time = tags.get("creation_time", "")
            if creation_time:
                # 2026-06-13T02:15:19.000000Z
                # 我们用 Camera1 的时间戳格式：1781316971431163 (微秒)
                # Camera1 第一帧时间: 1781316971_431163
                # 全景视频起始时间: 2026-06-13T02:15:19 UTC
                # 转换为 unix timestamp 毫秒
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                ts_ms = int(dt.timestamp() * 1000)
                return ts_ms
    except Exception as e:
        print(f"[WARN] Could not get video start time: {e}")
    
    # 回退：使用 Camera1 第一帧的时间戳（近似）
    # Camera1_1781316971_431163 → 1781316971431 毫秒
    return 1781316971431


def generate_colmap_camera(width: int, height: int) -> dict:
    """生成 COLMAP EQUIRECTANGULAR 相机模型
    
    EQUIRECTANGULAR 模型参数: [] (空，不需要焦距)
    COLMAP 会根据图像宽高自动计算
    """
    return {
        "model": "EQUIRECTANGULAR",
        "width": width,
        "height": height,
        "params": [],  # EQUIRECTANGULAR 不需要内参参数
    }


def main():
    parser = argparse.ArgumentParser(description="从 Camera4 全景视频提取帧")
    parser.add_argument("--video", type=str,
                        default="/home/wm1/jwang/dataset/baoding/2026-06-13-BD-GLKGQ/IMG/Camera4/VID_20260613_101519_00_024.mp4",
                        help="全景视频路径")
    parser.add_argument("--output", type=str,
                        default="/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam/Camera4/images",
                        help="输出目录")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="提取帧率（默认 1.0，即每秒1帧）")
    parser.add_argument("--prefix", type=str, default="Camera4",
                        help="文件名前缀")
    args = parser.parse_args()
    
    n = extract_panoramic_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        prefix=args.prefix,
    )
    
    if n > 0:
        # 生成相机模型信息
        from PIL import Image
        sample = sorted(Path(args.output).glob(f"{args.prefix}_*.jpg"))[0]
        img = Image.open(sample)
        cam = generate_colmap_camera(img.width, img.height)
        
        cam_json = {
            "id": 4,
            **cam,
        }
        
        cam_path = Path(args.output).parent / "camera4_info.json"
        with open(cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)
        print(f"[info] Camera model saved to {cam_path}")
        print(f"[info] Camera model: {cam['model']}, size: {cam['width']}x{cam['height']}")
        print(f"[info] Total frames: {n}")


if __name__ == "__main__":
    main()
