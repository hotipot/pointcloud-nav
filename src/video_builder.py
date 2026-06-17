"""
视频生成模块

将帧序列合成为 MP4 视频，支持自定义帧率和分辨率。
"""

import os
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def build_video(
    frames: List[np.ndarray],
    output_path: str = "output/video.mp4",
    fps: float = 30.0,
    codec: str = "mp4v",
    verbose: bool = True,
) -> str:
    """
    将帧序列合成为 MP4 视频

    Args:
        frames: RGB 图像列表，每个为 (H, W, 3) uint8
        output_path: 输出视频路径
        fps: 帧率
        codec: 编码器（mp4v / avc1 / xvid）
        verbose: 是否打印进度

    Returns:
        输出视频的绝对路径
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "需要 OpenCV 来生成视频。请安装: pip install opencv-python"
        )

    if len(frames) == 0:
        raise ValueError("帧列表为空，无法生成视频")

    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    h, w = frames[0].shape[:2]

    # 确保所有帧尺寸一致
    consistent_frames = []
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, w):
            import cv2 as cv
            frame = cv.resize(frame, (w, h))
        consistent_frames.append(frame)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    for i, frame in enumerate(consistent_frames):
        # OpenCV 使用 BGR 格式
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame
        writer.write(bgr_frame)

        if verbose and (i + 1) % 100 == 0:
            logger.info(f"写入视频帧: {i + 1}/{len(consistent_frames)}")

    writer.release()

    abs_path = os.path.abspath(output_path)
    file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)

    if verbose:
        logger.info(f"视频已保存: {abs_path} ({file_size_mb:.1f} MB)")
        logger.info(f"  分辨率: {w}x{h}, 帧率: {fps}, 总帧数: {len(consistent_frames)}")
        logger.info(f"  时长: {len(consistent_frames) / fps:.2f}s")

    return abs_path


def frames_from_directory(
    directory: str,
    pattern: str = "%06d.jpg",
    start: int = 0,
    end: Optional[int] = None,
) -> List[np.ndarray]:
    """
    从目录读取帧图像序列

    Args:
        directory: 图像目录
        pattern: 文件名模式
        start: 起始帧号
        end: 结束帧号（None 读取全部）

    Returns:
        RGB 图像列表
    """
    import cv2

    frames = []
    i = start

    while True:
        if end is not None and i >= end:
            break

        filepath = os.path.join(directory, pattern % i)
        if not os.path.exists(filepath):
            break

        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        i += 1

    logger.info(f"从目录读取 {len(frames)} 帧: {directory}")
    return frames
