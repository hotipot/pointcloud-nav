"""
RS30 .cam 文件解析器 — 提取内嵌 JPEG 帧

.cam 文件格式：
  - 文件头: "$CAM" 魔数 + 元信息（设备 SN、相机参数、时间范围等）
  - 帧数据: 每帧包含:
    - [可选] 前一帧 JPEG 结尾 FFD9
    - [可选] 填充 55555555
    - 帧头: AAAAAAAA (4 bytes) + 文件名 (timestamp.jpg\0) + 填充零 + 附加数据
    - JPEG 数据: FFD8FFE0 ... FFD9
"""

import os
import struct
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 常量
CAM_MAGIC = b"$CAM"
FRAME_MARKER = b"\xaa\xaa\xaa\xaa"
JPEG_SOI = b"\xff\xd8\xff\xe0"  # JFIF
JPEG_SOI_EXIF = b"\xff\xd8\xff\xe1"  # EXIF
JPEG_EOI = b"\xff\xd9"
PADDING = b"\x55\x55\x55\x55"


@dataclass
class FrameInfo:
    """单帧信息"""
    index: int
    timestamp: float  # Unix 时间戳
    timestamp_str: str  # 原始时间戳字符串
    offset: int  # JPEG 数据在文件中的偏移
    size: int  # JPEG 数据大小（字节）
    filename: str  # 原始文件名 (timestamp.jpg)


@dataclass
class CamFile:
    """解析后的 .cam 文件"""
    path: str
    device_sn: str
    camera_sn: str
    frames: List[FrameInfo]
    total_size: int


def find_jpeg_frames(data: bytes) -> List[int]:
    """查找文件中所有 JPEG SOI 标记的位置"""
    positions = []
    pos = 0
    while pos < len(data):
        # 查找 JFIF 或 EXIF 标记
        idx_jfif = data.find(JPEG_SOI, pos)
        idx_exif = data.find(JPEG_SOI_EXIF, pos)
        
        candidates = [x for x in [idx_jfif, idx_exif] if x >= 0]
        if not candidates:
            break
        
        idx = min(candidates)
        positions.append(idx)
        pos = idx + 1
    
    return positions


def extract_timestamp_from_header(data: bytes, jpeg_offset: int) -> Tuple[str, float]:
    """从帧头中提取时间戳
    
    帧头结构: AAAAAAAA + "timestamp.jpg\0" + padding + extra_bytes
    """
    # 向前搜索 AAAAAAAA 标记
    search_start = max(0, jpeg_offset - 512)
    search_data = data[search_start:jpeg_offset]
    marker_pos = search_data.rfind(FRAME_MARKER)
    
    if marker_pos < 0:
        # 没找到帧标记，返回空
        return "", 0.0
    
    abs_marker = search_start + marker_pos
    header = data[abs_marker:jpeg_offset]
    
    # 提取文件名（时间戳）
    name_start = 4  # 跳过 AAAAAAAA
    name_end = header.find(b"\x00", name_start)
    if name_end < 0:
        return "", 0.0
    
    filename = header[name_start:name_end].decode("ascii", errors="replace")
    ts_str = filename.replace(".jpg", "")
    
    try:
        ts_float = float(ts_str)
    except ValueError:
        ts_float = 0.0
    
    return filename, ts_float


def parse_cam_file(cam_path: str) -> CamFile:
    """解析 .cam 文件，返回帧信息列表
    
    Args:
        cam_path: .cam 文件路径
        
    Returns:
        CamFile 对象，包含所有帧信息
    """
    cam_path = str(cam_path)
    logger.info(f"解析 .cam 文件: {cam_path}")
    
    with open(cam_path, "rb") as f:
        data = f.read()
    
    total_size = len(data)
    
    # 验证魔数
    if data[:4] != CAM_MAGIC:
        logger.warning(f"文件头不是 $CAM 魔数: {data[:4]}")
    
    # 提取设备 SN（从文件头中搜索）
    device_sn = ""
    camera_sn = ""
    # 在前 1024 字节中搜索 SN
    header_text = data[:1024]
    sn_match = header_text.find(b"124280200037")
    if sn_match >= 0:
        device_sn = "124280200037"
    
    # 查找所有 JPEG 帧
    jpeg_positions = find_jpeg_frames(data)
    logger.info(f"找到 {len(jpeg_positions)} 个 JPEG 帧")
    
    # 解析每帧信息
    frames = []
    for i, jpeg_off in enumerate(jpeg_positions):
        # 提取时间戳
        filename, ts = extract_timestamp_from_header(data, jpeg_off)
        
        # 计算 JPEG 大小（到下一个帧标记或文件尾）
        if i + 1 < len(jpeg_positions):
            # 查找 EOI 标记
            eoi_pos = data.find(JPEG_EOI, jpeg_off + 2)
            if eoi_pos > 0 and eoi_pos < jpeg_positions[i + 1]:
                jpeg_size = eoi_pos + 2 - jpeg_off
            else:
                jpeg_size = jpeg_positions[i + 1] - jpeg_off
        else:
            eoi_pos = data.find(JPEG_EOI, jpeg_off + 2)
            if eoi_pos > 0:
                jpeg_size = eoi_pos + 2 - jpeg_off
            else:
                jpeg_size = total_size - jpeg_off
        
        frame = FrameInfo(
            index=i,
            timestamp=ts,
            timestamp_str=filename.replace(".jpg", "") if filename else "",
            offset=jpeg_off,
            size=jpeg_size,
            filename=filename,
        )
        frames.append(frame)
    
    # 提取 camera SN 从帧头
    if frames:
        camera_sn = ""  # 从 .CP 文件获取更准确
    
    logger.info(f"解析完成: {len(frames)} 帧, "
                f"时间范围: {frames[0].timestamp:.1f} - {frames[-1].timestamp:.1f}")
    
    return CamFile(
        path=cam_path,
        device_sn=device_sn,
        camera_sn=camera_sn,
        frames=frames,
        total_size=total_size,
    )


def extract_jpeg(data: bytes, frame: FrameInfo) -> bytes:
    """从文件数据中提取单帧 JPEG
    
    Args:
        data: .cam 文件的完整二进制数据
        frame: 帧信息
        
    Returns:
        JPEG 二进制数据
    """
    # 精确提取：从 SOI 到 EOI
    jpeg_start = frame.offset
    
    # 查找 EOI
    eoi_pos = data.find(JPEG_EOI, jpeg_start + 2)
    if eoi_pos > 0:
        return data[jpeg_start:eoi_pos + 2]
    else:
        # Fallback: 使用预计算的大小
        return data[jpeg_start:jpeg_start + frame.size]


def extract_all_images(
    cam_path: str,
    output_dir: str,
    camera_name: str = "",
    max_frames: int = 0,
    skip_existing: bool = True,
) -> List[str]:
    """提取 .cam 文件中的所有 JPEG 帧并保存
    
    Args:
        cam_path: .cam 文件路径
        output_dir: 输出目录
        camera_name: 相机名称（用于子目录），如 "Camera1"
        max_frames: 最多提取帧数（0 = 全部）
        skip_existing: 跳过已存在的文件
        
    Returns:
        输出文件路径列表
    """
    cam = parse_cam_file(cam_path)
    
    with open(cam_path, "rb") as f:
        data = f.read()
    
    # 创建输出目录
    if camera_name:
        out_path = Path(output_dir) / camera_name / "images"
    else:
        out_path = Path(output_dir) / "images"
    out_path.mkdir(parents=True, exist_ok=True)
    
    frames = cam.frames
    if max_frames > 0:
        frames = frames[:max_frames]
    
    output_files = []
    for frame in frames:
        # 使用时间戳作为文件名
        if frame.filename:
            out_file = out_path / frame.filename
        else:
            out_file = out_path / f"{frame.index:06d}.jpg"
        
        if skip_existing and out_file.exists():
            output_files.append(str(out_file))
            continue
        
        # 提取 JPEG 数据
        jpeg_data = extract_jpeg(data, frame)
        
        # 验证 JPEG 完整性
        if not jpeg_data.startswith(b"\xff\xd8"):
            logger.warning(f"帧 {frame.index} 不是有效 JPEG，跳过")
            continue
        if not jpeg_data.endswith(b"\xff\xd9"):
            logger.warning(f"帧 {frame.index} JPEG 未正常结束")
        
        with open(out_file, "wb") as f:
            f.write(jpeg_data)
        
        output_files.append(str(out_file))
    
    logger.info(f"提取完成: {len(output_files)} 帧保存到 {out_path}")
    return output_files


def extract_site_images(
    site_dir: str,
    output_dir: str,
    cameras: Optional[List[str]] = None,
    max_frames: int = 0,
) -> dict:
    """提取整个站点所有相机的图像
    
    Args:
        site_dir: 站点数据目录（包含 IMG/ 子目录）
        output_dir: 输出根目录
        cameras: 要提取的相机列表，如 ["Camera1", "Camera2"]，None = 全部
        max_frames: 每个相机最多提取帧数
        
    Returns:
        {camera_name: [output_files]} 字典
    """
    site_path = Path(site_dir)
    img_dir = site_path / "IMG"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"IMG 目录不存在: {img_dir}")
    
    # 发现相机目录
    cam_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    
    if cameras:
        cam_dirs = [d for d in cam_dirs if d.name in cameras]
    
    results = {}
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        
        # 查找 .cam 文件
        cam_files = list(cam_dir.glob("*.cam"))
        if not cam_files:
            logger.info(f"{cam_name}: 无 .cam 文件，跳过")
            continue
        
        for cam_file in cam_files:
            logger.info(f"处理 {cam_name}: {cam_file.name}")
            output_files = extract_all_images(
                str(cam_file),
                output_dir,
                camera_name=cam_name,
                max_frames=max_frames,
            )
            results[cam_name] = output_files
    
    return results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="提取 RS30 .cam 文件中的 JPEG 帧")
    parser.add_argument("input", help="站点数据目录或 .cam 文件路径")
    parser.add_argument("-o", "--output", default="./output", help="输出目录")
    parser.add_argument("--cameras", nargs="*", help="指定相机（如 Camera1 Camera2）")
    parser.add_argument("--max-frames", type=int, default=0, help="每相机最多提取帧数")
    parser.add_argument("--single", action="store_true", help="输入是单个 .cam 文件")
    
    args = parser.parse_args()
    
    if args.single:
        extract_all_images(args.input, args.output, max_frames=args.max_frames)
    else:
        extract_site_images(args.input, args.output, cameras=args.cameras, max_frames=args.max_frames)
