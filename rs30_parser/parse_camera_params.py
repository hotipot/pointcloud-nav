"""
RS30 .CP 文件解析器 — 提取相机内参

.CP 文件格式: Protobuf 编码，包含：
  - 设备 SN、相机 SN、相机型号
  - 相机内参: fx, fy, cx, cy, 畸变系数
  - IMU-LiDAR-相机 外参标定
  - 采集配置参数

关键字段映射（基于逆向分析）:
  Field 1: 设备信息消息
    Field 1 (string): 设备 SN (如 "124280200037")
    Field 2 (string): 相机 SN (如 "c292428037")
    Field 3 (string): 相机型号 (如 "Camera_HO")
    Field 4 (string): 图像尺寸代码 (如 "11" → 1944x2592)
    Field 5 (double): 开始时间戳
    Field 6 (double): 结束时间戳
    Field 8: 相机内参消息
      Field 1 (double): fx (焦距 x, 像素)
      Field 2 (double): fy (焦距 y, 像素)
      Field 3 (double): cx (主点 x)
      Field 4 (double): cy (主点 y)
      Field 5-7 (double): 未知内参（可能为 skew / 其他）
    Field 9: 畸变参数消息
      Field 1 (varint): 畸变模型标志 (255 = 全模型)
      Field 2 (double): k1 (径向畸变)
      Field 3 (double): k2
      Field 4 (double): p1 (切向畸变)
      Field 5 (double): p2
      Field 6-18 (double): k3-k15 (高阶畸变，大多为0)
    Field 10: 外参标定消息
      Field 1-3 (double): 角度偏移 (roll/pitch/yaw 初始值)
      Field 4 (double): LiDAR-camera 时间偏移
      Field 5-12 (double): 外参旋转/平移参数
    Field 19 (float): 采样频率 (Hz)
    Field 20 (float): 曝光时间 (ms)
"""

import struct
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    # 畸变系数
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0  # 高阶径向畸变
    # 元信息
    camera_model: str = ""
    device_sn: str = ""
    camera_sn: str = ""
    image_width: int = 0
    image_height: int = 0
    sample_rate: float = 0.0  # Hz
    exposure_ms: float = 0.0


# 图像尺寸代码映射（基于 RS30 设备）
IMAGE_SIZE_CODES = {
    "11": (1944, 2592),  # 实测确认
    # 更多尺寸代码待发现
}


def decode_varint(data: bytes, pos: int) -> tuple:
    """解码 Protobuf varint"""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def parse_protobuf_message(data: bytes) -> dict:
    """解析 Protobuf 消息为字段字典
    
    Returns:
        {field_number: [(wire_type, value), ...]} 字典
    """
    fields = {}
    pos = 0
    
    while pos < len(data):
        try:
            tag, pos = decode_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x07
            
            if field_number not in fields:
                fields[field_number] = []
            
            if wire_type == 0:  # varint
                value, pos = decode_varint(data, pos)
                fields[field_number].append(("varint", value))
            elif wire_type == 1:  # 64-bit (double)
                value = struct.unpack_from("<d", data, pos)[0]
                pos += 8
                fields[field_number].append(("double", value))
            elif wire_type == 2:  # length-delimited
                length, pos = decode_varint(data, pos)
                value = data[pos:pos + length]
                pos += length
                # 尝试解码为 UTF-8 字符串
                try:
                    s = value.decode("utf-8")
                    if len(s) < 500 and all(32 <= ord(c) < 127 or c in "\n\r\t" for c in s):
                        fields[field_number].append(("string", s))
                        continue
                except:
                    pass
                fields[field_number].append(("bytes", value))
            elif wire_type == 5:  # 32-bit (float)
                value = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                fields[field_number].append(("float", value))
            else:
                logger.warning(f"未知 wire_type={wire_type} at offset {pos}")
                break
        except Exception as e:
            logger.warning(f"解析错误 at offset {pos}: {e}")
            break
    
    return fields


def extract_intrinsics_from_fields(fields: dict) -> CameraIntrinsics:
    """从解析的 Protobuf 字段中提取相机内参"""
    
    intrinsics = CameraIntrinsics()
    
    # 基本元信息
    if 1 in fields:
        for _, val in fields[1]:
            if isinstance(val, str):
                intrinsics.device_sn = val
    
    if 2 in fields:
        for _, val in fields[2]:
            if isinstance(val, str):
                intrinsics.camera_sn = val
    
    if 3 in fields:
        for _, val in fields[3]:
            if isinstance(val, str):
                intrinsics.camera_model = val
    
    if 4 in fields:
        for _, val in fields[4]:
            if isinstance(val, str):
                size_code = val
                if size_code in IMAGE_SIZE_CODES:
                    intrinsics.image_width, intrinsics.image_height = IMAGE_SIZE_CODES[size_code]
    
    # 相机内参 (Field 8)
    if 8 in fields:
        for wire_type, val in fields[8]:
            if isinstance(val, bytes):
                inner = parse_protobuf_message(val)
                # fx, fy, cx, cy
                intrinsics.fx = inner.get(1, [("double", 0.0)])[-1][1]
                intrinsics.fy = inner.get(2, [("double", 0.0)])[-1][1]
                intrinsics.cx = inner.get(3, [("double", 0.0)])[-1][1]
                intrinsics.cy = inner.get(4, [("double", 0.0)])[-1][1]
    
    # 畸变参数 (Field 9)
    if 9 in fields:
        for wire_type, val in fields[9]:
            if isinstance(val, bytes):
                inner = parse_protobuf_message(val)
                intrinsics.k1 = inner.get(2, [("double", 0.0)])[-1][1]
                intrinsics.k2 = inner.get(3, [("double", 0.0)])[-1][1]
                intrinsics.p1 = inner.get(4, [("double", 0.0)])[-1][1]
                intrinsics.p2 = inner.get(5, [("double", 0.0)])[-1][1]
                intrinsics.k3 = inner.get(6, [("double", 0.0)])[-1][1]
    
    # 采集参数
    if 19 in fields:
        for wire_type, val in fields[19]:
            if wire_type == "float":
                intrinsics.sample_rate = val
    
    if 20 in fields:
        for wire_type, val in fields[20]:
            if wire_type == "float":
                intrinsics.exposure_ms = val
    
    return intrinsics


def parse_cp_file(cp_path: str) -> CameraIntrinsics:
    """解析 .CP 文件，返回相机内参
    
    Args:
        cp_path: .CP 文件路径
        
    Returns:
        CameraIntrinsics 对象
    """
    logger.info(f"解析 .CP 文件: {cp_path}")
    
    with open(cp_path, "rb") as f:
        data = f.read()
    
    # .CP 文件可能包含多条 protobuf 消息
    # 第一条通常是初始标定，第二条是最终标定（含更多数据）
    # 顶层 Field 1 是嵌套消息，包含所有实际数据
    top_fields = parse_protobuf_message(data)
    
    # 解析嵌套消息
    if 1 in top_fields:
        for wire_type, val in top_fields[1]:
            if isinstance(val, bytes):
                fields = parse_protobuf_message(val)
                break
        else:
            fields = top_fields
    else:
        fields = top_fields
    
    intrinsics = extract_intrinsics_from_fields(fields)
    
    # 如果没有尺寸信息，从内参推断
    if intrinsics.image_width == 0 and intrinsics.cy > 0:
        # cx/cy 通常接近图像中心
        intrinsics.image_width = int(intrinsics.cx * 2)
        intrinsics.image_height = int(intrinsics.cy * 2)
    
    logger.info(f"相机内参: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
                f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")
    logger.info(f"畸变: k1={intrinsics.k1:.4f}, k2={intrinsics.k2:.4f}, "
                f"p1={intrinsics.p1:.4f}, p2={intrinsics.p2:.4f}")
    
    return intrinsics


def parse_all_cameras(site_dir: str) -> dict:
    """解析站点所有相机的内参
    
    Args:
        site_dir: 站点数据目录
        
    Returns:
        {camera_name: CameraIntrinsics} 字典
    """
    site_path = Path(site_dir)
    img_dir = site_path / "IMG"
    
    results = {}
    cam_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        cp_files = list(cam_dir.glob("*.CP"))
        
        if not cp_files:
            logger.info(f"{cam_name}: 无 .CP 文件")
            continue
        
        intrinsics = parse_cp_file(str(cp_files[0]))
        results[cam_name] = intrinsics
    
    return results


def intrinsics_to_colmap_dict(intrinsics: CameraIntrinsics) -> dict:
    """转换为 COLMAP 相机模型参数
    
    COLMAP PINHOLE 模型: fx, fy, cx, cy
    COLMAP OPENCV 模型: fx, fy, cx, cy, k1, k2, p1, p2
    
    Returns:
        COLMAP 相机参数字典
    """
    # 如果有畸变系数，使用 OPENCV 模型
    if abs(intrinsics.k1) > 1e-6 or abs(intrinsics.k2) > 1e-6:
        model = "OPENCV"
        params = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                  intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2]
    else:
        model = "PINHOLE"
        params = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
    
    return {
        "model": model,
        "width": intrinsics.image_width,
        "height": intrinsics.image_height,
        "params": params,
    }


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="解析 RS30 .CP 文件中的相机内参")
    parser.add_argument("input", help="站点数据目录或 .CP 文件路径")
    parser.add_argument("--single", action="store_true", help="输入是单个 .CP 文件")
    parser.add_argument("--colmap", action="store_true", help="输出 COLMAP 格式")
    
    args = parser.parse_args()
    
    if args.single:
        intrinsics = parse_cp_file(args.input)
    else:
        all_intrinsics = parse_all_cameras(args.input)
        for name, intrinsics in all_intrinsics.items():
            print(f"\n=== {name} ===")
            if args.colmap:
                colmap_dict = intrinsics_to_colmap_dict(intrinsics)
                print(f"  Model: {colmap_dict['model']}")
                print(f"  Width: {colmap_dict['width']}, Height: {colmap_dict['height']}")
                print(f"  Params: {colmap_dict['params']}")
            else:
                print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
                print(f"  cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")
                print(f"  k1={intrinsics.k1:.6f}, k2={intrinsics.k2:.6f}")
                print(f"  p1={intrinsics.p1:.6f}, p2={intrinsics.p2:.6f}")
                print(f"  Width: {intrinsics.image_width}, Height: {intrinsics.image_height}")
