"""
3DGS 格式 PLY 文件加载器

解析 3D Gaussian Splatting 格式的 PLY 文件，提取高斯椭球体参数，
供 gsplat 渲染使用。支持计算场景边界框和地面平面检测。
"""

import gc
import os
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GaussianData:
    """3DGS 高斯数据容器"""
    # 基础位置 (N, 3)
    positions: np.ndarray
    # 缩放参数 (N, 3) - log scale
    scales: np.ndarray
    # 旋转四元数 (N, 4) - [w, x, y, z]
    rotations: np.ndarray
    # 球谐函数 DC 分量 (N, 3) - RGB 颜色
    f_dc: np.ndarray
    # 球谐函数高阶分量 (N, 45) - 颜色细节
    f_rest: np.ndarray
    # 不透明度 (N,) - logit
    opacity: np.ndarray

    # 计算属性（延迟加载）
    _bbox: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, init=False, repr=False)
    _ground_plane: Optional[Tuple[np.ndarray, float]] = field(default=None, init=False, repr=False)

    @property
    def count(self) -> int:
        """高斯数量"""
        return self.positions.shape[0]

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """场景边界框 (min_xyz, max_xyz)"""
        if self._bbox is None:
            self._bbox = (
                self.positions.min(axis=0),
                self.positions.max(axis=0),
            )
        return self._bbox

    @property
    def ground_plane(self) -> Tuple[np.ndarray, float]:
        """
        地面平面 (法向量, 距离)
        使用最低 5% 高斯点的位置拟合平面
        """
        if self._ground_plane is None:
            self._ground_plane = self._estimate_ground_plane()
        return self._ground_plane

    @property
    def rgb_colors(self) -> np.ndarray:
        """将 SH DC 分量转换为 RGB 颜色 [0, 1]"""
        # SH_C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        colors = 0.5 + self.f_dc / C0
        return np.clip(colors, 0.0, 1.0)

    @property
    def actual_scales(self) -> np.ndarray:
        """将 log scale 转换为实际缩放值"""
        return np.exp(self.scales)

    @property
    def actual_opacity(self) -> np.ndarray:
        """将 logit opacity 转换为 [0, 1]"""
        return 1.0 / (1.0 + np.exp(-self.opacity))

    def _estimate_ground_plane(self) -> Tuple[np.ndarray, float]:
        """使用最低位置的高斯点拟合地面平面"""
        z_coords = self.positions[:, 2]  # 假设 Z 轴向上
        threshold = np.percentile(z_coords, 5)
        ground_points = self.positions[z_coords <= threshold]

        if len(ground_points) < 3:
            # 点太少，简单返回 Z=最低值 的平面
            min_z = z_coords.min()
            return np.array([0.0, 0.0, 1.0]), -min_z

        # 降采样到 5000 点以加速 SVD（875K 点取最低 5% = 43757 点，SVD 慢）
        max_samples = 5000
        if len(ground_points) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(ground_points), max_samples, replace=False)
            ground_points = ground_points[indices]

        # 使用 SVD 拟合平面
        centroid = ground_points.mean(axis=0)
        centered = ground_points - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # 最小奇异值对应的向量

        # 确保法向量朝上
        if normal[2] < 0:
            normal = -normal

        distance = -np.dot(normal, centroid)
        return normal, distance

    def get_ground_height(self, x: float, y: float) -> float:
        """获取给定 (x, y) 位置的地面高度"""
        normal, distance = self.ground_plane
        if abs(normal[2]) < 1e-6:
            return 0.0
        return -(normal[0] * x + normal[1] * y + distance) / normal[2]


def load_3dgs_ply(ply_path: str, verbose: bool = True) -> GaussianData:
    """
    加载 3DGS 格式的 PLY 文件

    Args:
        ply_path: PLY 文件路径（支持 ~ 展开和相对路径）
        verbose: 是否打印加载信息

    Returns:
        GaussianData: 高斯数据容器

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不正确
    """
    ply_path = os.path.expanduser(ply_path)
    ply_path = os.path.abspath(ply_path)

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

    file_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
    if verbose:
        logger.info(f"加载 PLY 文件: {ply_path} ({file_size_mb:.1f} MB)")

    # 解析 PLY 头部
    header, data_offset, is_binary = _parse_ply_header(ply_path)

    if verbose:
        logger.info(f"顶点数: {header['vertex_count']}, 属性数: {len(header['properties'])}")
        logger.info(f"属性列表: {[p['name'] for p in header['properties']]}")

    # 读取数据
    if is_binary:
        vertex_data = _read_binary_ply(ply_path, data_offset, header)
    else:
        vertex_data = _read_ascii_ply(ply_path, data_offset, header)

    # 提取 3DGS 属性
    gaussian = _extract_gaussian_data(vertex_data, header)

    # 释放 vertex_data（大字典，含 875K 个高斯体的所有属性数组）
    # 对于 875K 高斯体，vertex_data 峰值内存 ~870MB
    del vertex_data
    gc.collect()

    if verbose:
        bbox_min, bbox_max = gaussian.bbox
        logger.info(f"场景边界: [{bbox_min}] ~ [{bbox_max}]")
        logger.info(f"地面高度范围: {bbox_min[2]:.2f} ~ {bbox_max[2]:.2f}")

    return gaussian


def _parse_ply_header(ply_path: str) -> Tuple[Dict[str, Any], int, bool]:
    """
    解析 PLY 文件头部

    Returns:
        (header_info, data_offset, is_binary)
    """
    header = {
        "format": "",
        "vertex_count": 0,
        "properties": [],
        "element_order": [],
    }

    with open(ply_path, "rb") as f:
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError(f"不是 PLY 文件: 首行 '{line}'")

        while True:
            line = f.readline().decode("ascii").strip()
            data_offset = f.tell()

            if line == "end_header":
                break

            tokens = line.split()

            if tokens[0] == "format":
                header["format"] = tokens[1]
            elif tokens[0] == "element":
                element_name = tokens[1]
                element_count = int(tokens[2])
                header["element_order"].append(element_name)
                if element_name == "vertex":
                    header["vertex_count"] = element_count
                    header["properties"] = []
            elif tokens[0] == "property":
                if len(tokens) == 3:
                    # 标量属性: property type name
                    header["properties"].append({
                        "name": tokens[2],
                        "type": tokens[1],
                    })
                elif len(tokens) == 5 and tokens[1] == "list":
                    # 列表属性: property list count_type value_type name
                    header["properties"].append({
                        "name": tokens[4],
                        "type": "list",
                        "count_type": tokens[2],
                        "value_type": tokens[3],
                    })

    is_binary = header["format"] in ("binary_little_endian", "binary_big_endian")
    return header, data_offset, is_binary


# PLY 数据类型到 struct 格式字符的映射
PLY_TYPE_MAP = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}

PLY_TYPE_SIZE = {
    "char": 1, "int8": 1,
    "uchar": 1, "uint8": 1,
    "short": 2, "int16": 2,
    "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4,
    "uint": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


def _read_binary_ply(ply_path: str, data_offset: int, header: Dict) -> Dict[str, np.ndarray]:
    """读取二进制格式 PLY 数据"""
    vertex_count = header["vertex_count"]
    properties = header["properties"]

    # 计算每行字节数
    row_size = sum(PLY_TYPE_SIZE.get(p["type"], 4) for p in properties if p["type"] != "list")

    # 构建格式字符串
    fmt_chars = []
    for p in properties:
        if p["type"] != "list":
            fmt_chars.append(PLY_TYPE_MAP.get(p["type"], "f"))

    # 检测字节序
    endian = "<" if header["format"] == "binary_little_endian" else ">"
    fmt_str = endian + "".join(fmt_chars)

    # 批量读取
    data = {}
    for p in properties:
        if p["type"] != "list":
            dtype = _ply_type_to_numpy(p["type"])
            data[p["name"]] = np.empty(vertex_count, dtype=dtype)

    with open(ply_path, "rb") as f:
        f.seek(data_offset)

        # 一次性读取全部数据，然后解包
        raw = f.read(row_size * vertex_count)

    # 使用 numpy 的 frombuffer 高效解析
    # 构建结构化 dtype
    np_dtypes = []
    for p in properties:
        if p["type"] != "list":
            np_dtypes.append((p["name"], _ply_type_to_numpy(p["type"])))

    structured = np.frombuffer(raw, dtype=np.dtype(np_dtypes))

    for p in properties:
        if p["type"] != "list":
            data[p["name"]] = structured[p["name"]].copy()

    return data


def _read_ascii_ply(ply_path: str, data_offset: int, header: Dict) -> Dict[str, np.ndarray]:
    """读取 ASCII 格式 PLY 数据"""
    vertex_count = header["vertex_count"]
    properties = header["properties"]
    prop_names = [p["name"] for p in properties if p["type"] != "list"]

    data = {name: [] for name in prop_names}

    with open(ply_path, "r") as f:
        f.seek(data_offset)

        for i in range(vertex_count):
            line = f.readline().strip()
            if not line:
                continue
            tokens = line.split()
            for j, name in enumerate(prop_names):
                if j < len(tokens):
                    data[name].append(float(tokens[j]))

    return {name: np.array(values) for name, values in data.items()}


def _ply_type_to_numpy(ply_type: str) -> np.dtype:
    """将 PLY 类型字符串转换为 numpy dtype"""
    type_map = {
        "char": np.int8, "int8": np.int8,
        "uchar": np.uint8, "uint8": np.uint8,
        "short": np.int16, "int16": np.int16,
        "ushort": np.uint16, "uint16": np.uint16,
        "int": np.int32, "int32": np.int32,
        "uint": np.uint32, "uint32": np.uint32,
        "float": np.float32, "float32": np.float32,
        "double": np.float64, "float64": np.float64,
    }
    return type_map.get(ply_type, np.float32)


def _extract_gaussian_data(vertex_data: Dict[str, np.ndarray], header: Dict) -> GaussianData:
    """从顶点数据中提取 3DGS 高斯参数"""

    def _get(name: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if name in vertex_data:
            return vertex_data[name].astype(np.float32)
        if default is not None:
            return default
        raise ValueError(f"缺少必要属性: {name}")

    # 位置
    positions = np.stack([_get("x"), _get("y"), _get("z")], axis=-1)

    # 缩放 (log scale)
    scales = np.stack([_get("scale_0"), _get("scale_1"), _get("scale_2")], axis=-1)

    # 旋转四元数 - PLY 中存储顺序为 [w, x, y, z]
    rotations = np.stack([
        _get("rot_0"),  # w
        _get("rot_1"),  # x
        _get("rot_2"),  # y
        _get("rot_3"),  # z
    ], axis=-1)

    # 归一化旋转四元数
    rot_norm = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rot_norm = np.where(rot_norm > 0, rot_norm, 1.0)
    rotations = rotations / rot_norm

    # SH DC 分量 (颜色)
    f_dc = np.stack([_get("f_dc_0"), _get("f_dc_1"), _get("f_dc_2")], axis=-1)

    # 不透明度 (logit)
    opacity = _get("opacity")

    # SH 高阶分量 (f_rest_0 ~ f_rest_44)
    f_rest_keys = [k for k in vertex_data.keys() if k.startswith("f_rest_")]
    f_rest_keys.sort(key=lambda k: int(k.split("_")[-1]))

    if f_rest_keys:
        f_rest = np.stack([_get(k) for k in f_rest_keys], axis=-1)
    else:
        # 没有 f_rest，用零填充
        f_rest = np.zeros((positions.shape[0], 45), dtype=np.float32)
        logger.warning("PLY 文件缺少 f_rest 属性，使用零填充")

    return GaussianData(
        positions=positions,
        scales=scales,
        rotations=rotations,
        f_dc=f_dc,
        f_rest=f_rest,
        opacity=opacity,
    )


if __name__ == "__main__":
    # 测试加载
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
    else:
        ply_path = "~/.openclaw/workspace-coder/AHOLO.ply"

    gaussian = load_3dgs_ply(ply_path)
    print(f"高斯数量: {gaussian.count}")
    print(f"边界框: {gaussian.bbox}")
    print(f"地面平面法向量: {gaussian.ground_plane[0]}")
    print(f"地面平面距离: {gaussian.ground_plane[1]:.2f}")
