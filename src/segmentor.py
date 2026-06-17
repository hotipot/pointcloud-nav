"""
3DGS 点云空间分割模块

使用 DBSCAN 聚类对 3DGS 高斯体进行实例级分割，
通过几何/颜色/密度特征筛选目标物体（如变电柜）。

主要功能：
- 大规模点云的 DBSCAN 聚类（支持空间降采样加速）
- 基于尺寸/形状/颜色/密度的物体筛选
- 候选物体排序与 JSON 导出
- 分割结果可视化
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """单个聚类簇的信息"""
    cluster_id: int
    indices: np.ndarray           # 属于该簇的高斯体索引
    center: np.ndarray            # 簇中心 (3,)
    bbox_min: np.ndarray          # 边界框最小角 (3,)
    bbox_max: np.ndarray          # 边界框最大角 (3,)
    size: np.ndarray              # 边界框尺寸 (3,) [宽, 深, 高]
    num_gaussians: int            # 高斯体数量
    dominant_color: np.ndarray    # 主色调 RGB [0,1] (3,)
    color_variance: float         # 颜色方差
    density: float                # 表面密度 (高斯体数/表面积)
    is_vertical: bool             # 是否竖直（高度>宽度且高度>深度）
    match_score: float = 0.0      # 与目标物体的匹配分数

    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "cluster_id": int(self.cluster_id),
            "num_gaussians": self.num_gaussians,
            "center": self.center.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "size": self.size.tolist(),
            "dominant_color": self.dominant_color.tolist(),
            "color_variance": float(self.color_variance),
            "density": float(self.density),
            "is_vertical": self.is_vertical,
            "match_score": float(self.match_score),
            "index_range": [int(self.indices[0]), int(self.indices[-1])],
            "total_indices": len(self.indices),
        }


@dataclass
class SegmentationResult:
    """分割结果"""
    clusters: List[ClusterInfo] = field(default_factory=list)
    candidates: List[ClusterInfo] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_clusters": len(self.clusters),
            "total_candidates": len(self.candidates),
            "clusters": [c.to_dict() for c in self.clusters],
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def save_json(self, path: str):
        """保存分割结果为 JSON"""
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"分割结果已保存: {path}")


def segment_scene(
    gaussian_data,
    config: Optional[dict] = None,
) -> SegmentationResult:
    """
    对 3DGS 场景进行空间聚类分割

    Args:
        gaussian_data: GaussianData 对象
        config: 分割配置（None 使用默认值）

    Returns:
        SegmentationResult: 分割结果
    """
    t0 = time.time()
    cfg = _merge_config(config)

    logger.info(f"开始分割: {gaussian_data.count:,} 个高斯体")
    logger.info(f"聚类参数: eps={cfg['dbscan_eps']}, min_samples={cfg['dbscan_min_samples']}")

    # 1. DBSCAN 聚类
    labels = _dbscan_cluster(
        gaussian_data.positions,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_min_samples"],
        sample_ratio=cfg.get("sample_ratio", 1.0),
    )

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(labels == -1))
    logger.info(f"聚类完成: {n_clusters} 个簇, {n_noise:,} 个噪声点")

    # 2. 构建簇信息
    clusters = _build_cluster_info(gaussian_data, labels)
    logger.info(f"构建簇信息完成: {len(clusters)} 个有效簇")

    # 3. 筛选候选物体
    candidates = _filter_candidates(clusters, cfg)
    logger.info(f"筛选完成: {len(candidates)} 个候选物体")

    result = SegmentationResult(
        clusters=clusters,
        candidates=candidates,
        params=cfg,
        elapsed_seconds=time.time() - t0,
    )

    # 打印候选结果
    if candidates:
        print(f"\n{'='*60}")
        print(f"  候选变电柜 ({len(candidates)} 个)")
        print(f"{'='*60}")
        for i, c in enumerate(candidates):
            print(f"  [{i+1}] 簇 #{c.cluster_id}")
            print(f"      尺寸: {c.size[0]:.2f}m × {c.size[1]:.2f}m × {c.size[2]:.2f}m (宽×深×高)")
            print(f"      中心: ({c.center[0]:.2f}, {c.center[1]:.2f}, {c.center[2]:.2f})")
            print(f"      高斯体数: {c.num_gaussians:,}")
            print(f"      主色调: RGB({c.dominant_color[0]:.2f}, {c.dominant_color[1]:.2f}, {c.dominant_color[2]:.2f})")
            print(f"      密度: {c.density:.1f} /m²")
            print(f"      匹配分数: {c.match_score:.3f}")
            print()
    else:
        print("\n⚠️ 未找到匹配的候选物体，尝试调整参数（减小 eps、放宽尺寸范围等）")

    return result


def _merge_config(user_config: Optional[dict]) -> dict:
    """合并用户配置与默认配置"""
    defaults = {
        # DBSCAN 参数
        "dbscan_eps": 0.15,              # 邻域半径（米）
        "dbscan_min_samples": 50,        # 最小点数
        "sample_ratio": 0.3,             # 降采样比例（875K点太多，先降采样聚类再传播标签）

        # 变电柜尺寸范围（米）
        "cabinet_width_range": [0.3, 1.5],   # 宽度
        "cabinet_depth_range": [0.2, 1.2],   # 深度
        "cabinet_height_range": [1.2, 2.8],  # 高度

        # 形状筛选
        "require_vertical": True,         # 要求竖直（高度>宽度且高度>深度）
        "height_width_ratio_min": 1.2,    # 高宽比最小值

        # 颜色筛选（灰色/绿色/黄色工业设备色）
        "color_filter_enabled": True,
        "color_ranges": [
            # 灰色系：低饱和度，中等亮度
            {"name": "gray", "sat_max": 0.15, "val_min": 0.2, "val_max": 0.8},
            # 绿色系：工业绿
            {"name": "green", "hue_min": 0.2, "hue_max": 0.45, "sat_min": 0.1, "val_min": 0.15},
            # 黄色系：警示黄
            {"name": "yellow", "hue_min": 0.08, "hue_max": 0.2, "sat_min": 0.2, "val_min": 0.3},
        ],

        # 密度筛选
        "min_density": 50.0,             # 最小表面密度（高斯体/m²）

        # 位置筛选
        "exclude_boundary_ratio": 0.1,   # 排除场景边界 10% 范围内的簇（墙壁/地面/天花板）
        "ground_offset_min": 0.3,        # 簇底面距地面最小高度（排除地面物体）
    }

    if user_config:
        # 递归合并
        for k, v in user_config.items():
            if isinstance(v, dict) and k in defaults and isinstance(defaults[k], dict):
                defaults[k].update(v)
            else:
                defaults[k] = v
    return defaults


def _dbscan_cluster(
    positions: np.ndarray,
    eps: float = 0.15,
    min_samples: int = 50,
    sample_ratio: float = 0.3,
) -> np.ndarray:
    """
    对高斯体位置进行 DBSCAN 聚类

    对大规模点云采用降采样策略：先对子集做 DBSCAN，
    再用最近邻将标签传播到全部点。

    Args:
        positions: (N, 3) 高斯体位置
        eps: DBSCAN 邻域半径
        min_samples: DBSCAN 最小样本数
        sample_ratio: 降采样比例

    Returns:
        labels: (N,) 每个点的簇标签，-1 为噪声
    """
    from sklearn.cluster import DBSCAN

    N = len(positions)
    logger.info(f"DBSCAN 聚类: {N:,} 个点, eps={eps}, min_samples={min_samples}")

    if sample_ratio < 1.0 and N > 100000:
        # 降采样聚类
        n_samples = max(int(N * sample_ratio), 10000)
        logger.info(f"降采样到 {n_samples:,} 个点进行聚类")

        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, n_samples, replace=False)
        sampled_positions = positions[sample_idx]

        # 对采样点做 DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
        sampled_labels = db.fit_predict(sampled_positions)

        n_found = len(set(sampled_labels)) - (1 if -1 in sampled_labels else 0)
        logger.info(f"采样点聚类: {n_found} 个簇")

        # 标签传播：用 KDTree 将标签分配给最近邻
        labels = _propagate_labels(positions, sampled_positions, sampled_labels)
    else:
        # 直接聚类
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
        labels = db.fit_predict(positions)

    return labels


def _propagate_labels(
    all_positions: np.ndarray,
    sampled_positions: np.ndarray,
    sampled_labels: np.ndarray,
) -> np.ndarray:
    """
    将采样点的聚类标签传播到全部点

    使用 KDTree 找最近邻采样点，继承其标签。
    距离超过阈值的点标记为噪声。

    Args:
        all_positions: (N, 3) 全部点位置
        sampled_positions: (M, 3) 采样点位置
        sampled_labels: (M,) 采样点标签

    Returns:
        labels: (N,) 全部点的标签
    """
    from sklearn.neighbors import KDTree

    logger.info("传播标签到全部点...")

    # 只对非噪声采样点建树
    valid_mask = sampled_labels != -1
    if not np.any(valid_mask):
        return np.full(len(all_positions), -1, dtype=np.int32)

    valid_positions = sampled_positions[valid_mask]
    valid_labels = sampled_labels[valid_mask]

    tree = KDTree(valid_positions)

    # 批量查询最近邻
    batch_size = 100000
    labels = np.full(len(all_positions), -1, dtype=np.int32)

    for start in range(0, len(all_positions), batch_size):
        end = min(start + batch_size, len(all_positions))
        dists, indices = tree.query(all_positions[start:end], k=1)
        dists = dists.ravel()
        indices = indices.ravel()

        # 距离阈值：2倍 eps（宽松一些，避免丢失边界点）
        max_dist = 0.3
        close_enough = dists < max_dist
        labels[start:end] = np.where(close_enough, valid_labels[indices], -1)

    return labels


def _build_cluster_info(gaussian_data, labels: np.ndarray) -> List[ClusterInfo]:
    """
    从聚类标签构建簇信息

    Args:
        gaussian_data: GaussianData 对象
        labels: (N,) 聚类标签

    Returns:
        簇信息列表
    """
    clusters = []
    unique_labels = set(labels)

    for label in sorted(unique_labels):
        if label == -1:
            continue  # 跳过噪声

        indices = np.where(labels == label)[0]
        if len(indices) < 10:
            continue  # 跳过极小簇

        positions = gaussian_data.positions[indices]
        colors = gaussian_data.rgb_colors[indices]

        # 边界框
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        size = bbox_max - bbox_min
        center = (bbox_min + bbox_max) / 2.0

        # 主色调（取中位数，比均值更鲁棒）
        dominant_color = np.median(colors, axis=0)

        # 颜色方差
        color_variance = float(np.mean(np.var(colors, axis=0)))

        # 表面密度 = 高斯体数 / 表面积
        surface_area = 2 * (size[0]*size[1] + size[0]*size[2] + size[1]*size[2])
        density = len(indices) / max(surface_area, 1e-6)

        # 是否竖直
        is_vertical = (size[2] > size[0]) and (size[2] > size[1])

        clusters.append(ClusterInfo(
            cluster_id=int(label),
            indices=indices,
            center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            size=size,
            num_gaussians=len(indices),
            dominant_color=dominant_color,
            color_variance=color_variance,
            density=density,
            is_vertical=is_vertical,
        ))

    # 按高斯体数量降序排列
    clusters.sort(key=lambda c: c.num_gaussians, reverse=True)
    return clusters


def _filter_candidates(clusters: List[ClusterInfo], cfg: dict) -> List[ClusterInfo]:
    """
    根据特征筛选候选变电柜

    筛选维度：尺寸、形状、颜色、密度、位置

    Args:
        clusters: 所有簇信息
        cfg: 筛选配置

    Returns:
        候选物体列表（按匹配分数降序）
    """
    candidates = []

    w_range = cfg["cabinet_width_range"]
    d_range = cfg["cabinet_depth_range"]
    h_range = cfg["cabinet_height_range"]

    for cluster in clusters:
        score = 0.0
        max_score = 0.0
        reasons = []

        # 1. 尺寸筛选（权重最高）
        w, d, h = cluster.size
        max_score += 3.0

        w_ok = w_range[0] <= w <= w_range[1]
        d_ok = d_range[0] <= d <= d_range[1]
        h_ok = h_range[0] <= h <= h_range[1]

        if w_ok and d_ok and h_ok:
            score += 3.0
            reasons.append("尺寸匹配")
        elif h_ok and (w_ok or d_ok):
            score += 1.5
            reasons.append("尺寸部分匹配")
        else:
            continue  # 尺寸完全不匹配，直接跳过

        # 2. 形状筛选：竖直长方体
        max_score += 2.0
        if cfg.get("require_vertical", True):
            if cluster.is_vertical:
                hw_ratio = h / max(w, 0.01)
                if hw_ratio >= cfg.get("height_width_ratio_min", 1.2):
                    score += 2.0
                    reasons.append(f"竖直形状(高宽比={hw_ratio:.1f})")
                else:
                    score += 0.5
            else:
                continue  # 不竖直，跳过
        else:
            score += 2.0

        # 3. 颜色筛选
        max_score += 2.0
        if cfg.get("color_filter_enabled", True):
            color_match = _check_color_match(cluster.dominant_color, cfg.get("color_ranges", []))
            if color_match:
                score += 2.0
                reasons.append(f"颜色匹配({color_match})")
            else:
                score += 0.3  # 颜色不匹配但不过滤
        else:
            score += 2.0

        # 4. 密度筛选
        max_score += 1.5
        min_density = cfg.get("min_density", 50.0)
        if cluster.density >= min_density:
            score += 1.5
            reasons.append(f"密度足够({cluster.density:.0f}/m²)")
        elif cluster.density >= min_density * 0.5:
            score += 0.5
        # 密度太低不过滤，只是降分

        # 5. 位置筛选：不在场景边界
        max_score += 1.5
        # (这个需要场景 bbox，暂时跳过，在 segment_scene 中补充)

        cluster.match_score = score / max_score if max_score > 0 else 0.0
        if cluster.match_score >= 0.4:  # 最低匹配阈值
            candidates.append(cluster)

    # 按匹配分数降序
    candidates.sort(key=lambda c: c.match_score, reverse=True)
    return candidates


def _check_color_match(color_rgb: np.ndarray, color_ranges: list) -> Optional[str]:
    """
    检查颜色是否匹配工业设备色系

    将 RGB 转换为 HSV 进行判断

    Args:
        color_rgb: RGB 颜色 [0, 1] (3,)
        color_ranges: 颜色范围配置列表

    Returns:
        匹配的颜色名称，或 None
    """
    # RGB → HSV
    r, g, b = color_rgb
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation
    s = 0.0 if cmax == 0 else delta / cmax

    # Hue
    if delta == 0:
        h = 0.0
    elif cmax == r:
        h = ((g - b) / delta) % 6
    elif cmax == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    h /= 6.0  # 归一化到 [0, 1]

    for cr in color_ranges:
        name = cr["name"]

        if name == "gray":
            # 灰色：低饱和度
            if s <= cr.get("sat_max", 0.15) and cr.get("val_min", 0.2) <= v <= cr.get("val_max", 0.8):
                return name

        elif name in ("green", "yellow"):
            # 有颜色：检查色相范围
            hue_min = cr.get("hue_min", 0)
            hue_max = cr.get("hue_max", 1)
            sat_min = cr.get("sat_min", 0)
            val_min = cr.get("val_min", 0)

            if hue_min <= h <= hue_max and s >= sat_min and v >= val_min:
                return name

    return None


def visualize_segmentation(
    gaussian_data,
    result: SegmentationResult,
    highlight_candidates: bool = True,
    point_size: float = 2.0,
):
    """
    可视化分割结果

    不同簇用不同颜色显示，候选变电柜高亮

    Args:
        gaussian_data: GaussianData 对象
        result: 分割结果
        highlight_candidates: 是否高亮候选物体
        point_size: 点大小
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("可视化需要 Open3D: pip install open3d")

    geometries = []

    # 1. 所有高斯体（灰色背景）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussian_data.positions)

    # 默认灰色
    bg_colors = np.full_like(gaussian_data.rgb_colors, 0.3)

    # 2. 为每个簇着色
    n_clusters = len(result.clusters)
    if n_clusters > 0:
        # 生成区分度高的颜色
        cluster_colors = _generate_cluster_colors(n_clusters)

        # 构建标签映射
        for i, cluster in enumerate(result.clusters):
            bg_colors[cluster.indices] = cluster_colors[i]

    # 3. 高亮候选变电柜
    if highlight_candidates:
        for candidate in result.candidates:
            bg_colors[candidate.indices] = [1.0, 0.0, 0.0]  # 红色高亮

    pcd.colors = o3d.utility.Vector3dVector(bg_colors)
    geometries.append(pcd)

    # 4. 候选物体的 bounding box
    for i, candidate in enumerate(result.candidates):
        bbox_lines = _create_bbox_lines_o3d(
            candidate.bbox_min, candidate.bbox_max,
            color=[1, 0, 0]  # 红色框
        )
        geometries.append(bbox_lines)

        # 标签球
        label_pos = candidate.bbox_max.copy()
        label_pos[2] += 0.2
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(label_pos)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    # 5. 场景边界框
    scene_bbox_lines = _create_bbox_lines_o3d(
        gaussian_data.bbox[0], gaussian_data.bbox[1],
        color=[0.5, 0.5, 0.5]
    )
    geometries.append(scene_bbox_lines)

    logger.info(f"可视化: {n_clusters} 个簇, {len(result.candidates)} 个候选高亮")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"分割结果 ({len(result.candidates)} 个候选变电柜)",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def _generate_cluster_colors(n: int) -> np.ndarray:
    """生成 n 个区分度高的 RGB 颜色"""
    colors = np.zeros((n, 3))
    for i in range(n):
        # 使用黄金角分布，确保颜色区分度
        hue = (i * 0.618033988749895) % 1.0
        # HSV → RGB (饱和度0.7, 亮度0.8)
        r, g, b = _hsv_to_rgb(hue, 0.7, 0.8)
        colors[i] = [r, g, b]
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """HSV 转 RGB"""
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


def _create_bbox_lines_o3d(bbox_min: np.ndarray, bbox_max: np.ndarray, color: list = None):
    """创建 Open3D 边界框线框"""
    import open3d as o3d

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    points = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color or [1, 1, 0]] * len(lines))

    return line_set


def get_cluster_indices(result: SegmentationResult, cluster_id: int) -> Optional[np.ndarray]:
    """
    获取指定簇的高斯体索引

    Args:
        result: 分割结果
        cluster_id: 簇 ID

    Returns:
        索引数组，或 None
    """
    for cluster in result.clusters:
        if cluster.cluster_id == cluster_id:
            return cluster.indices
    return None


def get_candidate_indices(result: SegmentationResult, candidate_idx: int = 0) -> Optional[np.ndarray]:
    """
    获取指定候选物体的高斯体索引

    Args:
        result: 分割结果
        candidate_idx: 候选序号（0=最佳匹配）

    Returns:
        索引数组，或 None
    """
    if 0 <= candidate_idx < len(result.candidates):
        return result.candidates[candidate_idx].indices
    return None
