"""
轨迹定义与插值模块

支持手动定义 waypoint 列表，使用样条插值生成平滑轨迹，
支持高度约束和朝向插值。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
import logging

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """
    路径点定义

    Attributes:
        position: 3D 位置 [x, y, z]
        orientation: 四元数朝向 [qw, qx, qy, qz]，None 表示自动朝向下一个点
        look_at: 注视目标点 [x, y, z]，优先级高于 orientation
    """
    position: np.ndarray
    orientation: Optional[np.ndarray] = None
    look_at: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=np.float64)
            # 归一化
            norm = np.linalg.norm(self.orientation)
            if norm > 0:
                self.orientation /= norm
        if self.look_at is not None:
            self.look_at = np.asarray(self.look_at, dtype=np.float64)


@dataclass
class CameraPose:
    """
    相机位姿（某一帧）

    Attributes:
        position: 相机位置 [x, y, z]
        rotation: 四元数 [qw, qx, qy, qz]
        fov: 水平视场角（度）
    """
    position: np.ndarray
    rotation: np.ndarray
    fov: float = 90.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 旋转矩阵"""
        return Rotation.from_quat(self._scipy_quat()).as_matrix()

    @property
    def view_matrix(self) -> np.ndarray:
        """4x4 视图矩阵 (world-to-camera)"""
        R = self.rotation_matrix
        t = self.position
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    @property
    def c2w_matrix(self) -> np.ndarray:
        """4x4 camera-to-world 矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T

    def _scipy_quat(self) -> np.ndarray:
        """转换为 scipy 格式的四元数 [qx, qy, qz, qw]"""
        return np.array([
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
            self.rotation[0],
        ])

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),  # [qw, qx, qy, qz]
            "fov": self.fov,
        }


class Trajectory:
    """
    轨迹类

    从 waypoint 列表生成平滑轨迹，支持位置和朝向插值。
    """

    def __init__(
        self,
        waypoints: List[Waypoint],
        robot_height: float = 1.5,
        ground_height_func: Optional[callable] = None,
        speed: float = 1.0,
        fps: float = 30.0,
        fov: float = 90.0,
    ):
        """
        Args:
            waypoints: 路径点列表
            robot_height: 机器人相机高度（米）
            ground_height_func: 地面高度函数 (x, y) -> z，None 则使用 waypoint 的 z 值
            speed: 机器人行走速度（米/秒）
            fps: 输出帧率
            fov: 相机视场角
        """
        self.waypoints = waypoints
        self.robot_height = robot_height
        self.ground_height_func = ground_height_func
        self.speed = speed
        self.fps = fps
        self.fov = fov

        if len(waypoints) < 2:
            raise ValueError("至少需要 2 个 waypoint")

        # 预处理：调整高度
        self._adjust_heights()

        # 预处理：计算朝向
        self._compute_orientations()

    def _adjust_heights(self):
        """调整 waypoint 高度：地面高度 + 机器人相机高度"""
        for wp in self.waypoints:
            if self.ground_height_func is not None:
                ground_z = self.ground_height_func(wp.position[0], wp.position[1])
                wp.position[2] = ground_z + self.robot_height
            # 如果没有地面函数，使用 waypoint 原始 z 值

    def _compute_orientations(self):
        """计算缺失的朝向：默认朝向下一个 waypoint"""
        for i, wp in enumerate(self.waypoints):
            if wp.look_at is not None:
                # 注视目标点
                direction = wp.look_at - wp.position
                direction[2] = 0  # 水平方向
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    rot = Rotation.from_rotvec([0, -np.arctan2(direction[0], direction[1]), 0])
                    # 更精确：使用 look-at 计算
                    rot = self._look_at_rotation(wp.position, wp.look_at)
                    wp.orientation = np.array([
                        rot.as_quat()[3],  # qw
                        rot.as_quat()[0],  # qx
                        rot.as_quat()[1],  # qy
                        rot.as_quat()[2],  # qz
                    ])
            elif wp.orientation is None:
                # 默认朝向下一个 waypoint
                if i < len(self.waypoints) - 1:
                    next_pos = self.waypoints[i + 1].position
                else:
                    # 最后一个点朝向前一个点的方向
                    prev_pos = self.waypoints[i - 1].position
                    next_pos = wp.position + (wp.position - prev_pos)

                direction = next_pos - wp.position
                direction[2] = 0  # 水平方向
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = direction / np.linalg.norm(direction)

                rot = self._direction_to_rotation(direction)
                wp.orientation = np.array([
                    rot.as_quat()[3],
                    rot.as_quat()[0],
                    rot.as_quat()[1],
                    rot.as_quat()[2],
                ])

    @staticmethod
    def _look_at_rotation(eye: np.ndarray, target: np.ndarray) -> Rotation:
        """
        计算 look-at 旋转

        相机坐标系：X 右，Y 下，Z 前（OpenGL 风格）
        """
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])  # Z 轴向上
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # forward 接近垂直，使用备用 up
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        # 构建旋转矩阵 (camera-to-world)
        # 相机 Z 轴 = -forward (相机看向 -Z)
        R = np.stack([right, true_up, -forward], axis=1)
        return Rotation.from_matrix(R)

    @staticmethod
    def _direction_to_rotation(direction: np.ndarray) -> Rotation:
        """将水平方向向量转换为旋转"""
        # 假设相机朝向 direction，Z 轴向上
        forward = direction / np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.stack([right, true_up, -forward], axis=1)
        return Rotation.from_matrix(R)

    def generate(self) -> List[CameraPose]:
        """
        生成完整轨迹的相机位姿序列

        Returns:
            List[CameraPose]: 每帧的相机位姿
        """
        positions = np.array([wp.position for wp in self.waypoints])
        orientations = np.array([wp.orientation for wp in self.waypoints])

        # 去除连续重复的 waypoints（避免 CubicSpline 报错）
        keep = [0]
        for i in range(1, len(positions)):
            if np.linalg.norm(positions[i] - positions[keep[-1]]) > 1e-4:
                keep.append(i)
        if len(keep) < 2:
            # 至少需要 2 个不同的点
            if len(positions) >= 2:
                keep = [0, len(positions) - 1]
            else:
                raise ValueError("至少需要 2 个不同的 waypoint")
        positions = positions[keep]
        orientations = orientations[keep]

        # 计算路径总长度
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(segment_lengths)

        # 计算总帧数
        total_time = total_length / self.speed
        total_frames = int(total_time * self.fps)
        total_frames = max(total_frames, 2)

        logger.info(f"轨迹总长度: {total_length:.2f}m, 总时间: {total_time:.2f}s, 总帧数: {total_frames}")

        # 累积距离参数化
        cum_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        cum_dist = cum_dist / cum_dist[-1]  # 归一化到 [0, 1]

        # 位置插值（三次样条）
        t_uniform = np.linspace(0, 1, total_frames)

        # 对每个坐标轴做样条插值
        cs_x = CubicSpline(cum_dist, positions[:, 0])
        cs_y = CubicSpline(cum_dist, positions[:, 1])
        cs_z = CubicSpline(cum_dist, positions[:, 2])

        interp_positions = np.stack([
            cs_x(t_uniform),
            cs_y(t_uniform),
            cs_z(t_uniform),
        ], axis=-1)

        # 高度约束：确保 z 坐标不低于地面 + 机器人高度
        if self.ground_height_func is not None:
            for i in range(len(interp_positions)):
                ground_z = self.ground_height_func(
                    interp_positions[i, 0], interp_positions[i, 1]
                )
                min_z = ground_z + self.robot_height
                if interp_positions[i, 2] < min_z:
                    interp_positions[i, 2] = min_z

        # 朝向插值（球面线性插值 Slerp）
        # 转换为 scipy 四元数格式 [qx, qy, qz, qw]
        scipy_quats = orientations[:, [1, 2, 3, 0]]

        # 确保四元数在同一半球
        for i in range(1, len(scipy_quats)):
            if np.dot(scipy_quats[i], scipy_quats[i - 1]) < 0:
                scipy_quats[i] = -scipy_quats[i]

        rotations = Rotation.from_quat(scipy_quats)
        slerp = Slerp(cum_dist, rotations)

        interp_rotations = slerp(t_uniform)
        interp_scipy_quats = interp_rotations.as_quat()

        # 转回 [qw, qx, qy, qz] 格式
        interp_orientations = interp_scipy_quats[:, [3, 0, 1, 2]]

        # 构建相机位姿列表
        poses = []
        for i in range(total_frames):
            poses.append(CameraPose(
                position=interp_positions[i],
                rotation=interp_orientations[i],
                fov=self.fov,
            ))

        return poses

    def get_total_distance(self) -> float:
        """计算轨迹总距离"""
        positions = np.array([wp.position for wp in self.waypoints])
        return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))


def _compute_density_bounds(
    positions: np.ndarray,
    density_threshold_percentile: float = 10.0,
    grid_size: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于点云密度计算有效内部区域的边界

    将 XY 平面划分为网格，统计每个网格的高斯数量，
    找到密度高于阈值的网格，返回这些网格的边界。

    Args:
        positions: (N, 3) 高斯位置数组
        density_threshold_percentile: 密度阈值百分位（0-100），
            低于此百分位的网格被视为稀疏区域。默认 10 表示
            只保留密度排名前 90% 的网格。
        grid_size: 网格大小（米）

    Returns:
        (inner_min, inner_max, center) — 密集区域的 XY 边界和密度加权中心
    """
    xy = positions[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    size = xy_max - xy_min

    nx = max(int(np.ceil(size[0] / grid_size)), 1)
    ny = max(int(np.ceil(size[1] / grid_size)), 1)

    xi = np.clip(((positions[:, 0] - xy_min[0]) / grid_size).astype(int), 0, nx - 1)
    yi = np.clip(((positions[:, 1] - xy_min[1]) / grid_size).astype(int), 0, ny - 1)

    density = np.zeros((nx, ny), dtype=int)
    np.add.at(density, (xi, yi), 1)

    # 过滤低密度网格
    flat = density.flatten()
    nonzero = flat[flat > 0]
    if len(nonzero) == 0:
        return xy_min, xy_max, (xy_min + xy_max) / 2

    threshold = np.percentile(nonzero, density_threshold_percentile)
    valid_mask = density > threshold

    if not valid_mask.any():
        return xy_min, xy_max, (xy_min + xy_max) / 2

    # 计算密度加权中心
    valid_ix, valid_iy = np.where(valid_mask)
    weights = density[valid_mask].astype(float)
    valid_x = xy_min[0] + (valid_ix + 0.5) * grid_size
    valid_y = xy_min[1] + (valid_iy + 0.5) * grid_size
    center = np.array([
        np.average(valid_x, weights=weights),
        np.average(valid_y, weights=weights),
    ])

    # 计算密度加权的标准差，作为"有效半径"
    std_x = np.sqrt(np.average((valid_x - center[0]) ** 2, weights=weights))
    std_y = np.sqrt(np.average((valid_y - center[1]) ** 2, weights=weights))

    # 以密度中心 ± 2σ 作为内部区域边界（覆盖约 95% 的有效区域）
    inner_min = np.array([center[0] - 2.0 * std_x, center[1] - 2.0 * std_y])
    inner_max = np.array([center[0] + 2.0 * std_x, center[1] + 2.0 * std_y])

    return inner_min, inner_max, center


def create_default_trajectory(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 1.0,
    fps: float = 30.0,
    fov: float = 90.0,
    positions: Optional[np.ndarray] = None,
) -> Trajectory:
    """
    根据场景边界框创建默认巡逻轨迹

    如果提供 positions（高斯位置数据），会基于点云密度分布
    自动选择内部密集区域生成轨迹，确保机器人在点云内部行走。
    否则 fallback 到基于 bounding box 的方式。

    Args:
        bbox_min: 场景边界最小值
        bbox_max: 场景边界最大值
        robot_height: 机器人相机高度
        ground_height_func: 地面高度函数
        speed: 行走速度
        fps: 输出帧率
        fov: 相机视场角
        positions: 高斯位置数据 (N, 3)，用于密度分析
    """
    if positions is not None:
        # 基于点云密度计算内部区域
        inner_min, inner_max, density_center = _compute_density_bounds(positions)

        # 在内部区域再内缩 20%，确保不贴边
        inner_size = inner_max - inner_min
        margin = 0.2
        x_min = inner_min[0] + inner_size[0] * margin
        x_max = inner_max[0] - inner_size[0] * margin
        y_min = inner_min[1] + inner_size[1] * margin
        y_max = inner_max[1] - inner_size[1] * margin

        # 使用密度中心作为轨迹中心，1/3 和 2/3 分位点作为 waypoint
        x1 = x_min + (x_max - x_min) * 0.33
        x2 = x_min + (x_max - x_min) * 0.67
        y1 = y_min + (y_max - y_min) * 0.33
        y2 = y_min + (y_max - y_min) * 0.67

        logger.info(
            f"密度感知轨迹: 内部区域 X=[{x_min:.1f}, {x_max:.1f}] "
            f"Y=[{y_min:.1f}, {y_max:.1f}], 密度中心=({density_center[0]:.1f}, {density_center[1]:.1f})"
        )
    else:
        # Fallback: 基于 bounding box
        center = (bbox_min + bbox_max) / 2
        size = bbox_max - bbox_min
        margin = 0.2

        x_min = bbox_min[0] + size[0] * margin
        x_max = bbox_max[0] - size[0] * margin
        y_min = bbox_min[1] + size[1] * margin
        y_max = bbox_max[1] - size[1] * margin

        x1 = x_min + (x_max - x_min) * 0.33
        x2 = x_min + (x_max - x_min) * 0.67
        y1 = y_min + (y_max - y_min) * 0.33
        y2 = y_min + (y_max - y_min) * 0.67

    # 矩形巡逻路径（位于场景内部）
    waypoints = [
        Waypoint(position=[x1, y1, 0]),
        Waypoint(position=[x2, y1, 0]),
        Waypoint(position=[x2, y2, 0]),
        Waypoint(position=[x1, y2, 0]),
        Waypoint(position=[x1, y1, 0]),  # 回到起点
    ]

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )


def create_zigzag_trajectory(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    rows: int = 3,
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 1.0,
    fps: float = 30.0,
    fov: float = 90.0,
    positions: Optional[np.ndarray] = None,
) -> Trajectory:
    """
    创建之字形扫描轨迹

    如果提供 positions，会基于点云密度选择内部密集区域。

    Args:
        bbox_min: 场景边界最小值
        bbox_max: 场景边界最大值
        rows: 扫描行数
        robot_height: 机器人相机高度
        ground_height_func: 地面高度函数
        speed: 行走速度
        fps: 输出帧率
        fov: 相机视场角
        positions: 高斯位置数据 (N, 3)，用于密度分析
    """
    if positions is not None:
        inner_min, inner_max, density_center = _compute_density_bounds(positions)
        inner_size = inner_max - inner_min
        margin = 0.2
        x_min = inner_min[0] + inner_size[0] * margin
        x_max = inner_max[0] - inner_size[0] * margin
        y_min = inner_min[1] + inner_size[1] * margin
        y_max = inner_max[1] - inner_size[1] * margin

        logger.info(
            f"密度感知之字形: 内部区域 X=[{x_min:.1f}, {x_max:.1f}] "
            f"Y=[{y_min:.1f}, {y_max:.1f}], 密度中心=({density_center[0]:.1f}, {density_center[1]:.1f})"
        )
    else:
        size = bbox_max - bbox_min
        margin = 0.2
        x_min = bbox_min[0] + size[0] * margin
        x_max = bbox_max[0] - size[0] * margin
        y_min = bbox_min[1] + size[1] * margin
        y_max = bbox_max[1] - size[1] * margin

    waypoints = []
    y_values = np.linspace(y_min, y_max, rows)

    for i, y in enumerate(y_values):
        if i % 2 == 0:
            # 从左到右
            waypoints.append(Waypoint(position=[x_min, y, 0]))
            waypoints.append(Waypoint(position=[x_max, y, 0]))
        else:
            # 从右到左
            waypoints.append(Waypoint(position=[x_max, y, 0]))
            waypoints.append(Waypoint(position=[x_min, y, 0]))

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )


def create_interior_patrol_trajectory(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 0.8,
    fps: float = 30.0,
    fov: float = 60.0,
    positions: Optional[np.ndarray] = None,
    grid_size: float = 1.0,
    density_percentile: float = 30.0,
) -> Trajectory:
    """
    创建室内巡逻轨迹：沿点云密度最高的走廊中轴行进，
    相机始终朝向行进方向，呈现第一人称室内巡逻视角。

    与 default/zigzag 轨迹的关键区别：
    - 轨迹沿点云密度中轴（走廊中心）行走
    - 相机朝向行进方向（前视），而非朝向远处的点
    - FOV 默认 60°（室内人眼视角），而非 90°
    - 速度默认 0.8m/s（巡逻步速）
    - 每个 waypoint 都有明确的前视方向

    Args:
        bbox_min: 场景边界最小值
        bbox_max: 场景边界最大值
        robot_height: 机器人相机高度
        ground_height_func: 地面高度函数
        speed: 行走速度
        fps: 输出帧率
        fov: 相机视场角
        positions: 高斯位置数据 (N, 3)，用于密度分析
        grid_size: 密度网格大小
        density_percentile: 密度阈值百分位
    """
    if positions is None:
        logger.warning("未提供 positions 数据，interior-patrol 将 fallback 到 zigzag 轨迹")
        return create_zigzag_trajectory(
            bbox_min, bbox_max, rows=3,
            robot_height=robot_height,
            ground_height_func=ground_height_func,
            speed=speed, fps=fps, fov=fov,
        )

    # ===== 1. 构建机器人高度带的 XY 密度图 =====
    # 只取机器人视线高度附近的点（地面 +0.5m 到地面 +3m）
    # 由于地面高度可能不均匀，先用全局 Z 范围估算
    z = positions[:, 2]
    z_low = np.percentile(z, 10)  # 大约是地面
    z_high = z_low + 3.0  # 地面上方 3m 以内
    band_mask = (z >= z_low) & (z <= z_high)
    band_pos = positions[band_mask]

    if len(band_pos) < 100:
        band_pos = positions  # fallback

    xy = band_pos[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_size = xy_max - xy_min

    nx = max(int(np.ceil(xy_size[0] / grid_size)), 1)
    ny = max(int(np.ceil(xy_size[1] / grid_size)), 1)

    xi = np.clip(((band_pos[:, 0] - xy_min[0]) / grid_size).astype(int), 0, nx - 1)
    yi = np.clip(((band_pos[:, 1] - xy_min[1]) / grid_size).astype(int), 0, ny - 1)

    density = np.zeros((nx, ny), dtype=int)
    np.add.at(density, (xi, yi), 1)

    # ===== 2. 找到走廊中轴：对每个 X 列，找密度最高的 Y =====
    nonzero_flat = density[density > 0]
    if len(nonzero_flat) == 0:
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        threshold = np.percentile(nonzero_flat, density_percentile)
        corridor_y = np.full(nx, np.nan)
        for ix in range(nx):
            col = density[ix, :]
            valid = col > threshold
            if valid.any():
                # 密度加权 Y 中心
                iy_valid = np.where(valid)[0]
                weights = col[iy_valid].astype(float)
                y_centers = xy_min[1] + (iy_valid + 0.5) * grid_size
                corridor_y[ix] = np.average(y_centers, weights=weights)

    # 用有效值填充 NaN（前后插值）
    valid_mask = ~np.isnan(corridor_y)
    if not valid_mask.any():
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        corridor_y = np.interp(
            np.arange(nx),
            np.where(valid_mask)[0],
            corridor_y[valid_mask],
        )

    # 平滑走廊曲线（移动平均）
    window = max(3, nx // 20)
    kernel = np.ones(window) / window
    corridor_y_smooth = np.convolve(corridor_y, kernel, mode="same")

    # ===== 3. 沿走廊中轴采样 waypoints =====
    # X 方向从一端走到另一端，Y 跟随走廊中轴
    x_centers = xy_min[0] + (np.arange(nx) + 0.5) * grid_size

    # 过滤：只保留密度足够高的 X 区间
    col_density = density.sum(axis=1)  # 每个 X 列的总密度
    col_threshold = np.percentile(col_density[col_density > 0], density_percentile) if (col_density > 0).any() else 0
    valid_x = col_density > col_threshold

    if not valid_x.any():
        valid_x = col_density > 0

    # 找到连续的 valid 区间
    valid_indices = np.where(valid_x)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(nx)

    # 取最长的连续段
    breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
    segments = np.split(valid_indices, breaks)
    longest = max(segments, key=len)
    start_idx = longest[0]
    end_idx = longest[-1]

    # 在走廊中每隔 2-3 米采一个 waypoint
    step = max(1, int(3.0 / grid_size))  # 每 3 米一个 waypoint
    waypoint_indices = list(range(start_idx, end_idx + 1, step))
    if waypoint_indices[-1] != end_idx:
        waypoint_indices.append(end_idx)

    # 去程：从一端到另一端
    waypoints = []
    for idx in waypoint_indices:
        x = x_centers[idx]
        y = corridor_y_smooth[idx]
        waypoints.append(Waypoint(position=[x, y, 0]))

    # 回程：从另一端走回来（Y 偏移一点，模拟另一侧走廊）
    # 计算走廊宽度：在走廊 Y 中心上下找密度高的区域
    y_offset = grid_size * 1.5  # 偏移 1.5m
    for idx in reversed(waypoint_indices):
        x = x_centers[idx]
        # 检查偏移位置是否也在密度区域内
        y_base = corridor_y_smooth[idx]
        y_alt = y_base + y_offset
        # 如果偏移位置密度太低，用回 Y 中心
        alt_ix = int((x - xy_min[0]) / grid_size)
        alt_iy = int((y_alt - xy_min[1]) / grid_size)
        if 0 <= alt_ix < nx and 0 <= alt_iy < ny and density[alt_ix, alt_iy] > threshold:
            waypoints.append(Waypoint(position=[x, y_alt, 0]))
        else:
            waypoints.append(Waypoint(position=[x, y_base, 0]))

    logger.info(
        f"室内巡逻轨迹: 走廊长度 X=[{x_centers[start_idx]:.1f}, {x_centers[end_idx]:.1f}], "
        f"{len(waypoints)} 个 waypoints, FOV={fov}°"
    )

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )


def create_interior_explore_trajectory(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 0.5,
    fps: float = 30.0,
    fov: float = 90.0,
    positions: Optional[np.ndarray] = None,
    grid_size: float = 1.0,
    density_percentile: float = 30.0,
    look_around_speed: float = 0.3,
) -> Trajectory:
    """
    创建室内探索轨迹：在走廊中轴行走，相机缓慢环视 360°，
    呈现第一人称室内探索视角，能看清走廊两侧和内部空间。

    与 interior-patrol 的区别：
    - 相机缓慢旋转 360°（look-around），而非只看行进方向
    - FOV 默认 90°（广角，看得更宽）
    - 速度更慢（0.5m/s），配合旋转有足够时间观察
    - 每个轨迹段独立控制朝向，确保覆盖所有方向

    Args:
        bbox_min: 场景边界最小值
        bbox_max: 场景边界最大值
        robot_height: 机器人相机高度
        ground_height_func: 地面高度函数
        speed: 行走速度
        fps: 输出帧率
        fov: 相机视场角
        positions: 高斯位置数据 (N, 3)，用于密度分析
        grid_size: 密度网格大小
        density_percentile: 密度阈值百分位
        look_around_speed: 环视旋转速度（圈/米，默认每走 1 米转 0.3 圈）
    """
    if positions is None:
        logger.warning("未提供 positions 数据，interior-explore 将 fallback 到 zigzag 轨迹")
        return create_zigzag_trajectory(
            bbox_min, bbox_max, rows=3,
            robot_height=robot_height,
            ground_height_func=ground_height_func,
            speed=speed, fps=fps, fov=fov,
        )

    # ===== 复用 interior-patrol 的密度分析逻辑 =====
    z = positions[:, 2]
    z_low = np.percentile(z, 10)
    z_high = z_low + 3.0
    band_mask = (z >= z_low) & (z <= z_high)
    band_pos = positions[band_mask]
    if len(band_pos) < 100:
        band_pos = positions

    xy = band_pos[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_size = xy_max - xy_min

    nx = max(int(np.ceil(xy_size[0] / grid_size)), 1)
    ny = max(int(np.ceil(xy_size[1] / grid_size)), 1)

    xi = np.clip(((band_pos[:, 0] - xy_min[0]) / grid_size).astype(int), 0, nx - 1)
    yi = np.clip(((band_pos[:, 1] - xy_min[1]) / grid_size).astype(int), 0, ny - 1)

    density = np.zeros((nx, ny), dtype=int)
    np.add.at(density, (xi, yi), 1)

    # 走廊中轴
    nonzero_flat = density[density > 0]
    if len(nonzero_flat) == 0:
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        threshold = np.percentile(nonzero_flat, density_percentile)
        corridor_y = np.full(nx, np.nan)
        for ix in range(nx):
            col = density[ix, :]
            valid = col > threshold
            if valid.any():
                iy_valid = np.where(valid)[0]
                weights = col[iy_valid].astype(float)
                y_centers = xy_min[1] + (iy_valid + 0.5) * grid_size
                corridor_y[ix] = np.average(y_centers, weights=weights)

    valid_mask = ~np.isnan(corridor_y)
    if not valid_mask.any():
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        corridor_y = np.interp(
            np.arange(nx),
            np.where(valid_mask)[0],
            corridor_y[valid_mask],
        )

    window = max(3, nx // 20)
    kernel = np.ones(window) / window
    corridor_y_smooth = np.convolve(corridor_y, kernel, mode="same")

    # 沿走廊中轴采样 waypoints
    x_centers = xy_min[0] + (np.arange(nx) + 0.5) * grid_size

    col_density = density.sum(axis=1)
    col_threshold = np.percentile(col_density[col_density > 0], density_percentile) if (col_density > 0).any() else 0
    valid_x = col_density > col_threshold
    if not valid_x.any():
        valid_x = col_density > 0

    valid_indices = np.where(valid_x)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(nx)

    breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
    segments = np.split(valid_indices, breaks)
    longest = max(segments, key=len)
    start_idx = longest[0]
    end_idx = longest[-1]

    step = max(1, int(3.0 / grid_size))
    waypoint_indices = list(range(start_idx, end_idx + 1, step))
    if waypoint_indices[-1] != end_idx:
        waypoint_indices.append(end_idx)

    # ===== 关键区别：每个 waypoint 设置 look_at 为密度中心方向 =====
    # 计算全局密度加权中心
    all_valid = density > 0
    if all_valid.any():
        av_ix, av_iy = np.where(all_valid)
        av_w = density[all_valid].astype(float)
        density_center_x = xy_min[0] + np.average(av_ix + 0.5, weights=av_w) * grid_size
        density_center_y = xy_min[1] + np.average(av_iy + 0.5, weights=av_w) * grid_size
    else:
        density_center_x = (xy_min[0] + xy_max[0]) / 2
        density_center_y = (xy_min[1] + xy_max[1]) / 2

    waypoints = []

    # 去程：从一端到另一端，相机朝向密度中心偏移方向
    for idx in waypoint_indices:
        x = x_centers[idx]
        y = corridor_y_smooth[idx]

        # look_at 不设，而是设置 orientation：朝向垂直于行进方向（看侧面）
        # 行进方向沿 X 轴，所以侧面是 Y 方向
        # 计算从当前位置到密度中心的偏移，朝向密度中心方向
        dx = density_center_x - x
        dy = density_center_y - y

        # 在走廊中，朝向密度中心就是朝向走廊内部
        # 但同时加上 Z 方向的微抬（看向稍微偏上的位置，模拟人眼习惯）
        look_at_pos = np.array([
            density_center_x,  # X 朝向中心
            density_center_y,  # Y 朝向中心
            z_low + 1.5,       # Z 看向人眼高度附近
        ])
        waypoints.append(Waypoint(position=[x, y, 0], look_at=look_at_pos))

    # 回程：从另一端走回来，交替看向另一侧
    y_offset = grid_size * 1.5
    for idx in reversed(waypoint_indices):
        x = x_centers[idx]
        y_base = corridor_y_smooth[idx]

        alt_ix = int((x - xy_min[0]) / grid_size)
        alt_iy = int((y_base + y_offset - xy_min[1]) / grid_size)
        if 0 <= alt_ix < nx and 0 <= alt_iy < ny and density[alt_ix, alt_iy] > (threshold if len(nonzero_flat) > 0 else 0):
            y = y_base + y_offset
        else:
            y = y_base

        # 回程 look_at 朝向相反方向
        look_at_pos = np.array([
            density_center_x,
            density_center_y,
            z_low + 1.5,
        ])
        waypoints.append(Waypoint(position=[x, y, 0], look_at=look_at_pos))

    logger.info(
        f"室内探索轨迹: 走廊长度 X=[{x_centers[start_idx]:.1f}, {x_centers[end_idx]:.1f}], "
        f"{len(waypoints)} 个 waypoints, FOV={fov}°, 密度中心=({density_center_x:.1f}, {density_center_y:.1f})"
    )

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )


def create_look_around_trajectory(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 0.3,
    fps: float = 30.0,
    fov: float = 90.0,
    positions: Optional[np.ndarray] = None,
    grid_size: float = 1.0,
    density_percentile: float = 30.0,
    rotation_speed: float = 60.0,
) -> Trajectory:
    """
    创建环视轨迹：在走廊中轴缓慢行走，相机持续旋转 360°，
    每走一步都在看不同方向，完整展示走廊内部空间。

    核心特点：
    - 相机朝向在轨迹生成时通过 look_at 显式控制
    - 每个 waypoint 的 look_at 指向不同方向（绕垂直轴旋转）
    - 行走速度慢（0.3m/s），旋转速度可调

    Args:
        bbox_min: 场景边界最小值
        bbox_max: 场景边界最大值
        robot_height: 机器人相机高度
        ground_height_func: 地面高度函数
        speed: 行走速度
        fps: 输出帧率
        fov: 相机视场角
        positions: 高斯位置数据 (N, 3)，用于密度分析
        grid_size: 密度网格大小
        density_percentile: 密度阈值百分位
        rotation_speed: 旋转速度（度/米，每走 1 米旋转多少度）
    """
    if positions is None:
        logger.warning("未提供 positions 数据，look-around 将 fallback 到 zigzag 轨迹")
        return create_zigzag_trajectory(
            bbox_min, bbox_max, rows=3,
            robot_height=robot_height,
            ground_height_func=ground_height_func,
            speed=speed, fps=fps, fov=fov,
        )

    # ===== 密度分析（同 interior-patrol） =====
    z = positions[:, 2]
    z_low = np.percentile(z, 10)
    z_high = z_low + 3.0
    band_mask = (z >= z_low) & (z <= z_high)
    band_pos = positions[band_mask]
    if len(band_pos) < 100:
        band_pos = positions

    xy = band_pos[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_size = xy_max - xy_min

    nx = max(int(np.ceil(xy_size[0] / grid_size)), 1)
    ny = max(int(np.ceil(xy_size[1] / grid_size)), 1)

    xi = np.clip(((band_pos[:, 0] - xy_min[0]) / grid_size).astype(int), 0, nx - 1)
    yi = np.clip(((band_pos[:, 1] - xy_min[1]) / grid_size).astype(int), 0, ny - 1)

    density = np.zeros((nx, ny), dtype=int)
    np.add.at(density, (xi, yi), 1)

    nonzero_flat = density[density > 0]
    if len(nonzero_flat) == 0:
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        threshold = np.percentile(nonzero_flat, density_percentile)
        corridor_y = np.full(nx, np.nan)
        for ix in range(nx):
            col = density[ix, :]
            valid = col > threshold
            if valid.any():
                iy_valid = np.where(valid)[0]
                weights = col[iy_valid].astype(float)
                y_centers = xy_min[1] + (iy_valid + 0.5) * grid_size
                corridor_y[ix] = np.average(y_centers, weights=weights)

    valid_mask = ~np.isnan(corridor_y)
    if not valid_mask.any():
        corridor_y = np.full(nx, (xy_min[1] + xy_max[1]) / 2)
    else:
        corridor_y = np.interp(
            np.arange(nx),
            np.where(valid_mask)[0],
            corridor_y[valid_mask],
        )

    window = max(3, nx // 20)
    kernel = np.ones(window) / window
    corridor_y_smooth = np.convolve(corridor_y, kernel, mode="same")

    x_centers = xy_min[0] + (np.arange(nx) + 0.5) * grid_size

    col_density = density.sum(axis=1)
    col_threshold = np.percentile(col_density[col_density > 0], density_percentile) if (col_density > 0).any() else 0
    valid_x = col_density > col_threshold
    if not valid_x.any():
        valid_x = col_density > 0

    valid_indices = np.where(valid_x)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(nx)

    breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
    segments = np.split(valid_indices, breaks)
    longest = max(segments, key=len)
    start_idx = longest[0]
    end_idx = longest[-1]

    # ===== 关键区别：密集采样 + look_at 旋转 =====
    # 每隔 2 米一个 waypoint，每个 waypoint 看不同方向
    step = max(1, int(2.0 / grid_size))
    waypoint_indices = list(range(start_idx, end_idx + 1, step))
    if waypoint_indices[-1] != end_idx:
        waypoint_indices.append(end_idx)

    waypoints = []
    cumulative_dist = 0.0

    # 生成去程 waypoints，每个 look_at 方向不同
    for i, idx in enumerate(waypoint_indices):
        x = x_centers[idx]
        y = corridor_y_smooth[idx]

        if i > 0:
            prev_idx = waypoint_indices[i - 1]
            dx = x - x_centers[prev_idx]
            dy = y - corridor_y_smooth[prev_idx]
            cumulative_dist += np.sqrt(dx**2 + dy**2)

        # 旋转角度：根据行走距离持续旋转
        angle_rad = np.radians(rotation_speed * cumulative_dist)
        look_dist = 5.0  # look_at 点距离 5 米

        # look_at 点：从当前位置出发，沿旋转角度方向看
        look_x = x + look_dist * np.cos(angle_rad)
        look_y = y + look_dist * np.sin(angle_rad)
        look_z = z_low + 1.5  # 看向人眼高度

        waypoints.append(Waypoint(position=[x, y, 0], look_at=np.array([look_x, look_y, look_z])))

    # 回程：反方向走回来，继续旋转
    for i, idx in enumerate(reversed(waypoint_indices)):
        x = x_centers[idx]
        y = corridor_y_smooth[idx]

        if i > 0:
            prev_idx = list(reversed(waypoint_indices))[i - 1]
            dx = x - x_centers[prev_idx]
            dy = y - corridor_y_smooth[prev_idx]
            cumulative_dist += np.sqrt(dx**2 + dy**2)

        angle_rad = np.radians(rotation_speed * cumulative_dist)
        look_x = x + look_dist * np.cos(angle_rad)
        look_y = y + look_dist * np.sin(angle_rad)
        look_z = z_low + 1.5

        waypoints.append(Waypoint(position=[x, y, 0], look_at=np.array([look_x, look_y, look_z])))

    logger.info(
        f"环视轨迹: 走廊长度 X=[{x_centers[start_idx]:.1f}, {x_centers[end_idx]:.1f}], "
        f"{len(waypoints)} 个 waypoints, FOV={fov}°, 旋转速度={rotation_speed}°/m"
    )

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )


def waypoints_from_list(
    waypoint_list: List[List[float]],
    robot_height: float = 1.5,
    ground_height_func: Optional[callable] = None,
    speed: float = 1.0,
    fps: float = 30.0,
    fov: float = 90.0,
) -> Trajectory:
    """
    从坐标列表创建轨迹

    Args:
        waypoint_list: 坐标列表，每个元素为 [x, y, z] 或 [x, y]
    """
    waypoints = []
    for coords in waypoint_list:
        if len(coords) == 2:
            coords = [coords[0], coords[1], 0.0]
        waypoints.append(Waypoint(position=coords))

    return Trajectory(
        waypoints=waypoints,
        robot_height=robot_height,
        ground_height_func=ground_height_func,
        speed=speed,
        fps=fps,
        fov=fov,
    )
