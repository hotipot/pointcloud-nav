"""
3D 可视化工具模块

提供交互式 3D 可视化功能：
- 点云 + 轨迹线叠加显示
- 相机视锥体显示
- 场景边界框显示
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def visualize_scene(
    gaussian_data,
    camera_poses: Optional[list] = None,
    show_trajectory: bool = True,
    show_frustums: bool = True,
    show_bbox: bool = True,
    point_size: float = 2.0,
    frustum_interval: int = 30,
):
    """
    交互式 3D 可视化场景

    Args:
        gaussian_data: GaussianData 对象
        camera_poses: CameraPose 列表（可选）
        show_trajectory: 是否显示轨迹线
        show_frustums: 是否显示相机视锥体
        show_bbox: 是否显示场景边界框
        point_size: 点云渲染大小
        frustum_interval: 视锥体显示间隔（每隔多少帧显示一个）
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("可视化需要 Open3D: pip install open3d")

    geometries = []

    # 1. 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussian_data.positions)
    pcd.colors = o3d.utility.Vector3dVector(gaussian_data.rgb_colors)
    geometries.append(pcd)

    # 2. 场景边界框
    if show_bbox:
        bbox_min, bbox_max = gaussian_data.bbox
        bbox_lines = _create_bbox_lines(bbox_min, bbox_max)
        geometries.append(bbox_lines)

    # 3. 轨迹线
    if show_trajectory and camera_poses is not None and len(camera_poses) > 1:
        traj_line = _create_trajectory_line(camera_poses)
        geometries.append(traj_line)

        # 轨迹起点/终点标记
        start_sphere = _create_sphere(camera_poses[0].position, radius=0.1, color=[0, 1, 0])
        end_sphere = _create_sphere(camera_poses[-1].position, radius=0.1, color=[1, 0, 0])
        geometries.append(start_sphere)
        geometries.append(end_sphere)

    # 4. 相机视锥体
    if show_frustums and camera_poses is not None:
        for i in range(0, len(camera_poses), frustum_interval):
            frustum = _create_frustum(camera_poses[i], scale=0.3)
            geometries.append(frustum)

    # 显示
    logger.info(f"显示场景: {gaussian_data.count} 个高斯点"
                + (f", {len(camera_poses)} 帧轨迹" if camera_poses else ""))

    o3d.visualization.draw_geometries(
        geometries,
        window_name="点云导航可视化",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def _create_bbox_lines(bbox_min: np.ndarray, bbox_max: np.ndarray):
    """创建边界框线框"""
    import open3d as o3d

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    points = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖线
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))

    return line_set


def _create_trajectory_line(camera_poses: list):
    """创建轨迹线"""
    import open3d as o3d

    points = [pose.position for pose in camera_poses]
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # 渐变色：绿 → 黄 → 红
    colors = []
    n = len(lines)
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append([t, 1 - t, 0])  # 绿到红
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def _create_sphere(center: np.ndarray, radius: float = 0.1, color: list = None):
    """创建球体标记"""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color or [1, 1, 0])
    sphere.compute_vertex_normals()
    return sphere


def _create_frustum(camera_pose, scale: float = 0.3):
    """创建相机视锥体"""
    import open3d as o3d

    R = camera_pose.rotation_matrix
    pos = camera_pose.position

    # 视锥体顶点（相机坐标系）
    fov_rad = np.radians(camera_pose.fov)
    aspect = 4.0 / 3.0  # 默认宽高比
    h = scale * np.tan(fov_rad / 2)
    w = h * aspect

    # 相机坐标系下的 5 个点：原点 + 近平面 4 角
    local_points = np.array([
        [0, 0, 0],        # 相机位置
        [-w, -h, scale],  # 左下
        [w, -h, scale],   # 右下
        [w, h, scale],    # 右上
        [-w, h, scale],   # 左上
    ])

    # 变换到世界坐标
    world_points = (R @ local_points.T).T + pos

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 4 条射线
        [1, 2], [2, 3], [3, 4], [4, 1],  # 近平面 4 边
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0.5, 1]] * len(lines))

    return line_set


def visualize_trajectory_only(
    camera_poses: list,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
):
    """
    仅可视化轨迹（不加载点云，速度快）

    Args:
        camera_poses: CameraPose 列表
        bbox_min: 场景边界（可选）
        bbox_max: 场景边界（可选）
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("可视化需要 Open3D: pip install open3d")

    geometries = []

    # 轨迹线
    if len(camera_poses) > 1:
        traj_line = _create_trajectory_line(camera_poses)
        geometries.append(traj_line)

    # 起终点标记
    start_sphere = _create_sphere(camera_poses[0].position, radius=0.1, color=[0, 1, 0])
    end_sphere = _create_sphere(camera_poses[-1].position, radius=0.1, color=[1, 0, 0])
    geometries.append(start_sphere)
    geometries.append(end_sphere)

    # 视锥体
    for i in range(0, len(camera_poses), max(1, len(camera_poses) // 20)):
        frustum = _create_frustum(camera_poses[i], scale=0.3)
        geometries.append(frustum)

    # 边界框
    if bbox_min is not None and bbox_max is not None:
        bbox_lines = _create_bbox_lines(bbox_min, bbox_max)
        geometries.append(bbox_lines)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="轨迹可视化",
        width=1280,
        height=720,
    )


def render_scene_screenshot(
    gaussian_data=None,
    camera_poses: Optional[list] = None,
    output_path: str = "output/scene_screenshot.png",
    width: int = 1280,
    height: int = 720,
    show_trajectory: bool = True,
    show_frustums: bool = True,
    show_bbox: bool = True,
    point_size: float = 2.0,
    frustum_interval: int = 30,
    trajectory_only: bool = False,
):
    """
    使用 Open3D Visualizer (visible=False) 渲染场景截图

    不会弹出窗口，适用于无头环境或后台运行。
    渲染后保存为 PNG 文件。

    Args:
        gaussian_data: GaussianData 对象（trajectory_only=True 时可为 None）
        camera_poses: CameraPose 列表
        output_path: 输出图片路径
        width: 图像宽度
        height: 图像高度
        show_trajectory: 是否显示轨迹线
        show_frustums: 是否显示相机视锥体
        show_bbox: 是否显示场景边界框
        point_size: 点云渲染大小
        frustum_interval: 视锥体显示间隔
        trajectory_only: 仅显示轨迹（不加载点云）

    Returns:
        保存的截图路径
    """
    import os
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("可视化需要 Open3D: pip install open3d")

    geometries = []

    # 点云
    if not trajectory_only and gaussian_data is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gaussian_data.positions)
        pcd.colors = o3d.utility.Vector3dVector(gaussian_data.rgb_colors)
        geometries.append(pcd)

    # 场景边界框
    if show_bbox:
        if gaussian_data is not None:
            bbox_min, bbox_max = gaussian_data.bbox
        elif camera_poses is not None:
            # 从轨迹推断边界
            positions = np.array([p.position for p in camera_poses])
            bbox_min = positions.min(axis=0) - 1.0
            bbox_max = positions.max(axis=0) + 1.0
        else:
            bbox_min = np.array([-5, -5, -5])
            bbox_max = np.array([5, 5, 5])
        bbox_lines = _create_bbox_lines(bbox_min, bbox_max)
        geometries.append(bbox_lines)

    # 轨迹线
    if show_trajectory and camera_poses is not None and len(camera_poses) > 1:
        traj_line = _create_trajectory_line(camera_poses)
        geometries.append(traj_line)

        start_sphere = _create_sphere(camera_poses[0].position, radius=0.1, color=[0, 1, 0])
        end_sphere = _create_sphere(camera_poses[-1].position, radius=0.1, color=[1, 0, 0])
        geometries.append(start_sphere)
        geometries.append(end_sphere)

    # 相机视锥体
    if show_frustums and camera_poses is not None:
        for i in range(0, len(camera_poses), frustum_interval):
            frustum = _create_frustum(camera_poses[i], scale=0.3)
            geometries.append(frustum)

    if not geometries:
        logger.warning("没有可渲染的几何体")
        return None

    # 使用不可见窗口渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    for geom in geometries:
        vis.add_geometry(geom)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = float(point_size)

    # 设置默认视角（从远处观察场景中心）
    ctr = vis.get_view_control()
    if geometries:
        # 计算场景中心：优先用点云/轨迹的中心
        if gaussian_data is not None:
            center = np.mean(gaussian_data.bbox, axis=0)
        elif camera_poses:
            positions = np.array([p.position for p in camera_poses])
            center = positions.mean(axis=0)
            # 使用第一个相机位姿的方向作为视角
            forward = -camera_poses[0].rotation_matrix[:, 2]
            eye = camera_poses[0].position + forward * 10.0
            ctr.set_front(forward)
            ctr.set_up(camera_poses[0].rotation_matrix[:, 1])
            ctr.set_lookat(center)
            ctr.set_zoom(1.0)

    vis.poll_events()
    vis.update_renderer()

    # 捕获并保存
    rgb_float = vis.capture_screen_float_buffer(do_render=True)
    rgb = np.asarray(rgb_float)
    rgb = np.clip(rgb[:, :, :3], 0.0, 1.0)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    vis.destroy_window()

    # 保存为 PNG
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from PIL import Image
    Image.fromarray(rgb_uint8).save(output_path)

    logger.info(f"场景截图已保存: {output_path} ({width}x{height})")
    return output_path


# ===================== gsplat 渲染可视化 =====================

def visualize_gsplat(
    gaussian_data,
    camera_poses: list,
    output_path: str = "output/gsplat_video.mp4",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    backend: Optional[str] = None,
):
    """
    使用 gsplat（或最佳可用后端）逐帧渲染轨迹并输出视频。

    复用 renderer.render_view() 和 video_builder.build_video()。
    在 macOS 上无 CUDA 时自动 fallback 到 Open3D 或 numpy。

    Args:
        gaussian_data: GaussianData 对象
        camera_poses: CameraPose 列表
        output_path: 输出视频路径
        width: 图像宽度
        height: 图像高度
        fps: 视频帧率
        backend: 强制指定渲染后端（None 自动选择）

    Returns:
        输出视频的绝对路径
    """
    from src.renderer import render_view, get_renderer_backend
    from src.video_builder import build_video

    if backend is None:
        backend = get_renderer_backend()

    print(f"渲染后端: {backend or '未安装'}")
    print(f"渲染参数: {len(camera_poses)} 帧, {width}x{height}")
    print(f"输出路径: {output_path}")
    print()

    frames = []
    total = len(camera_poses)
    for i, pose in enumerate(camera_poses):
        rgb, _ = render_view(
            gaussian_data, pose, width, height,
            render_depth=False, backend=backend,
        )
        frames.append(rgb)

        if (i + 1) % 30 == 0:
            print(f"  渲染进度: {i+1}/{total}")

    video_path = build_video(frames, output_path, fps=fps)
    print(f"\n✅ 视频已保存: {video_path}")
    return video_path


# ===================== 交互式 gsplat 可视化 =====================

def visualize_gsplat_interactive(
    gaussian_data,
    port: int = 8080,
    width: int = 1280,
    height: int = 720,
    fov: float = 90.0,
    image_format: str = "png",
    gpu_id: int = 0,
):
    """
    使用 viser 启动交互式 gsplat 渲染可视化（纯 viser 实现，无 nerfview 依赖）。

    启动 viser WebSocket 服务器，在浏览器中实时渲染 3DGS 场景。
    鼠标拖拽旋转视角，滚轮缩放。

    Args:
        gaussian_data: GaussianData 对象
        port: viser 服务器端口（默认 8080）
        width: 渲染图像宽度（默认 640）
        height: 渲染图像高度（默认 480）
        fov: 初始垂直视场角（度，默认 90.0）

        image_format: 传输图像格式 ("png" 无损但慢, "jpeg" 快但有损, 默认 png)
        gpu_id: 使用的 GPU 编号（多卡时选择哪张卡，默认 0）

    Note:
        需要 NVIDIA GPU (CUDA)，macOS 不可用。
        推荐分辨率 1280x720 或 1920x1080，640x480 会显得模糊。
        双卡环境可用 --gpu 选择渲染卡，或用 CUDA_VISIBLE_DEVICES 环境变量控制。
    """
    import time
    import platform
    import threading
    import numpy as np
    import torch

    # --- CUDA 检查 ---
    if not torch.cuda.is_available():
        system = platform.system()
        if system == "Darwin":
            msg = (
                "❌ macOS 不支持 CUDA，无法使用 gsplat 交互式渲染。\n"
                "   gsplat 的 rasterization 内核基于 CUDA C++，仅在 NVIDIA GPU 上可用。\n"
                "   \n"
                "   替代方案：\n"
                "   - 使用 Open3D 点云可视化：python main.py visualize\n"
                "   - 在配备 NVIDIA GPU 的 Linux 系统上运行此功能"
            )
        else:
            msg = (
                "❌ 未检测到 CUDA。gsplat 交互式渲染需要 NVIDIA GPU。\n"
                "   请确保：\n"
                "   1. 已安装 NVIDIA 驱动\n"
                "   2. 已安装 CUDA toolkit\n"
                "   3. PyTorch 的 CUDA 版本与系统 CUDA 版本匹配"
            )
        raise RuntimeError(msg)

    # --- GPU 信息 & 选择 ---
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 块 NVIDIA GPU:")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        marker = " ← 选中" if i == gpu_id else ""
        print(f"  GPU {i}: {name} ({mem:.1f} GB){marker}")

    if gpu_id >= num_gpus:
        print(f"⚠️  --gpu {gpu_id} 超出范围，自动切换到 GPU 0")
        gpu_id = 0

    device = torch.device(f"cuda:{gpu_id}")

    # --- 准备高斯参数（转换为 GPU tensor，与 _render_gsplat 保持一致） ---
    means = torch.tensor(gaussian_data.positions, dtype=torch.float32, device=device)
    scales = torch.tensor(gaussian_data.actual_scales, dtype=torch.float32, device=device)
    quats = torch.tensor(gaussian_data.rotations, dtype=torch.float32, device=device)
    opacities = torch.tensor(gaussian_data.actual_opacity, dtype=torch.float32, device=device)

    # SH 颜色处理（复用 renderer.py _render_gsplat 的 SH 逻辑）
    if gaussian_data.f_rest is not None and gaussian_data.f_rest.shape[1] >= 45:
        sh_degree = 3
        sh_coeffs = np.concatenate([gaussian_data.f_dc, gaussian_data.f_rest], axis=-1)  # (N, 48)
        n_gaussians = sh_coeffs.shape[0]
        sh_coeffs = sh_coeffs.reshape(n_gaussians, 16, 3)  # (N, 16, 3)
        colors = torch.tensor(sh_coeffs, dtype=torch.float32, device=device)
    else:
        sh_degree = 0
        colors = torch.tensor(gaussian_data.f_dc[:, np.newaxis, :], dtype=torch.float32, device=device)

    print(f"高斯数量: {means.shape[0]:,}")
    print(f"SH degree: {sh_degree}")
    print(f"设备: {device}")
    print(f"渲染分辨率: {width}x{height}")

    # --- 检查 viser 依赖（纯 viser，无需 nerfview） ---
    try:
        import viser
        import viser.transforms as vt
    except ImportError:
        raise ImportError("交互式可视化需要 viser: pip install viser")

    from gsplat import rasterization

    # --- 计算初始相机位姿（从场景中心向外看） ---
    center = gaussian_data.positions.mean(axis=0)
    bbox_min, bbox_max = gaussian_data.bbox
    scene_extent = float(np.max(np.linalg.norm(gaussian_data.positions - center, axis=1)))
    initial_dist = max(scene_extent * 1.5, 2.0)

    # 根据场景边界推断 up 轴：哪个轴跨度最小就是 up 方向
    bbox_range = bbox_max - bbox_min
    up_axis = int(np.argmin(bbox_range))  # 0=X, 1=Y, 2=Z
    up_axis_name = ['X', 'Y', 'Z'][up_axis]
    print(f"场景跨度: X={bbox_range[0]:.2f}, Y={bbox_range[1]:.2f}, Z={bbox_range[2]:.2f}")
    print(f"检测到 up 轴: {up_axis_name} (跨度最小 {bbox_range[up_axis]:.2f})")

    # 构造初始相机位置：沿水平面内最长的轴方向偏移
    horizontal_axes = [i for i in range(3) if i != up_axis]
    primary_axis = horizontal_axes[np.argmax(bbox_range[horizontal_axes])]

    # 相机沿 primary_axis 正方向偏移，看向场景中心
    offset = np.zeros(3, dtype=np.float64)
    offset[primary_axis] = initial_dist
    initial_position = center + offset

    # 计算 initial_look_at 和 up 向量，供 viser 使用
    # viser 使用 OpenGL 风格相机 (Y-up, Z-backward)
    # 需要设置正确的 look_at 和 up 方向
    look_dir = center - initial_position  # 从相机指向场景中心
    look_dir = look_dir / np.linalg.norm(look_dir)

    # up 向量：沿场景 up 轴
    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up_axis] = 1.0

    # 如果 look_dir 和 up_vec 接近平行，需要调整
    if abs(np.dot(look_dir, up_vec)) > 0.99:
        # 选另一个轴作为临时 up
        for ax in range(3):
            if ax != up_axis:
                up_vec = np.zeros(3, dtype=np.float64)
                up_vec[ax] = 1.0
                break

    print(f"场景中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"初始相机位置: [{initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f}]")
    print(f"初始 look_dir: [{look_dir[0]:.3f}, {look_dir[1]:.3f}, {look_dir[2]:.3f}]")

    # --- 创建 viser 服务器 ---
    server = viser.ViserServer(host="0.0.0.0", port=port)

    # 渲染防重入锁：避免前一次渲染未完成时重复触发
    _render_lock = threading.Lock()

    def render_view(camera):
        """
        从 viser CameraHandle 提取位姿，调用 gsplat rasterization 渲染，
        将结果设为场景背景图像。
        """
        if not _render_lock.acquire(blocking=False):
            # 前一次渲染尚未完成，跳过此帧
            return

        try:
            # 构造 c2w 矩阵（camera-to-world）
            # viser 使用 OpenGL 风格相机（Y-up, Z-backward）
            # gsplat rasterization 期望 OpenCV 风格（Y-down, Z-forward）
            # 需要在 c2w 左乘一个坐标变换矩阵
            # OpenGL -> OpenCV: flip Y 和 Z
            #   [1,  0,  0, 0]
            #   [0, -1,  0, 0]
            #   [0,  0, -1, 0]
            #   [0,  0,  0, 1]
            gl_to_cv = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1],
            ], dtype=np.float32)

            c2w_gl = np.eye(4, dtype=np.float32)
            c2w_gl[:3, :3] = vt.SO3(camera.wxyz).as_matrix()
            c2w_gl[:3, 3] = np.array(camera.position, dtype=np.float32)

            # 检查旋转矩阵是否有效（无 NaN）
            if np.any(np.isnan(c2w_gl)) or np.any(np.isinf(c2w_gl)):
                return

            # 转换为 OpenCV 风格 c2w
            c2w = gl_to_cv @ c2w_gl

            # 构造内参 K 矩阵
            # viser 的 fov 为垂直视场角（弧度）
            fov_rad = camera.fov
            if fov_rad <= 0 or fov_rad >= np.pi:
                fov_rad = np.radians(fov)  # fallback to default
            fy = height / (2.0 * np.tan(fov_rad / 2.0))
            fx = fy * width / height  # 保持像素正方形
            K = np.array(
                [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            # 转 GPU tensor
            c2w_t = torch.from_numpy(c2w).to(device)
            K_t = torch.from_numpy(K).to(device)

            # 计算 world-to-camera (viewmat)
            # 使用 torch.linalg.inv 并加异常保护
            try:
                viewmat = torch.linalg.inv(c2w_t).unsqueeze(0)  # (1, 4, 4) world-to-camera
            except torch._C._LinAlgError:
                # 奇异矩阵，跳过此帧
                return

            Ks = K_t.unsqueeze(0)  # (1, 3, 3)

            with torch.inference_mode():
                # gsplat 1.5.x: 不传 backgrounds 参数
                render_colors, render_alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    render_mode="RGB",
                )

            rgb = render_colors[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
            alpha = render_alphas[0, ..., 0].cpu().numpy()
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            # 调试信息（仅前 3 帧）
            if not hasattr(render_view, '_debug_count'):
                render_view._debug_count = 0
            if render_view._debug_count < 3:
                print(f"  [debug] rgb min={rgb.min():.3f} max={rgb.max():.3f} mean={rgb.mean():.3f}")
                print(f"  [debug] alpha min={alpha.min():.3f} max={alpha.max():.3f} mean={alpha.mean():.3f}")
                print(f"  [debug] cam_pos_gl={c2w_gl[:3,3]}")
                print(f"  [debug] cam_pos_cv={c2w[:3,3]}")
                render_view._debug_count += 1

            # 将渲染结果设为场景背景
            server.scene.set_background_image(rgb_uint8, format=image_format)
        finally:
            _render_lock.release()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # 设置初始相机位姿
        # viser 使用 OpenGL 风格 (Y-up, Z-backward)
        # look_at 指向场景中心，position 在场景外侧
        client.camera.position = tuple(initial_position.tolist())
        client.camera.look_at = tuple(center.tolist())
        client.camera.fov = float(np.radians(fov))
        # up 向量：viser 默认 Y-up，如果场景 up 轴不是 Y，需要设置
        # viser 的 up 向量通过 camera.up_direction 设置
        if hasattr(client.camera, 'up_direction'):
            client.camera.up_direction = tuple(up_vec.tolist())

        # 注册相机更新回调：每次相机变化时重新渲染
        @client.camera.on_update
        def _(_: viser.CameraHandle):
            render_view(client.camera)

        # 初始渲染
        render_view(client.camera)

    print(f"\n{'='*55}")
    print(f"  🎨 gsplat 交互式可视化已启动")
    print(f"  🌐 打开浏览器访问: http://localhost:{port}")
    print(f"  🖱️  鼠标拖拽旋转 | 滚轮缩放")
    print(f"  ⏹️  按 Ctrl+C 退出")
    print(f"{'='*55}\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 可视化已停止")


def visualize_preview(
    gaussian_data,
    camera_poses: list,
    output_dir: str = "output/preview",
    width: int = 640,
    height: int = 480,
    backend: Optional[str] = None,
):
    """
    渲染轨迹关键帧预览图片（起点、1/4、1/2、3/4、终点）。

    复用 renderer.render_view()，每帧保存为 PNG 文件。

    Args:
        gaussian_data: GaussianData 对象
        camera_poses: CameraPose 列表
        output_dir: 输出目录
        width: 图像宽度
        height: 图像高度
        backend: 强制指定渲染后端（None 自动选择）

    Returns:
        保存的 PNG 文件路径列表
    """
    import os
    from PIL import Image
    from src.renderer import render_view, get_renderer_backend

    if backend is None:
        backend = get_renderer_backend()

    total = len(camera_poses)
    if total < 2:
        logger.warning("轨迹帧数不足，无法生成预览")
        return []

    # 关键帧索引: 起点, 1/4, 1/2, 3/4, 终点
    keyframe_indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    # 去重并保持顺序
    seen = set()
    keyframe_indices = [i for i in keyframe_indices if not (i in seen or seen.add(i))]

    labels = {
        0: "start",
        total // 4: "quarter",
        total // 2: "middle",
        3 * total // 4: "three_quarter",
        total - 1: "end",
    }

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"渲染后端: {backend or '未安装'}")
    print(f"预览关键帧: {len(keyframe_indices)} 帧, {width}x{height}")
    print(f"输出目录: {output_dir}")
    print()

    saved_paths = []
    for idx in keyframe_indices:
        pose = camera_poses[idx]
        label = labels.get(idx, f"frame_{idx:04d}")

        rgb, _ = render_view(
            gaussian_data, pose, width, height,
            render_depth=False, backend=backend,
        )

        out_path = os.path.join(output_dir, f"preview_{label}_{idx:04d}.png")
        Image.fromarray(rgb).save(out_path)
        saved_paths.append(out_path)
        print(f"  已保存: {out_path}")

    print(f"\n✅ 预览图已保存至: {output_dir}/")
    return saved_paths
