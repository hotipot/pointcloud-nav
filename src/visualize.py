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
