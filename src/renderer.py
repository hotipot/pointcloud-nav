"""
3DGS 渲染器模块

支持使用 gsplat 进行高质量 3DGS 渲染，如果 gsplat 安装困难，
则 fallback 到 Open3D 的点云渲染。在 macOS 无头环境下，
Open3D 的 OffscreenRenderer 不支持 EGL，此时使用纯 numpy
点云投影渲染作为最终 fallback。

渲染功能包括：
- 指定相机位姿的 RGB 图像渲染
- 深度图渲染
- 批量渲染
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

import sys
import platform

_IS_MACOS = platform.system() == "Darwin"


def _detect_renderer_backend():
    """检测可用的渲染后端：gsplat → Open3D Offscreen → Open3D Visualizer → numpy"""

    # 1. 尝试 gsplat（需要 PyTorch + CUDA，macOS 通常不可用）
    try:
        import gsplat
        logger.info("渲染后端: gsplat")
        return "gsplat"
    except ImportError:
        pass

    # 2. 尝试 Open3D
    try:
        import open3d as o3d

        if _IS_MACOS:
            # macOS: 优先 Visualizer(visible=False)，再用 OffscreenRenderer
            # OffscreenRenderer 需要 EGL，macOS 不支持，但作为 fallback 尝试
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=64, height=64)
                vis.destroy_window()
                logger.info("渲染后端: Open3D Visualizer (visible=False)")
                return "open3d_vis"
            except Exception:
                pass

            try:
                import open3d.visualization.rendering as rendering
                test = rendering.OffscreenRenderer(64, 64)
                del test
                logger.info("渲染后端: Open3D OffscreenRenderer")
                return "open3d"
            except (RuntimeError, Exception):
                pass
        else:
            # Linux/Windows: 优先 OffscreenRenderer（需要 EGL/headless 支持）
            try:
                import open3d.visualization.rendering as rendering
                test = rendering.OffscreenRenderer(64, 64)
                del test
                logger.info("渲染后端: Open3D OffscreenRenderer")
                return "open3d"
            except (RuntimeError, Exception):
                pass

            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=64, height=64)
                vis.destroy_window()
                logger.info("渲染后端: Open3D Visualizer (visible=False)")
                return "open3d_vis"
            except Exception:
                pass
    except ImportError:
        pass

    # 3. numpy 纯 CPU 渲染（最终 fallback）
    logger.info("渲染后端: numpy 投影渲染 (gsplat/Open3D 均不可用)")
    return "numpy"


_RENDERER_BACKEND = _detect_renderer_backend()


def get_renderer_backend() -> Optional[str]:
    """获取当前渲染后端"""
    return _RENDERER_BACKEND


# ===================== gsplat 渲染器 =====================

def _render_gsplat(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: float = 90.0,
    render_depth: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用 gsplat 渲染单帧

    Args:
        gaussian_data: GaussianData 对象
        camera_pose: CameraPose 对象
        width: 图像宽度
        height: 图像高度
        fov: 水平视场角（度）
        render_depth: 是否渲染深度图

    Returns:
        (rgb_image, depth_image) - rgb 为 (H, W, 3) uint8，depth 为 (H, W) float32 或 None
    """
    import torch
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备高斯参数
    means = torch.tensor(gaussian_data.positions, dtype=torch.float32, device=device)
    scales = torch.tensor(gaussian_data.actual_scales, dtype=torch.float32, device=device)
    quats = torch.tensor(gaussian_data.rotations, dtype=torch.float32, device=device)
    opacities = torch.tensor(gaussian_data.actual_opacity, dtype=torch.float32, device=device)

    # SH 颜色
    C0 = 0.28209479177387814
    sh_degree = 0
    if gaussian_data.f_rest is not None and gaussian_data.f_rest.shape[1] >= 45:
        sh_degree = 3
        colors = torch.tensor(
            np.concatenate([gaussian_data.f_dc, gaussian_data.f_rest], axis=-1),
            dtype=torch.float32, device=device,
        )
    else:
        sh_degree = 0
        colors = torch.tensor(gaussian_data.f_dc, dtype=torch.float32, device=device)

    # 视图矩阵 (world-to-camera)
    viewmat = torch.tensor(camera_pose.view_matrix, dtype=torch.float32, device=device).T

    # 焦距计算
    fx = width / (2 * np.tan(np.radians(fov / 2)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    # 渲染
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        background_color=torch.zeros(3, device=device),
        K=torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=device),
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_depth=render_depth,
    )

    rgb = render_colors[0, :, :, :3].cpu().numpy()
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    depth = None
    if render_depth and "depth" in info:
        depth = info["depth"][0, 0].cpu().numpy()

    return rgb, depth


# ===================== numpy 投影渲染器 =====================

def _render_numpy(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: float = 90.0,
    render_depth: bool = False,
    max_gaussians: int = 200000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用纯 numpy 实现点云 splat 渲染（CPU fallback）

    将 3DGS 高斯体投影到图像平面，根据 scale 属性计算屏幕空间 splat 半径，
    使用深度排序（画家算法）和 alpha blending 进行渲染。
    比简单点投影更接近真实 3DGS 外观。

    Args:
        gaussian_data: GaussianData 对象
        camera_pose: CameraPose 对象
        width: 图像宽度
        height: 图像高度
        fov: 水平视场角（度）
        render_depth: 是否渲染深度图
        max_gaussians: 最多使用的高斯体数量（超过则随机降采样）

    Returns:
        (rgb_image, depth_image) - rgb 为 (H, W, 3) uint8，depth 为 (H, W) float32 或 None
    """
    import gc

    # 相机参数
    R = camera_pose.rotation_matrix  # 3x3 camera-to-world rotation
    t = camera_pose.position  # (3,)

    # 焦距计算
    fx = width / (2 * np.tan(np.radians(fov / 2)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    # 世界坐标转相机坐标: p_cam = R^T @ (p_world - t)
    positions = gaussian_data.positions  # (N, 3)
    p_cam = (positions - t) @ R  # (N, 3)

    # 只保留在相机前方的点 (z > near)
    near = 0.01
    far = 20.0  # 室内巡逻：只渲染 20m 以内的点
    valid_mask = (p_cam[:, 2] > near) & (p_cam[:, 2] < far)
    if not np.any(valid_mask):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32) if render_depth else None
        return rgb, depth

    p_cam = p_cam[valid_mask]
    colors = gaussian_data.rgb_colors[valid_mask]
    opacities = gaussian_data.actual_opacity[valid_mask]
    scales_3d = gaussian_data.actual_scales[valid_mask]

    # 透视投影到图像平面
    z = p_cam[:, 2]
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy

    # 计算屏幕空间 splat 半径：取 3 轴最大 scale，按透视投影缩放
    max_scale = np.max(scales_3d, axis=1)
    radius_px = fx * max_scale / z
    # 限制半径范围，避免过大或过小的 splat
    radius_px = np.clip(np.ceil(radius_px), 1, 50).astype(np.int32)

    # 筛选在图像范围内（含边界扩展）的点
    margin = min(int(radius_px.max()), 50)
    in_image = (u >= -margin) & (u < width + margin) & (v >= -margin) & (v < height + margin)
    if not np.any(in_image):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32) if render_depth else None
        return rgb, depth

    u = u[in_image]; v = v[in_image]; z = z[in_image]
    colors = colors[in_image]; opacities = opacities[in_image]; radius_px = radius_px[in_image]

    # 随机降采样以控制渲染时间
    if len(u) > max_gaussians:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(u), max_gaussians, replace=False)
        u = u[indices]; v = v[indices]; z = z[indices]
        colors = colors[indices]; opacities = opacities[indices]; radius_px = radius_px[indices]

    # 距离衰减：近处点更不透明，远处点更透明（模拟室内视角）
    # 衰减函数：opacity *= exp(-z / decay_distance)
    decay_distance = 8.0  # 8m 处衰减到 1/e
    distance_decay = np.exp(-z / decay_distance)
    opacities = opacities * distance_decay

    # 按深度排序（远的先画，近的覆盖远的 — 画家算法）
    sort_idx = np.argsort(-z)  # 从远到近
    u = u[sort_idx]; v = v[sort_idx]; z = z[sort_idx]
    colors = colors[sort_idx]; opacities = opacities[sort_idx]; radius_px = radius_px[sort_idx]

    # 渲染缓冲区
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    alpha_accum = np.zeros((height, width), dtype=np.float64)

    # 预计算不同半径的高斯核，避免重复计算
    kernels = {}

    for i in range(len(u)):
        ui, vi, ri = int(u[i]), int(v[i]), int(radius_px[i])
        alpha_i = float(opacities[i])
        color_i = colors[i]

        # 计算 splat 在图像上的裁剪范围
        y0 = max(0, vi - ri)
        y1 = min(height, vi + ri + 1)
        x0 = max(0, ui - ri)
        x1 = min(width, ui + ri + 1)

        if y0 >= y1 or x0 >= x1:
            continue

        # 获取或创建高斯核
        if ri not in kernels:
            ks = 2 * ri + 1
            ky, kx = np.ogrid[-ri:ri + 1, -ri:ri + 1]
            sigma = ri / 2.0
            kernels[ri] = np.exp(-0.5 * (kx * kx + ky * ky) / (sigma * sigma))

        kernel = kernels[ri]

        # 将核对齐到图像裁剪区域
        k_y0 = y0 - (vi - ri)
        k_y1 = y1 - (vi - ri)
        k_x0 = x0 - (ui - ri)
        k_x1 = x1 - (ui - ri)
        weight = kernel[k_y0:k_y1, k_x0:k_x1]

        # Alpha compositing: output = existing * (1 - alpha) + new_color * alpha
        contrib = weight * alpha_i
        contrib_3d = contrib[:, :, np.newaxis]

        patch_rgb = rgb[y0:y1, x0:x1]
        patch_alpha = alpha_accum[y0:y1, x0:x1]

        rgb[y0:y1, x0:x1] = patch_rgb * (1.0 - contrib_3d) + color_i[np.newaxis, np.newaxis, :] * contrib_3d
        alpha_accum[y0:y1, x0:x1] = patch_alpha + contrib * (1.0 - patch_alpha)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # 渲染深度图
    depth = None
    if render_depth:
        depth = np.full((height, width), np.inf, dtype=np.float32)
        for i in range(len(u)):
            ui, vi, ri = int(u[i]), int(v[i]), int(radius_px[i])
            y0 = max(0, vi - ri)
            y1 = min(height, vi + ri + 1)
            x0 = max(0, ui - ri)
            x1 = min(width, ui + ri + 1)
            if y0 < y1 and x0 < x1:
                depth[y0:y1, x0:x1] = np.minimum(depth[y0:y1, x0:x1], z[i])
        depth[depth == np.inf] = 0.0

    return rgb, depth


# ===================== Open3D 渲染器 =====================

def _render_open3d(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: float = 90.0,
    render_depth: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用 Open3D 渲染单帧（fallback 方案）

    将 3DGS 高斯体转换为彩色点云进行渲染
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussian_data.positions)
    pcd.colors = o3d.utility.Vector3dVector(gaussian_data.rgb_colors)

    # 创建离线渲染器
    renderer = rendering.OffscreenRenderer(width, height)

    # 添加点云
    mat = rendering.MaterialRecord()
    mat.point_size = 2.0
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("point_cloud", pcd, mat)

    # 设置相机
    # Open3D 相机坐标系：X 右，Y 下，Z 前
    R = camera_pose.rotation_matrix
    eye = camera_pose.position

    # 计算 look-at 和 up
    forward = -R[:, 2]  # 相机看向 -Z
    target = eye + forward * 10.0
    up = R[:, 1]  # Y 轴

    renderer.setup_camera(fov, eye, target, up)

    # 设置背景色
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    # 渲染 RGB
    rgb_image = renderer.render_to_image()
    rgb = np.asarray(rgb_image)

    # 渲染深度
    depth = None
    if render_depth:
        depth_image = renderer.render_to_depth_image()
        depth = np.asarray(depth_image).astype(np.float32)

    return rgb, depth


# ===================== Open3D Visualizer 渲染器 =====================

def _render_open3d_vis(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: float = 90.0,
    render_depth: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用 Open3D Visualizer (visible=False) 渲染单帧

    适用于 macOS 环境，OffscreenRenderer 需要 EGL 在 macOS 上不可用，
    但 Visualizer.create_window(visible=False) + capture_screen_float_buffer 可用，
    即使显示器处于休眠状态。
    """
    import open3d as o3d

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussian_data.positions)
    pcd.colors = o3d.utility.Vector3dVector(gaussian_data.rgb_colors)

    # 创建不可见窗口的渲染器
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 2.0

    # 设置相机
    ctr = vis.get_view_control()

    R = camera_pose.rotation_matrix
    eye = camera_pose.position

    # 相机坐标系：X 右，Y 下，Z 前；forward = -R[:, 2]
    forward = -R[:, 2]
    lookat = eye + forward * 5.0
    up = R[:, 1]

    # 将 FOV 转换为 Open3D Visualizer 的 zoom 参数
    # zoom ≈ 1 / tan(fov_vertical / 2)
    aspect = width / height if height > 0 else 1.0
    fov_rad = np.radians(fov)
    fov_y = 2.0 * np.arctan(np.tan(fov_rad / 2.0) / aspect)
    zoom = 1.0 / np.tan(fov_y / 2.0)

    ctr.set_lookat(lookat)
    ctr.set_front(forward)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    # 渲染
    vis.poll_events()
    vis.update_renderer()

    rgb_float = vis.capture_screen_float_buffer(do_render=True)
    rgb = np.asarray(rgb_float)
    rgb = np.clip(rgb[:, :, :3], 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    # 渲染深度
    depth = None
    if render_depth:
        depth_float = vis.capture_depth_float_buffer(do_render=True)
        depth = np.asarray(depth_float).astype(np.float32)

    vis.destroy_window()
    return rgb, depth


# ===================== 统一接口 =====================

def render_view(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: Optional[float] = None,
    render_depth: bool = False,
    backend: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    渲染单帧图像

    Args:
        gaussian_data: GaussianData 对象
        camera_pose: CameraPose 对象
        width: 图像宽度
        height: 图像高度
        fov: 视场角（None 则使用 camera_pose.fov）
        render_depth: 是否渲染深度图
        backend: 强制指定后端 ("gsplat" / "open3d")，None 自动选择

    Returns:
        (rgb_image, depth_image)
        - rgb_image: (H, W, 3) uint8
        - depth_image: (H, W) float32 或 None
    """
    if fov is None:
        fov = camera_pose.fov

    use_backend = backend or _RENDERER_BACKEND

    if use_backend == "gsplat":
        return _render_gsplat(gaussian_data, camera_pose, width, height, fov, render_depth)
    elif use_backend == "open3d":
        return _render_open3d(gaussian_data, camera_pose, width, height, fov, render_depth)
    elif use_backend == "open3d_vis":
        return _render_open3d_vis(gaussian_data, camera_pose, width, height, fov, render_depth)
    elif use_backend == "numpy":
        return _render_numpy(gaussian_data, camera_pose, width, height, fov, render_depth)
    else:
        raise RuntimeError(
            f"没有可用的渲染后端！请安装 gsplat、Open3D 或确保 numpy 可用。\n"
            f"  pip install gsplat\n"
            f"  pip install open3d"
        )


def render_depth(
    gaussian_data,
    camera_pose,
    width: int = 640,
    height: int = 480,
    fov: Optional[float] = None,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    渲染深度图

    Returns:
        depth_image: (H, W) float32
    """
    _, depth = render_view(
        gaussian_data, camera_pose, width, height, fov,
        render_depth=True, backend=backend,
    )
    return depth


def render_batch(
    gaussian_data,
    camera_poses: list,
    width: int = 640,
    height: int = 480,
    render_depth: bool = False,
    backend: Optional[str] = None,
    callback=None,
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    批量渲染

    Args:
        gaussian_data: GaussianData 对象
        camera_poses: CameraPose 列表
        width: 图像宽度
        height: 图像高度
        render_depth: 是否渲染深度图
        backend: 渲染后端
        callback: 进度回调 callback(frame_idx, total_frames)

    Returns:
        List[(rgb, depth)]
    """
    total = len(camera_poses)
    results = []

    for i, pose in enumerate(camera_poses):
        rgb, depth = render_view(
            gaussian_data, pose, width, height,
            render_depth=render_depth, backend=backend,
        )
        results.append((rgb, depth))

        if callback:
            callback(i, total)

        if (i + 1) % 50 == 0:
            logger.info(f"渲染进度: {i + 1}/{total}")

    return results
