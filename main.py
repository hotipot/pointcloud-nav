#!/usr/bin/env python3
"""
点云导航项目 - 主入口

用法:
    python main.py visualize          # 可视化场景
    python main.py render-video       # 渲染视频
    python main.py collect-vln        # 采集 VLN 数据
    python main.py info               # 查看 PLY 文件信息
"""

import argparse
import logging
import os
import sys
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ply_loader import load_3dgs_ply
from src.trajectory import (
    Trajectory, Waypoint,
    create_default_trajectory,
    create_zigzag_trajectory,
    create_interior_patrol_trajectory,
    create_interior_explore_trajectory,
    create_look_around_trajectory,
    waypoints_from_list,
)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    config_path = os.path.expanduser(config_path)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_gaussian_data(config: dict, verbose: bool = True):
    """加载 PLY 数据"""
    ply_path = config.get("scene", {}).get("ply_path")
    if not ply_path:
        raise ValueError(
            "配置文件缺少 scene.ply_path，请检查 configs/default.yaml\n"
            "示例:\n"
            "  scene:\n"
            "    ply_path: \"~/.openclaw/workspace-coder/pointcloud-nav/pointcloud_data/Aholo-603.ply\""
        )
    return load_3dgs_ply(ply_path, verbose=verbose)


def build_trajectory(config: dict, gaussian_data, traj_type: str = "default"):
    """根据配置构建轨迹"""
    robot_cfg = config.get("robot", {})
    traj_cfg = config.get("trajectory", {})

    robot_height = robot_cfg.get("height", 1.5)
    fov = robot_cfg.get("fov", 90.0)
    speed = traj_cfg.get("speed", 1.0)
    fps = traj_cfg.get("fps", 30.0)

    ground_func = gaussian_data.get_ground_height if gaussian_data else None
    bbox_min, bbox_max = gaussian_data.bbox
    positions = gaussian_data.positions if gaussian_data else None

    # 如果有自定义 waypoints
    custom_waypoints = traj_cfg.get("waypoints", [])
    if custom_waypoints and len(custom_waypoints) >= 2:
        return waypoints_from_list(
            custom_waypoints,
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=speed,
            fps=fps,
            fov=fov,
        )

    # 根据类型生成默认轨迹
    if traj_type == "zigzag":
        return create_zigzag_trajectory(
            bbox_min, bbox_max,
            rows=traj_cfg.get("rows", 3),
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=speed,
            fps=fps,
            fov=fov,
            positions=positions,
        )
    elif traj_type == "interior-patrol":
        return create_interior_patrol_trajectory(
            bbox_min, bbox_max,
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=speed,
            fps=fps,
            fov=fov,
            positions=positions,
            grid_size=traj_cfg.get("grid_size", 1.0),
            density_percentile=traj_cfg.get("density_percentile", 30.0),
        )
    elif traj_type == "interior-explore":
        return create_interior_explore_trajectory(
            bbox_min, bbox_max,
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=traj_cfg.get("speed", 0.5),
            fps=fps,
            fov=fov,
            positions=positions,
            grid_size=traj_cfg.get("grid_size", 1.0),
            density_percentile=traj_cfg.get("density_percentile", 30.0),
            look_around_speed=traj_cfg.get("look_around_speed", 0.3),
        )
    elif traj_type == "look-around":
        return create_look_around_trajectory(
            bbox_min, bbox_max,
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=traj_cfg.get("speed", 0.3),
            fps=fps,
            fov=fov,
            positions=positions,
            grid_size=traj_cfg.get("grid_size", 1.0),
            density_percentile=traj_cfg.get("density_percentile", 30.0),
            rotation_speed=traj_cfg.get("rotation_speed", 60.0),
        )
    else:
        return create_default_trajectory(
            bbox_min, bbox_max,
            robot_height=robot_height,
            ground_height_func=ground_func,
            speed=speed,
            fps=fps,
            fov=fov,
            positions=positions,
        )


def cmd_info(args):
    """查看 PLY 文件信息"""
    gaussian = get_gaussian_data(load_config(args.config))
    bbox_min, bbox_max = gaussian.bbox
    print(f"\n{'='*50}")
    print(f"  3DGS PLY 文件信息")
    print(f"{'='*50}")
    print(f"  高斯数量: {gaussian.count:,}")
    print(f"  场景边界:")
    print(f"    X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] (跨度 {bbox_max[0]-bbox_min[0]:.2f}m)")
    print(f"    Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] (跨度 {bbox_max[1]-bbox_min[1]:.2f}m)")
    print(f"    Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}] (跨度 {bbox_max[2]-bbox_min[2]:.2f}m)")
    print(f"  地面平面:")
    normal, dist = gaussian.ground_plane
    print(f"    法向量: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
    print(f"    距离: {dist:.2f}")
    print(f"{'='*50}\n")


def cmd_visualize(args):
    """可视化场景"""
    from src.visualize import visualize_scene, visualize_trajectory_only, render_scene_screenshot

    config = load_config(args.config)
    gaussian = get_gaussian_data(config)
    traj = build_trajectory(config, gaussian, traj_type=args.trajectory_type)
    camera_poses = traj.generate()

    print(f"轨迹信息: {len(camera_poses)} 帧, 总距离 {traj.get_total_distance():.2f}m")

    # 截图模式：使用 offscreen 渲染，不弹出窗口
    if args.save_screenshot is not None:
        screenshot_path = render_scene_screenshot(
            gaussian_data=None if args.trajectory_only else gaussian,
            camera_poses=camera_poses,
            output_path=args.save_screenshot,
            show_trajectory=True,
            show_frustums=True,
            point_size=float(args.point_size),
            frustum_interval=int(args.frustum_interval),
            trajectory_only=args.trajectory_only,
        )
        if screenshot_path:
            print(f"\n✅ 截图已保存: {screenshot_path}")
        return

    if args.trajectory_only:
        visualize_trajectory_only(camera_poses, *gaussian.bbox)
    else:
        visualize_scene(
            gaussian,
            camera_poses=camera_poses,
            show_trajectory=True,
            show_frustums=True,
            point_size=float(args.point_size),
            frustum_interval=int(args.frustum_interval),
        )


def cmd_render_video(args):
    """渲染视频"""
    from src.renderer import render_view, get_renderer_backend
    from src.video_builder import build_video

    config = load_config(args.config)
    gaussian = get_gaussian_data(config)
    traj = build_trajectory(config, gaussian, traj_type=args.trajectory_type)
    camera_poses = traj.generate()

    cam_cfg = config.get("camera", {})
    width, height = cam_cfg.get("resolution", [640, 480])
    if args.width:
        width = int(args.width)
    if args.height:
        height = int(args.height)

    output_cfg = config.get("output", {})
    output_path = args.output or output_cfg.get("video_path", "output/video.mp4")

    backend = get_renderer_backend()
    print(f"渲染后端: {backend or '未安装'}")
    print(f"渲染参数: {len(camera_poses)} 帧, {width}x{height}")
    print(f"输出路径: {output_path}")
    print()

    # 逐帧渲染
    frames = []
    for i, pose in enumerate(camera_poses):
        rgb, _ = render_view(gaussian, pose, width, height, render_depth=False, backend=backend)
        frames.append(rgb)

        if (i + 1) % 30 == 0:
            print(f"  渲染进度: {i+1}/{len(camera_poses)}")

    # 合成视频
    fps = config.get("trajectory", {}).get("fps", 30.0)
    video_path = build_video(frames, output_path, fps=fps)
    print(f"\n✅ 视频已保存: {video_path}")


def cmd_collect_vln(args):
    """采集 VLN 数据"""
    from src.vln_collector import collect_vln_data

    config = load_config(args.config)
    gaussian = get_gaussian_data(config)
    traj = build_trajectory(config, gaussian, traj_type=args.trajectory_type)
    camera_poses = traj.generate()

    cam_cfg = config.get("camera", {})
    width, height = cam_cfg.get("resolution", [640, 480])

    output_cfg = config.get("output", {})
    output_dir = args.output or output_cfg.get("vln_data_path", "output/vln_data")

    ply_path = config.get("scene", {}).get("ply_path", "")
    scene_id = os.path.splitext(os.path.basename(ply_path))[0] if ply_path else "scene"

    print(f"VLN 数据采集: {len(camera_poses)} 帧 -> {output_dir}")

    vln_data = collect_vln_data(
        gaussian_data=gaussian,
        camera_poses=camera_poses,
        output_dir=output_dir,
        scene_id=scene_id,
        width=width,
        height=height,
        render_depth=not args.no_depth,
    )

    print(f"\n✅ VLN 数据已保存: {output_dir}")
    print(f"  总帧数: {vln_data['total_frames']}")
    print(f"  图像目录: {output_dir}/frames/")
    print(f"  元数据: {output_dir}/vln_data.json")


def main():
    parser = argparse.ArgumentParser(
        description="点云导航 - 3DGS 场景渲染与 VLN 数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py info                              # 查看 PLY 文件信息
  python main.py visualize                         # 可视化场景 + 默认轨迹
  python main.py visualize --trajectory-only       # 仅查看轨迹
  python main.py visualize -t zigzag               # 之字形轨迹
  python main.py render-video                      # 渲染视频
  python main.py render-video -o my_video.mp4      # 指定输出路径
  python main.py collect-vln                       # 采集 VLN 数据
  python main.py collect-vln --no-depth            # 不采集深度图
        """,
    )

    parser.add_argument("-c", "--config", default="configs/default.yaml",
                        help="配置文件路径 (默认: configs/default.yaml)")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # info 命令
    p_info = subparsers.add_parser("info", help="查看 PLY 文件信息")

    # visualize 命令
    p_vis = subparsers.add_parser("visualize", help="可视化场景")
    p_vis.add_argument("-t", "--trajectory-type", default="default",
                       choices=["default", "zigzag", "interior-patrol", "interior-explore", "look-around"],
                       help="轨迹类型")
    p_vis.add_argument("--trajectory-only", action="store_true",
                       help="仅显示轨迹，不加载点云（更快）")
    p_vis.add_argument("--point-size", default="2.0", help="点云渲染大小")
    p_vis.add_argument("--frustum-interval", default="30", help="视锥体显示间隔")
    p_vis.add_argument("--save-screenshot", default=None, metavar="PATH",
                       help="保存场景截图为 PNG 文件（使用 offscreen 渲染，不弹出窗口）")

    # render-video 命令
    p_video = subparsers.add_parser("render-video", help="渲染视频")
    p_video.add_argument("-t", "--trajectory-type", default="default",
                         choices=["default", "zigzag", "interior-patrol", "interior-explore", "look-around"],
                         help="轨迹类型")
    p_video.add_argument("-o", "--output", default=None, help="输出视频路径")
    p_video.add_argument("--width", default=None, help="图像宽度")
    p_video.add_argument("--height", default=None, help="图像高度")

    # collect-vln 命令
    p_vln = subparsers.add_parser("collect-vln", help="采集 VLN 数据")
    p_vln.add_argument("-t", "--trajectory-type", default="default",
                       choices=["default", "zigzag", "interior-patrol", "interior-explore", "look-around"],
                       help="轨迹类型")
    p_vln.add_argument("-o", "--output", default=None, help="输出目录")
    p_vln.add_argument("--no-depth", action="store_true", help="不采集深度图")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "info":
        cmd_info(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "render-video":
        cmd_render_video(args)
    elif args.command == "collect-vln":
        cmd_collect_vln(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
