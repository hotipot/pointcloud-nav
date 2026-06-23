"""RS30 数据解析器 — CLI 入口"""

import sys
import logging
from .to_colmap import convert_site_to_colmap


def main():
    """CLI 入口点"""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(
        prog="rs30_parser",
        description="RS30 移动扫描数据解析与转换工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # to_colmap 子命令
    colmap_parser = subparsers.add_parser(
        "to_colmap",
        help="一键转换站点数据为 COLMAP 格式（支持 SuGaR/3DGS）",
    )
    colmap_parser.add_argument("site_dir", help="RS30 站点数据目录")
    colmap_parser.add_argument("-o", "--output", default=None, help="输出 COLMAP 项目目录")
    colmap_parser.add_argument("--cameras", nargs="*", help="指定相机（如 Camera1 Camera2）")
    colmap_parser.add_argument("--max-frames", type=int, default=0, help="每相机最大帧数")
    colmap_parser.add_argument("--with-poses", action="store_true", help="写入 SLAM 位姿")
    colmap_parser.add_argument("--no-undistort", action="store_true",
                               help="不去畸变图像（保留 OPENCV 模型，SuGaR 不支持）")
    colmap_parser.add_argument("--force-pinhole", action="store_true",
                               help="忽略畸变，强制 PINHOLE 模型（不推荐）")
    
    # extract 子命令
    extract_parser = subparsers.add_parser(
        "extract",
        help="提取 .cam 文件中的 JPEG 图像",
    )
    extract_parser.add_argument("input", help="站点数据目录或 .cam 文件路径")
    extract_parser.add_argument("-o", "--output", default="./output", help="输出目录")
    extract_parser.add_argument("--cameras", nargs="*", help="指定相机")
    extract_parser.add_argument("--max-frames", type=int, default=0, help="每相机最大帧数")
    extract_parser.add_argument("--single", action="store_true", help="输入是单个 .cam 文件")
    
    # info 子命令
    info_parser = subparsers.add_parser(
        "info",
        help="查看站点数据信息",
    )
    info_parser.add_argument("site_dir", help="RS30 站点数据目录")
    info_parser.add_argument("--colmap", action="store_true", help="输出 COLMAP 格式参数")
    
    args = parser.parse_args()
    
    if args.command == "to_colmap":
        output = args.output or None
        convert_site_to_colmap(
            site_dir=args.site_dir,
            output_dir=output,
            cameras=args.cameras,
            max_frames_per_camera=args.max_frames,
            write_initial_poses=args.with_poses,
            undistort=not args.no_undistort,
            force_pinhole=args.force_pinhole,
        )
    elif args.command == "extract":
        from .extract_images import extract_site_images, extract_all_images
        if args.single:
            extract_all_images(args.input, args.output, max_frames=args.max_frames)
        else:
            extract_site_images(args.input, args.output, cameras=args.cameras, max_frames=args.max_frames)
    elif args.command == "info":
        from .parse_camera_params import parse_all_cameras, intrinsics_to_colmap_dict
        from pathlib import Path
        site_path = Path(args.site_dir)
        
        print(f"站点: {site_path.name}")
        
        # 解析相机内参
        intrinsics = parse_all_cameras(args.site_dir)
        for name, intr in intrinsics.items():
            print(f"\n=== {name} ===")
            if args.colmap:
                colmap_dict = intrinsics_to_colmap_dict(intr)
                print(f"  Model: {colmap_dict['model']}")
                print(f"  Width: {colmap_dict['width']}, Height: {colmap_dict['height']}")
                print(f"  Params: {colmap_dict['params']}")
            else:
                print(f"  fx={intr.fx:.2f}, fy={intr.fy:.2f}")
                print(f"  cx={intr.cx:.2f}, cy={intr.cy:.2f}")
                print(f"  k1={intr.k1:.6f}, k2={intr.k2:.6f}")
                print(f"  p1={intr.p1:.6f}, p2={intr.p2:.6f}")
                print(f"  Width: {intr.image_width}, Height: {intr.image_height}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
