#!/usr/bin/env python3
"""
RS30 三路摄像头 COLMAP + SuGaR 全流程脚本

一键完成：
  1. 从 .cam 文件提取 3 路摄像头图像
  2. 解析 .CP 内参 + 去畸变
  3. 生成 COLMAP 项目
  4. 运行 COLMAP SfM（特征提取 + 匹配 + 重建）
  5. 选择最佳稀疏模型
  6. 运行 SuGaR 全流程训练

使用方法：
  python run_3cam_pipeline.py \
      --site-dir /home/wm1/jwang/dataset/baoding/2026-06-13-BD-GLKGQ \
      --output-dir /home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam \
      --sugar-dir /home/wm1/jwang/pointcloud-nav/SuGaR \
      --gpu 0

注意：
  - Camera4 是全景相机，默认跳过
  - 去畸变默认开启（RS30 k1=0.39，必须去畸变）
  - COLMAP 特征匹配对 1443 张图可能很慢，可选降采样
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
import struct
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step 1: 数据准备（调用 rs30_parser）
# ──────────────────────────────────────────────

def step1_prepare_data(site_dir, output_dir, cameras, max_frames, skip_existing):
    """用 rs30_parser 提取图像 + 去畸变 + 生成 COLMAP 项目"""
    logger.info("=" * 60)
    logger.info("Step 1: 数据准备 — 提取图像 + 去畸变 + COLMAP 格式")
    logger.info("=" * 60)

    # 检查是否已完成
    meta_file = Path(output_dir) / "conversion_meta.json"
    if skip_existing and meta_file.exists():
        logger.info(f"数据准备已完成（{meta_file} 存在），跳过")
        logger.info("  如需重新生成，使用 --no-skip-existing")
        return True

    # 尝试用 rs30_parser 包
    try:
        from rs30_parser.to_colmap import convert_site_to_colmap
        convert_site_to_colmap(
            site_dir=site_dir,
            output_dir=output_dir,
            cameras=cameras,
            max_frames_per_camera=max_frames,
            write_initial_poses=False,
            undistort=True,
            force_pinhole=False,
        )
        return True
    except ImportError:
        logger.warning("无法导入 rs30_parser，尝试 CLI 方式")

    # Fallback: 通过 CLI 调用
    cmd = [
        sys.executable, "-m", "rs30_parser", "to_colmap",
        site_dir,
        "-o", output_dir,
    ]
    if cameras:
        cmd.extend(["--cameras"] + cameras)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"rs30_parser 失败:\n{result.stderr}")
        return False

    logger.info(result.stdout)
    return True


# ──────────────────────────────────────────────
# Step 2: COLMAP SfM
# ──────────────────────────────────────────────

def step2_colmap_sfm(output_dir, max_features, use_sequential, skip_existing):
    """运行 COLMAP 特征提取 + 匹配 + 稀疏重建"""
    logger.info("=" * 60)
    logger.info("Step 2: COLMAP SfM — 特征提取 + 匹配 + 重建")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    db_path = output_path / "database.db"
    images_dir = output_path / "images"
    sparse_dir = output_path / "sparse"

    # 检查是否已完成
    if skip_existing and db_path.exists() and sparse_dir.exists():
        existing_models = [d for d in sparse_dir.iterdir() if d.is_dir() and (d / "cameras.bin").exists()]
        if existing_models:
            total_images = 0
            for model_dir in existing_models:
                n = count_registered_images(model_dir / "images.bin")
                total_images = max(total_images, n)
            if total_images > 10:
                logger.info(f"COLMAP SfM 已完成（找到 {len(existing_models)} 个模型，最大注册 {total_images} 张图），跳过")
                logger.info("  如需重新运行，使用 --no-skip-existing")
                return True

    if not images_dir.exists():
        logger.error(f"图像目录不存在: {images_dir}")
        return False

    # 统计图像数量
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
    logger.info(f"图像总数: {len(image_files)}")

    # 2.1 特征提取
    logger.info("--- 2.1 特征提取 ---")
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.max_num_features", str(max_features),
        "--SiftExtraction.peak_threshold", "0.005",  # 变电站低纹理场景，降低阈值
        "--SiftExtraction.edge_threshold", "8",
    ]
    logger.info(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"特征提取失败:\n{result.stderr}")
        return False
    logger.info("特征提取完成")

    # 2.2 特征匹配
    logger.info("--- 2.2 特征匹配 ---")
    if use_sequential:
        # 顺序匹配：只匹配相邻帧，速度快
        # 适合视频序列（RS30 2FPS 时间连续）
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.max_num_matches", "65536",
        ]
    else:
        # 暴力匹配：所有图像对，精度高但慢
        # O(n²) 复杂度，1443 张图 = ~100 万对
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.max_num_matches", "65536",
        ]
    logger.info(f"运行: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"特征匹配失败:\n{result.stderr}")
        return False
    logger.info(f"特征匹配完成 ({elapsed:.0f}s)")

    # 2.3 稀疏重建
    logger.info("--- 2.3 稀疏重建 ---")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.init_min_num_inliers", "50",     # 降低初始阈值（变电站低纹理）
        "--Mapper.init_max_error", "4.0",           # 放宽初始误差
        "--Mapper.abs_pose_min_num_inliers", "15",  # 降低绝对姿态最小内点数
        "--Mapper.abs_pose_min_inlier_ratio", "0.15",  # 降低内点比例
        "--Mapper.filter_max_reproj_error", "4.0",  # 放宽重投影误差
        "--Mapper.tri_complete_max_reproj_error", "8.0",  # 放宽三角化误差
    ]
    logger.info(f"运行: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"稀疏重建失败:\n{result.stderr}")
        return False
    logger.info(f"稀疏重建完成 ({elapsed:.0f}s)")

    # 输出重建结果统计
    print_colmap_stats(sparse_dir)
    return True


def count_registered_images(images_bin_path):
    """读取 COLMAP images.bin 的图像数量"""
    try:
        with open(images_bin_path, "rb") as f:
            magic = struct.unpack("<Q", f.read(8))[0]
            num_images = struct.unpack("<Q", f.read(8))[0]
            return num_images
    except Exception:
        return 0


def print_colmap_stats(sparse_dir):
    """打印 COLMAP 重建统计"""
    logger.info("--- 重建结果统计 ---")
    models = []
    for d in sorted(sparse_dir.iterdir()):
        if d.is_dir() and (d / "images.bin").exists():
            n_images = count_registered_images(d / "images.bin")
            pts_size = os.path.getsize(d / "points3D.bin") / 1024
            models.append((d.name, n_images, pts_size))
            logger.info(f"  sparse/{d.name}: 注册图像={n_images}, 点云大小={pts_size:.1f}KB")

    if models:
        best = max(models, key=lambda x: x[1])
        logger.info(f"  最佳模型: sparse/{best[0]} ({best[1]} 张注册图像)")


# ──────────────────────────────────────────────
# Step 3: 选择最佳 COLMAP 模型
# ──────────────────────────────────────────────

def step3_select_best_model(output_dir, skip_existing):
    """选择注册图像最多的 COLMAP 模型，设为 sparse/0"""
    logger.info("=" * 60)
    logger.info("Step 3: 选择最佳 COLMAP 模型")
    logger.info("=" * 60)

    sparse_dir = Path(output_dir) / "sparse"
    if not sparse_dir.exists():
        logger.error(f"sparse 目录不存在: {sparse_dir}")
        return None

    # 找所有模型
    models = []
    for d in sorted(sparse_dir.iterdir()):
        if d.is_dir() and (d / "images.bin").exists():
            n_images = count_registered_images(d / "images.bin")
            pts_size = os.path.getsize(d / "points3D.bin") / 1024
            models.append((d, n_images, pts_size))

    if not models:
        logger.error("未找到任何 COLMAP 模型")
        return None

    # 按注册图像数排序（其次按点云大小）
    models.sort(key=lambda x: (x[1], x[2]), reverse=True)

    best_dir, best_n, best_pts = models[0]
    logger.info(f"最佳模型: {best_dir.name} (注册图像={best_n}, 点云={best_pts:.1f}KB)")

    for d, n, p in models:
        logger.info(f"  {d.name}: 注册图像={n}, 点云={p:.1f}KB")

    # 如果最佳模型不是 sparse/0，备份旧的 sparse/0 并替换
    target_dir = sparse_dir / "0"
    if best_dir != target_dir:
        if target_dir.exists():
            backup_dir = sparse_dir / "0_original"
            if not backup_dir.exists():
                shutil.move(str(target_dir), str(backup_dir))
                logger.info(f"原 sparse/0 已备份为 sparse/0_original")
            else:
                shutil.rmtree(str(target_dir))

        shutil.copytree(str(best_dir), str(target_dir))
        logger.info(f"已将 sparse/{best_dir.name} 复制为 sparse/0")

    # 验证
    final_n = count_registered_images(target_dir / "images.bin")
    logger.info(f"最终 sparse/0: {final_n} 张注册图像")

    if final_n < 10:
        logger.warning(f"⚠️ 注册图像数过少 ({final_n})，重建可能失败！")
        logger.warning("建议检查：")
        logger.warning("  1. 图像去畸变是否正确")
        logger.warning("  2. 尝试 --max-features 16384")
        logger.warning("  3. 尝试 --sequential 匹配模式")
        logger.warning("  4. 减少 --max-frames 采样")

    return str(target_dir)


# ──────────────────────────────────────────────
# Step 4: 运行 SuGaR
# ──────────────────────────────────────────────

def step4_run_sugar(output_dir, sugar_dir, gpu, regularization, high_poly, 
                    eval_split, white_background, refinement_time, skip_existing):
    """运行 SuGaR 全流程训练"""
    logger.info("=" * 60)
    logger.info("Step 4: SuGaR 训练 — 3DGS + 表面正则化 + 网格提取 + 精炼")
    logger.info("=" * 60)

    sugar_path = Path(sugar_dir)
    if not sugar_path.exists():
        logger.error(f"SuGaR 目录不存在: {sugar_dir}")
        return False

    # 检查是否已有结果
    scene_name = Path(output_dir).name
    refined_ply = sugar_path / "output" / "refined_ply" / scene_name
    refined_mesh = sugar_path / "output" / "refined_mesh" / scene_name

    if skip_existing and refined_mesh.exists():
        obj_files = list(refined_mesh.glob("*.obj"))
        if obj_files:
            logger.info(f"SuGaR 结果已存在: {obj_files[0]}")
            logger.info("  如需重新训练，使用 --no-skip-existing")
            return True

    # 构建 SuGaR 命令
    cmd = [
        sys.executable, str(sugar_path / "train_full_pipeline.py"),
        "-s", output_dir,
        "-r", regularization,
        "--gpu", str(gpu),
        "--eval", str(eval_split),
        "--white_background", str(white_background),
        "--export_obj", "True",
        "--export_ply", "True",
    ]

    if high_poly:
        cmd.extend(["--high_poly", "True"])
    else:
        cmd.extend(["--low_poly", "True"])

    if refinement_time:
        cmd.extend(["--refinement_time", refinement_time])

    logger.info(f"运行: {' '.join(cmd)}")
    logger.info("⚠️ SuGaR 训练可能需要 1-3 小时，请耐心等待...")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(sugar_path))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"SuGaR 训练失败 (耗时 {elapsed:.0f}s)")
        return False

    logger.info(f"SuGaR 训练完成！耗时 {elapsed / 60:.0f} 分钟")

    # 输出结果位置
    logger.info("=" * 60)
    logger.info("结果文件:")
    logger.info("=" * 60)

    output_base = sugar_path / "output"
    for subdir in ["coarse_mesh", "refined_mesh", "refined_ply"]:
        d = output_base / subdir / scene_name
        if d.exists():
            files = list(d.iterdir())
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                logger.info(f"  {subdir}/{scene_name}/{f.name} ({size_mb:.1f} MB)")

    return True


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def check_prerequisites():
    """检查必要的工具是否已安装"""
    logger.info("检查环境...")

    # 检查 colmap
    try:
        result = subprocess.run(["colmap", "--version"], capture_output=True, text=True)
        logger.info(f"  COLMAP: {result.stdout.strip() or '已安装'}")
    except FileNotFoundError:
        logger.error("COLMAP 未安装！请运行: sudo apt install colmap")
        return False

    # 检查 CUDA
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True)
        logger.info(f"  GPU: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.warning("nvidia-smi 未找到，GPU 可能不可用")

    # 检查 rs30_parser
    try:
        from rs30_parser import to_colmap  # noqa
        logger.info("  rs30_parser: 已安装（模块模式）")
    except ImportError:
        logger.info("  rs30_parser: 将通过 CLI 调用")

    return True


def print_summary(output_dir, sugar_dir):
    """打印最终总结"""
    logger.info("=" * 60)
    logger.info("全流程完成！")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    sugar_output = Path(sugar_dir) / "output"
    scene_name = output_path.name

    # COLMAP 结果
    sparse_0 = output_path / "sparse" / "0"
    if (sparse_0 / "images.bin").exists():
        n_images = count_registered_images(sparse_0 / "images.bin")
        pts_size = os.path.getsize(sparse_0 / "points3D.bin") / 1024
        logger.info(f"\n📊 COLMAP 重建:")
        logger.info(f"  注册图像: {n_images}")
        logger.info(f"  点云大小: {pts_size:.1f} KB")

    # SuGaR 结果
    for subdir, label in [
        ("coarse_mesh", "粗糙 Mesh"),
        ("refined_mesh", "精细 Mesh"),
        ("refined_ply", "精细高斯"),
    ]:
        d = sugar_output / subdir / scene_name
        if d.exists():
            files = list(d.iterdir())
            logger.info(f"\n🎨 {label}:")
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                logger.info(f"  {f} ({size_mb:.1f} MB)")

    logger.info(f"\n📁 数据目录: {output_dir}")
    logger.info(f"\n下一步:")
    logger.info(f"  查看 Mesh: CloudCompare.CloudCompare {sugar_output}/refined_mesh/{scene_name}/*.obj")
    logger.info(f"  查看高斯: python -m pointcloud_nav visualize --gsplat-interactive")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RS30 三路摄像头 COLMAP + SuGaR 全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（3 路摄像头）
  python run_3cam_pipeline.py \\
      --site-dir /home/wm1/jwang/dataset/baoding/2026-06-13-BD-GLKGQ \\
      --output-dir /home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam \\
      --sugar-dir /home/wm1/jwang/pointcloud-nav/SuGaR

  # 只跑数据准备 + COLMAP（不跑 SuGaR）
  python run_3cam_pipeline.py \\
      --site-dir ... --output-dir ... \\
      --stop-at colmap

  # 降采样测试（每路 100 帧）
  python run_3cam_pipeline.py \\
      --site-dir ... --output-dir ... \\
      --max-frames 100 \\
      --sequential

  # 重新跑某一步
  python run_3cam_pipeline.py \\
      --site-dir ... --output-dir ... \\
      --start-at colmap \\
      --no-skip-existing
        """,
    )

    # 基本路径
    parser.add_argument("--site-dir", required=True,
                        help="RS30 站点数据目录（包含 IMG/ 子目录）")
    parser.add_argument("--output-dir", required=True,
                        help="输出 COLMAP 项目目录")
    parser.add_argument("--sugar-dir", default="",
                        help="SuGaR 代码目录（不指定则跳过 SuGaR 训练）")

    # 摄像头选择
    parser.add_argument("--cameras", nargs="+", default=["Camera1", "Camera2", "Camera3"],
                        help="要使用的摄像头（默认 Camera1 Camera2 Camera3）")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="每路摄像头最大帧数（0=全部，测试用 100）")

    # COLMAP 参数
    parser.add_argument("--max-features", type=int, default=8192,
                        help="SIFT 最大特征数（默认 8192，低纹理场景可设 16384）")
    parser.add_argument("--sequential", action="store_true",
                        help="使用顺序匹配（比暴力匹配快很多，适合视频序列）")

    # SuGaR 参数
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--regularization", default="dn_consistency",
                        choices=["sdf", "density", "dn_consistency"],
                        help="SuGaR 正则化方法（默认 dn_consistency）")
    parser.add_argument("--high-poly", action="store_true", default=True,
                        help="高精度网格（100 万顶点，默认开启）")
    parser.add_argument("--low-poly", action="store_true",
                        help="低精度网格（20 万顶点，更快）")
    parser.add_argument("--refinement-time", default="medium",
                        choices=["short", "medium", "long"],
                        help="精炼时间（short=2k, medium=7k, long=15k 迭代）")
    parser.add_argument("--eval", type=bool, default=False,
                        help="是否做评估分割（默认 False，用全部数据训练）")
    parser.add_argument("--white-background", action="store_true",
                        help="白色背景（室内场景可选）")

    # 流程控制
    parser.add_argument("--start-at", default="prepare",
                        choices=["prepare", "colmap", "select", "sugar"],
                        help="从哪一步开始（默认 prepare）")
    parser.add_argument("--stop-at", default="sugar",
                        choices=["prepare", "colmap", "select", "sugar"],
                        help="到哪一步停止（默认 sugar）")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="不跳过已完成的步骤")

    args = parser.parse_args()

    # 参数修正
    if args.low_poly:
        args.high_poly = False

    skip_existing = not args.no_skip_existing

    logger.info("🚀 RS30 三路摄像头全流程")
    logger.info(f"  站点: {args.site_dir}")
    logger.info(f"  输出: {args.output_dir}")
    logger.info(f"  摄像头: {args.cameras}")
    logger.info(f"  最大帧数: {args.max_frames or '全部'}")
    logger.info(f"  匹配模式: {'顺序' if args.sequential else '暴力'}")

    # 检查环境
    if not check_prerequisites():
        sys.exit(1)

    # 定义步骤
    steps = [
        ("prepare", lambda: step1_prepare_data(
            args.site_dir, args.output_dir, args.cameras, args.max_frames, skip_existing)),
        ("colmap", lambda: step2_colmap_sfm(
            args.output_dir, args.max_features, args.sequential, skip_existing)),
        ("select", lambda: step3_select_best_model(
            args.output_dir, skip_existing)),
        ("sugar", lambda: step4_run_sugar(
            args.output_dir, args.sugar_dir, args.gpu, args.regularization,
            args.high_poly, args.eval, args.white_background,
            args.refinement_time, skip_existing)),
    ]

    step_names = [s[0] for s in steps]
    start_idx = step_names.index(args.start_at)
    stop_idx = step_names.index(args.stop_at)

    if start_idx > stop_idx:
        logger.error(f"--start-at ({args.start_at}) 不能在 --stop-at ({args.stop_at}) 之后")
        sys.exit(1)

    # 执行步骤
    best_model = None
    for i in range(start_idx, stop_idx + 1):
        name, fn = steps[i]
        logger.info(f"\n▶ 执行步骤 {i+1}/{len(steps)}: {name}")

        result = fn()

        # select 步骤返回模型路径
        if name == "select":
            if result is None:
                logger.error("未找到有效 COLMAP 模型，无法继续")
                logger.info("建议：")
                logger.info("  1. 检查图像去畸变是否正确")
                logger.info("  2. 尝试 --max-features 16384")
                logger.info("  3. 尝试 --sequential 匹配模式")
                logger.info("  4. 减少 --max-frames 100 快速测试")
                sys.exit(1)
            best_model = result

        # 其他步骤返回 bool
        if name != "select" and not result:
            logger.error(f"步骤 {name} 失败")
            sys.exit(1)

    # 打印总结
    print_summary(args.output_dir, args.sugar_dir)


if __name__ == "__main__":
    main()
