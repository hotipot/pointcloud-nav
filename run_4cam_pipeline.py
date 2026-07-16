#!/usr/bin/env python3
"""4路 Camera COLMAP SfM Pipeline（含 Camera4 全景）

整合 Camera1/2/3 (PINHOLE) + Camera4 (EQUIRECTANGULAR)，
重新运行 COLMAP 特征提取 + 匹配 + SfM。

关键点：
- Camera1/2/3: 已去畸变，PINHOLE 模型
- Camera4: 全景等距柱状投影，EQUIRECTANGULAR 模型
- COLMAP 支持 EQUIRECTANGULAR，但特征提取时需要正确设置相机模型
- 4 路 camera 混合可以提供跨视角链接，有望解决 SfM 碎片化问题

COLMAP EQUIRECTANGULAR 模型说明：
- model = "EQUIRECTANGULAR"
- params = [] (空，不需要焦距参数)
- COLMAP 根据图像宽高自动处理投影
- 标准等距柱状投影宽高比 = 2:1，但裁剪过的也可以用
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# COLMAP 环境变量（修复已知问题）
COLMAP_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    # 移除 conda 的 LD_LIBRARY_PATH 避免 SQLite 冲突
}


def get_colmap_cmd():
    """获取 colmap 命令路径"""
    # 优先使用 conda-forge 版（有 CUDA）
    conda_colmap = "/home/wm1/anaconda3/envs/pointcloud/bin/colmap"
    if os.path.exists(conda_colmap):
        return conda_colmap
    return "colmap"


def prepare_image_list(
    output_dir: str,
    cam4_dir: str = None,
):
    """准备图像列表，含 Camera1/2/3 + Camera4

    COLMAP 对混合相机模型需要用 image_list 或正确的 ImageReader 参数。
    最可靠的方式是用 --ImageReader.camera_model 为每张图指定模型。
    
    但 COLMAP 的 ImageReader 只支持全局 camera_model 参数，
    所以对于混合模型，需要分步处理：
    1. 先提取 Camera1/2/3 (PINHOLE)
    2. 再提取 Camera4 (EQUIRECTANGULAR)
    3. 合并数据库
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    
    # 统计图像
    cam123_images = sorted(images_dir.glob("Camera[123]_*.jpg"))
    cam4_images = []
    
    if cam4_dir:
        cam4_path = Path(cam4_dir)
        # 复制 Camera4 图像到 images/ 目录
        cam4_src_images = sorted(cam4_path.glob("Camera4_*.jpg"))
        logger.info(f"Camera4: 发现 {len(cam4_src_images)} 张全景图")
        
        import shutil
        for img in cam4_src_images:
            dst = images_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            cam4_images.append(dst)
    
    logger.info(f"图像统计: Camera1/2/3={len(cam123_images)}, Camera4={len(cam4_images)}, 总计={len(cam123_images)+len(cam4_images)}")
    return len(cam123_images), len(cam4_images)


def run_colmap_step1_feature_extraction_cam123(
    database_path: str,
    image_path: str,
    colmap_cmd: str,
):
    """Step 1a: 提取 Camera1/2/3 的特征（PINHOLE 模型）"""
    logger.info("=" * 60)
    logger.info("Step 1a: 特征提取 - Camera1/2/3 (PINHOLE, GPU)")
    logger.info("=" * 60)
    
    # 创建 Camera1/2/3 专用图像列表
    image_path_obj = Path(image_path)
    cam123_list = image_path_obj.parent / "cam123_image_list.txt"
    with open(cam123_list, "w") as f:
        for img in sorted(image_path_obj.glob("Camera[123]_*.jpg")):
            f.write(f"{img.name}\n")
    
    cmd = [
        colmap_cmd, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", str(cam123_list),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera_per_folder", "1",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "8192",
    ]
    
    env = {**os.environ, **COLMAP_ENV}
    # 移除 conda 的 LD_LIBRARY_PATH
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CONDA_PREFIX", None)
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"特征提取 (cam123) 失败:\n{result.stderr}")
        # 尝试旧版参数
        logger.info("尝试 COLMAP 3.x 兼容参数...")
        cmd_compat = [
            colmap_cmd, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--image_list_path", str(cam123_list),
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera_per_folder", "1",
            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.max_image_size", "3200",
        ]
        result = subprocess.run(cmd_compat, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"兼容模式也失败:\n{result.stderr}")
            return False
    
    logger.info("Camera1/2/3 特征提取完成 ✅")
    return True


def run_colmap_step1b_feature_extraction_cam4(
    database_path: str,
    image_path: str,
    colmap_cmd: str,
):
    """Step 1b: 提取 Camera4 的特征（EQUIRECTANGULAR 模型）"""
    logger.info("=" * 60)
    logger.info("Step 1b: 特征提取 - Camera4 (EQUIRECTANGULAR, GPU)")
    logger.info("=" * 60)
    
    # 创建 Camera4 专用图像列表
    image_path_obj = Path(image_path)
    cam4_list = image_path_obj.parent / "cam4_image_list.txt"
    with open(cam4_list, "w") as f:
        for img in sorted(image_path_obj.glob("Camera4_*.jpg")):
            f.write(f"{img.name}\n")
    
    cmd = [
        colmap_cmd, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", str(cam4_list),
        "--ImageReader.camera_model", "EQUIRECTANGULAR",
        "--ImageReader.single_camera_per_folder", "1",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "8192",
    ]
    
    env = {**os.environ, **COLMAP_ENV}
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CONDA_PREFIX", None)
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"特征提取 (cam4) 失败:\n{result.stderr}")
        return False
    
    logger.info("Camera4 特征提取完成 ✅")
    return True


def run_colmap_step2_matching(
    database_path: str,
    colmap_cmd: str,
    matching_mode: str = "exhaustive",
):
    """Step 2: 特征匹配"""
    logger.info("=" * 60)
    logger.info(f"Step 2: 特征匹配 ({matching_mode})")
    logger.info("=" * 60)
    
    env = {**os.environ, **COLMAP_ENV}
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CONDA_PREFIX", None)
    
    if matching_mode == "exhaustive":
        cmd = [
            colmap_cmd, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.max_num_matches", "65536",
        ]
        # COLMAP 4.1 用 FeatureMatching.max_num_matches
        # COLMAP 3.x 用 SiftMatching.max_num_matches
    elif matching_mode == "sequential":
        cmd = [
            colmap_cmd, "sequential_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.max_num_matches", "65536",
        ]
    elif matching_mode == "vocab_tree":
        cmd = [
            colmap_cmd, "vocab_tree_matcher",
            "--database_path", database_path,
            "--VocabTreeMatching.match_id", "0",
            "--FeatureMatching.use_gpu", "1",
        ]
    else:
        logger.error(f"未知匹配模式: {matching_mode}")
        return False
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"特征匹配失败:\n{result.stderr}")
        # 重试旧参数名
        if "max_num_matches" in str(cmd):
            logger.info("尝试 COLMAP 3.x 兼容参数名...")
            cmd_compat = [c.replace("FeatureMatching.max_num_matches", "SiftMatching.max_num_matches") for c in cmd]
            result = subprocess.run(cmd_compat, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"兼容模式也失败:\n{result.stderr}")
                return False
    
    logger.info("特征匹配完成 ✅")
    return True


def run_colmap_step3_mapper(
    database_path: str,
    image_path: str,
    sparse_path: str,
    colmap_cmd: str,
):
    """Step 3: 稀疏重建 (SfM)"""
    logger.info("=" * 60)
    logger.info("Step 3: 稀疏重建 (SfM)")
    logger.info("=" * 60)
    
    env = {**os.environ, **COLMAP_ENV}
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CONDA_PREFIX", None)
    
    sparse_dir = Path(sparse_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # SfM 参数针对变电站低纹理场景调优
    cmd = [
        colmap_cmd, "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", str(sparse_dir),
        # 降低初始化门槛
        "--init_min_num_inliers", "50",
        "--init_max_error", "8",
        # 注册参数
        "--ba_refine_extra_params", "1",
        "--tri_max_angle", "16",
        # 增加迭代次数
        "--ba_max_num_iterations", "50",
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Mapper 失败:\n{result.stderr[-2000:]}")
        return False
    
    logger.info("稀疏重建完成 ✅")
    return True


def analyze_results(sparse_path: str):
    """分析 SfM 结果"""
    import struct
    
    sparse_dir = Path(sparse_path)
    results = []
    
    for sub_dir in sorted(sparse_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        if not (sub_dir / "cameras.bin").exists():
            continue
        
        # 读取 cameras.bin
        with open(sub_dir / "cameras.bin", "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            num_cameras = struct.unpack("<Q", f.read(8))[0]
        
        # 读取 images.bin
        with open(sub_dir / "images.bin", "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            num_images = struct.unpack("<Q", f.read(8))[0]
        
        # 读取 points3D.bin
        with open(sub_dir / "points3D.bin", "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            num_points = struct.unpack("<Q", f.read(8))[0]
        
        results.append({
            "model": sub_dir.name,
            "cameras": num_cameras,
            "images": num_images,
            "points": num_points,
        })
    
    # 按注册图像数排序
    results.sort(key=lambda x: -x["images"])
    
    logger.info("=" * 60)
    logger.info("SfM 结果分析")
    logger.info("=" * 60)
    logger.info(f"总子模型数: {len(results)}")
    
    if results:
        logger.info("\nTop 10 子模型 (按注册图像数):")
        logger.info(f"  {'Model':>8} {'Cams':>6} {'Images':>8} {'Points':>10}")
        for r in results[:10]:
            logger.info(f"  {r['model']:>8} {r['cameras']:>6} {r['images']:>8} {r['points']:>10}")
        
        best = results[0]
        logger.info(f"\n最佳模型: {best['model']} — {best['images']} 张注册图像, {best['points']} 个3D点")
        
        if best["images"] >= 100:
            logger.info("✅ 注册图像数较好，可以继续 SuGaR 训练")
        elif best["images"] >= 50:
            logger.info("⚠️ 注册图像数一般，SuGaR 效果可能有限")
        else:
            logger.info("❌ 注册图像数太少，需要调整参数或方案")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="4路 Camera COLMAP SfM Pipeline")
    parser.add_argument("--output-dir", type=str,
                        default="/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_4cam",
                        help="输出目录")
    parser.add_argument("--cam4-dir", type=str,
                        default="/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam/Camera4/images",
                        help="Camera4 全景图目录")
    parser.add_argument("--matching", type=str, default="exhaustive",
                        choices=["exhaustive", "sequential", "vocab_tree"],
                        help="特征匹配模式")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="跳过特征提取（如果已完成）")
    parser.add_argument("--skip-matching", action="store_true",
                        help="跳过特征匹配（如果已完成）")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")
    args = parser.parse_args()
    
    # 日志配置
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers,
    )
    
    colmap_cmd = get_colmap_cmd()
    logger.info(f"COLMAP: {colmap_cmd}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 0: 准备图像目录
    logger.info("=" * 60)
    logger.info("Step 0: 准备图像")
    logger.info("=" * 60)
    
    # 复制 Camera1/2/3 图像
    src_images = Path("/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam/images")
    dst_images = output_path / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    
    import shutil
    cam123_count = 0
    for img in sorted(src_images.glob("Camera[123]_*.jpg")):
        dst = dst_images / img.name
        if not dst.exists():
            shutil.copy2(str(img), str(dst))
        cam123_count += 1
    
    # 复制 Camera4 图像
    cam4_count = 0
    if args.cam4_dir:
        cam4_src = Path(args.cam4_dir)
        for img in sorted(cam4_src.glob("Camera4_*.jpg")):
            dst = dst_images / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            cam4_count += 1
    
    logger.info(f"图像准备完成: Camera1/2/3={cam123_count}, Camera4={cam4_count}, 总计={cam123_count+cam4_count}")
    
    database_path = str(output_path / "database.db")
    image_path = str(dst_images)
    sparse_path = str(output_path / "sparse")
    
    # Step 1: 特征提取
    if not args.skip_extraction:
        ok = run_colmap_step1_feature_extraction_cam123(
            database_path, image_path, colmap_cmd
        )
        if not ok:
            logger.error("Camera1/2/3 特征提取失败，退出")
            sys.exit(1)
        
        ok = run_colmap_step1b_feature_extraction_cam4(
            database_path, image_path, colmap_cmd
        )
        if not ok:
            logger.error("Camera4 特征提取失败，退出")
            sys.exit(1)
    
    # Step 2: 特征匹配
    if not args.skip_matching:
        ok = run_colmap_step2_matching(
            database_path, colmap_cmd, args.matching
        )
        if not ok:
            logger.error("特征匹配失败，退出")
            sys.exit(1)
    
    # Step 3: 稀疏重建
    ok = run_colmap_step3_mapper(
        database_path, image_path, sparse_path, colmap_cmd
    )
    if not ok:
        logger.error("稀疏重建失败")
        sys.exit(1)
    
    # Step 4: 分析结果
    results = analyze_results(sparse_path)
    
    # 保存元数据
    meta = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "4cam_colmap",
        "cam123_count": cam123_count,
        "cam4_count": cam4_count,
        "total_images": cam123_count + cam4_count,
        "matching_mode": args.matching,
        "results": results,
    }
    with open(output_path / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Pipeline 完成! 输出目录: {output_path}")


if __name__ == "__main__":
    main()
