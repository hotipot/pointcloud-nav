#!/usr/bin/env python3
"""4路 Camera COLMAP SfM Pipeline v2

修复 v1 的问题：
1. 相机内参：Camera1/2/3 各有不同内参，不能共享一个相机
   → 使用 image_list + 手动设置相机参数
2. 匹配策略：exhaustive 对 1680 张图太慢（~100 小时）
   → 改用 sequential matcher + spatial matcher

策略：
- 分 4 次运行 feature_extractor，每次指定不同的相机模型和内参
- 使用 sequential matcher（基于图像顺序匹配相邻帧）
- 可选 vocab_tree matcher（基于视觉相似性匹配）

COLMAP 4.1 相机模型参数：
- PINHOLE: [fx, fy, cx, cy]
- EQUIRECTANGULAR: [] (空)
"""

import argparse
import json
import logging
import os
import sqlite3
import struct
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

COLMAP_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
}

# Camera 内参（来自 RS30 .CP 文件，去畸变后的 PINHOLE 参数）
CAMERA_PARAMS = {
    "Camera1": {
        "model": "PINHOLE",
        "width": 1940,
        "height": 2588,
        "params": [1283.894575892754, 1761.8402718451073, 1129.5899358037268, 565.6121936596221],
    },
    "Camera2": {
        "model": "PINHOLE",
        "width": 1940,
        "height": 2588,
        "params": [1258.0577177962025, 1762.6856863120317, 1184.5645712934258, 433.5451397456157],
    },
    "Camera3": {
        "model": "PINHOLE",
        "width": 1940,
        "height": 2588,
        "params": [1264.2194133537223, 1796.0672322028327, 1136.3123902536774, 561.1533012280892],
    },
    "Camera4": {
        "model": "EQUIRECTANGULAR",
        "width": 1920,
        "height": 1080,
        "params": [],  # EQUIRECTANGULAR 不需要内参
    },
}


def get_colmap_cmd():
    conda_colmap = "/home/wm1/anaconda3/envs/pointcloud/bin/colmap"
    if os.path.exists(conda_colmap):
        return conda_colmap
    return "colmap"


def get_env():
    env = {**os.environ, **COLMAP_ENV}
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CONDA_PREFIX", None)
    return env


def run_cmd(cmd, label="", timeout=None):
    """运行命令并检查返回值"""
    logger.info(f"[{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=get_env(), capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"[{label}] FAILED:\n{result.stderr[-1000:]}")
        return False
    if result.stdout:
        for line in result.stdout.split("\n")[-5:]:
            if line.strip():
                logger.info(f"[{label}] {line}")
    return True


def step0_prepare_images(output_dir, cam4_dir):
    """准备图像目录"""
    logger.info("=" * 60)
    logger.info("Step 0: 准备图像")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制 Camera1/2/3
    src_images = Path("/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam/images")
    cam123_count = 0
    for img in sorted(src_images.glob("Camera[123]_*.jpg")):
        dst = images_dir / img.name
        if not dst.exists():
            shutil.copy2(str(img), str(dst))
        cam123_count += 1
    
    # 复制 Camera4
    cam4_count = 0
    if cam4_dir:
        cam4_src = Path(cam4_dir)
        for img in sorted(cam4_src.glob("Camera4_*.jpg")):
            dst = images_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            cam4_count += 1
    
    logger.info(f"图像: Camera1/2/3={cam123_count}, Camera4={cam4_count}, 总计={cam123_count+cam4_count}")
    return cam123_count, cam4_count


def step1_feature_extraction(database_path, image_path, colmap_cmd):
    """特征提取：分 4 次运行，每次一个相机
    
    关键：使用 --ImageReader.single_camera 0（不共享相机）
    然后手动修改数据库中的相机参数
    """
    logger.info("=" * 60)
    logger.info("Step 1: 特征提取")
    logger.info("=" * 60)
    
    images_dir = Path(image_path)
    
    # 先一次性提取所有图像的特征
    # 使用 PINHOLE 模型（COLMAP 会自动为每张图创建相机）
    # 然后手动修改数据库中的相机参数
    
    # 方案：分两步
    # Step 1a: 提取 Camera1/2/3 (PINHOLE)
    # Step 1b: 提取 Camera4 (EQUIRECTANGULAR)
    
    # Camera1/2/3 图像列表
    cam123_list = images_dir.parent / "cam123_image_list.txt"
    with open(cam123_list, "w") as f:
        for img in sorted(images_dir.glob("Camera[123]_*.jpg")):
            f.write(f"{img.name}\n")
    
    # Camera4 图像列表
    cam4_list = images_dir.parent / "cam4_image_list.txt"
    with open(cam4_list, "w") as f:
        for img in sorted(images_dir.glob("Camera4_*.jpg")):
            f.write(f"{img.name}\n")
    
    # Step 1a: Camera1/2/3 (PINHOLE, single_camera=0 → 每张图独立相机)
    # 注意：single_camera_per_folder=1 在 flat 目录下会把所有图归为一个相机
    # 我们需要 single_camera=0，然后手动合并同型号的相机
    ok = run_cmd([
        colmap_cmd, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", str(cam123_list),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "0",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "8192",
    ], "cam123-extract")
    if not ok:
        return False
    
    # Step 1b: Camera4 (EQUIRECTANGULAR)
    ok = run_cmd([
        colmap_cmd, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", str(cam4_list),
        "--ImageReader.camera_model", "EQUIRECTANGULAR",
        "--ImageReader.single_camera", "0",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", "3200",
        "--SiftExtraction.max_num_features", "8192",
    ], "cam4-extract")
    if not ok:
        return False
    
    # Step 1c: 修正数据库中的相机参数
    # COLMAP 为每张图创建了独立相机，我们需要：
    # 1. 合并同型号的相机（Camera1/2/3 共享参数但各自独立，Camera4 独立）
    # 2. 设置正确的内参
    fix_camera_params(database_path)
    
    return True


def fix_camera_params(database_path):
    """修正数据库中的相机参数
    
    COLMAP feature_extractor 用 single_camera=0 时，每张图一个相机。
    我们需要：
    1. 为每个 camera_name 创建一个独立的相机记录（即使型号/尺寸相同，内参可能不同）
    2. 将图像的 camera_id 指向正确的相机
    3. 删除多余的相机记录
    """
    logger.info("=" * 60)
    logger.info("Step 1c: 修正相机参数")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    
    # 查看当前状态
    n_cameras = cur.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    n_images = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    logger.info(f"修正前: {n_cameras} 个相机, {n_images} 张图")
    
    # 为每个 camera_name 创建独立的相机记录
    # 先删除所有现有相机
    cur.execute("DELETE FROM cameras")
    
    cam_name_to_cam_id = {}
    for cam_name, params in CAMERA_PARAMS.items():
        # 插入新相机
        params_blob = struct.pack(f"<{len(params['params'])}d", *params["params"]) if params["params"] else b""
        cur.execute(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, 1)",
            (get_model_id(params["model"]), params["width"], params["height"], params_blob)
        )
        cam_id = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
        cam_name_to_cam_id[cam_name] = cam_id
        logger.info(f"  {cam_name} → camera_id={cam_id}, model={params['model']}, size={params['width']}x{params['height']}")
    
    # 更新每张图的 camera_id
    images = cur.execute("SELECT image_id, name FROM images").fetchall()
    updated = 0
    for img_id, img_name in images:
        # 从文件名推断相机
        if img_name.startswith("Camera1_"):
            new_cam_id = cam_name_to_cam_id["Camera1"]
        elif img_name.startswith("Camera2_"):
            new_cam_id = cam_name_to_cam_id["Camera2"]
        elif img_name.startswith("Camera3_"):
            new_cam_id = cam_name_to_cam_id["Camera3"]
        elif img_name.startswith("Camera4_"):
            new_cam_id = cam_name_to_cam_id["Camera4"]
        else:
            logger.warning(f"  未知相机: {img_name}")
            continue
        
        cur.execute("UPDATE images SET camera_id=? WHERE image_id=?", (new_cam_id, img_id))
        updated += 1
    
    conn.commit()
    
    # 验证
    n_cameras_after = cur.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    logger.info(f"修正后: {n_cameras_after} 个相机, {updated} 张图已更新")
    
    # 打印最终相机列表
    cams = cur.execute("SELECT camera_id, model, width, height, params FROM cameras").fetchall()
    model_names = {0:'INVALID',1:'PINHOLE',2:'SIMPLE_PINHOLE',3:'SIMPLE_RADIAL',4:'RADIAL',5:'OPENCV',6:'OPENCV_FISHEYE',7:'FULL_OPENCV',8:'FOV',9:'THIN_PRISM_FISHEYE',12:'RAD_TAN_THIN_PRISM',17:'EQUIRECTANGULAR'}
    for c in cams:
        mname = model_names.get(c[1], f'UNKNOWN({c[1]})')
        params_str = ""
        if c[4]:
            n_params = len(c[4]) // 8
            params_vals = struct.unpack(f"<{n_params}d", c[4])
            params_str = f", params={[f'{v:.2f}' for v in params_vals]}"
        logger.info(f"  cam_id={c[0]}: {mname} {c[2]}x{c[3]}{params_str}")
    
    conn.close()
    return True


def get_model_id(model_name):
    """COLMAP 相机模型 ID"""
    model_ids = {
        "INVALID": 0, "PINHOLE": 1, "SIMPLE_PINHOLE": 2,
        "SIMPLE_RADIAL": 3, "RADIAL": 4, "OPENCV": 5,
        "OPENCV_FISHEYE": 6, "FULL_OPENCV": 7, "FOV": 8,
        "THIN_PRISM_FISHEYE": 9, "RAD_TAN_THIN_PRISM": 12,
        "EQUIRECTANGULAR": 17,
    }
    return model_ids.get(model_name, 0)


def step2_matching(database_path, colmap_cmd, mode="sequential"):
    """特征匹配"""
    logger.info("=" * 60)
    logger.info(f"Step 2: 特征匹配 ({mode})")
    logger.info("=" * 60)
    
    if mode == "sequential":
        # Sequential matcher: 只匹配相邻帧（按图像名排序）
        # 适合视频序列采集的数据
        ok = run_cmd([
            colmap_cmd, "sequential_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.max_num_matches", "65536",
            # Sequential 参数
            "--SequentialMatching.overlap", "10",  # 匹配前后 10 帧
            "--SequentialMatching.loop_detection", "1",  # 启用回环检测
            "--SequentialMatching.quadratic_overlap", "1",  # 二次采样（1,2,4,8...）
        ], "sequential-match")
        return ok
    
    elif mode == "spatial":
        # Spatial matcher: 基于空间位置匹配（需要先有位姿）
        # 我们没有位姿，所以不适用
        logger.error("Spatial matcher 需要先有位姿，不适用")
        return False
    
    elif mode == "vocab_tree":
        # VocabTree matcher: 基于视觉词袋匹配
        # 需要预训练的词袋树
        vocab_tree_path = "/home/wm1/jwang/pointcloud-nav/vocab_tree_flickr100K_words256K.bin"
        if not os.path.exists(vocab_tree_path):
            logger.warning(f"VocabTree 不存在: {vocab_tree_path}")
            logger.info("尝试下载...")
            # 下载 COLMAP 官方词袋树
            import urllib.request
            url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
            try:
                urllib.request.urlretrieve(url, vocab_tree_path)
                logger.info(f"下载完成: {vocab_tree_path}")
            except Exception as e:
                logger.error(f"下载失败: {e}")
                return False
        
        ok = run_cmd([
            colmap_cmd, "vocab_tree_matcher",
            "--database_path", database_path,
            "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.max_num_matches", "65536",
            "--VocabTreeMatching.num_images", "20",  # 每张图匹配 top-20 最相似图
        ], "vocab-tree-match")
        return ok
    
    elif mode == "exhaustive":
        # Exhaustive: 全量匹配（非常慢，1680 张图需要 ~100 小时）
        logger.warning("Exhaustive matching 对 1680 张图需要 ~100 小时！")
        ok = run_cmd([
            colmap_cmd, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.max_num_matches", "65536",
        ], "exhaustive-match")
        return ok
    
    else:
        logger.error(f"未知匹配模式: {mode}")
        return False


def step3_mapper(database_path, image_path, sparse_path, colmap_cmd):
    """稀疏重建"""
    logger.info("=" * 60)
    logger.info("Step 3: 稀疏重建 (SfM)")
    logger.info("=" * 60)
    
    sparse_dir = Path(sparse_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    ok = run_cmd([
        colmap_cmd, "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", str(sparse_dir),
        "--init_min_num_inliers", "50",
        "--init_max_error", "8",
        "--ba_refine_extra_params", "1",
        "--tri_max_angle", "16",
        "--ba_max_num_iterations", "50",
    ], "mapper")
    return ok


def analyze_results(sparse_path):
    """分析 SfM 结果"""
    sparse_dir = Path(sparse_path)
    results = []
    
    for sub_dir in sorted(sparse_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        cameras_bin = sub_dir / "cameras.bin"
        images_bin = sub_dir / "images.bin"
        points_bin = sub_dir / "points3D.bin"
        
        if not cameras_bin.exists() or not images_bin.exists():
            continue
        
        try:
            with open(images_bin, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                num_images = struct.unpack("<Q", f.read(8))[0]
            
            with open(points_bin, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                num_points = struct.unpack("<Q", f.read(8))[0]
            
            with open(cameras_bin, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                num_cameras = struct.unpack("<Q", f.read(8))[0]
            
            results.append({
                "model": sub_dir.name,
                "cameras": num_cameras,
                "images": num_images,
                "points": num_points,
            })
        except Exception as e:
            logger.warning(f"读取 {sub_dir.name} 失败: {e}")
    
    results.sort(key=lambda x: -x["images"])
    
    logger.info("=" * 60)
    logger.info("SfM 结果分析")
    logger.info("=" * 60)
    logger.info(f"总子模型数: {len(results)}")
    
    if results:
        logger.info(f"\n  {'Model':>8} {'Cams':>6} {'Images':>8} {'Points':>10}")
        for r in results[:15]:
            logger.info(f"  {r['model']:>8} {r['cameras']:>6} {r['images']:>8} {r['points']:>10}")
        
        best = results[0]
        logger.info(f"\n最佳模型: {best['model']} — {best['images']} 张注册图像, {best['points']} 个3D点")
        
        if best["images"] >= 500:
            logger.info("✅ 注册图像数很好！可以继续 SuGaR 训练")
        elif best["images"] >= 100:
            logger.info("⚠️ 注册图像数一般，SuGaR 效果可能有限")
        else:
            logger.info("❌ 注册图像数太少，需要调整参数或方案")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="4路 Camera COLMAP SfM Pipeline v2")
    parser.add_argument("--output-dir", type=str,
                        default="/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_4cam_v2",
                        help="输出目录")
    parser.add_argument("--cam4-dir", type=str,
                        default="/home/wm1/jwang/pointcloud-nav/output/BD-GLKGQ_3cam/Camera4/images",
                        help="Camera4 全景图目录")
    parser.add_argument("--matching", type=str, default="sequential",
                        choices=["exhaustive", "sequential", "vocab_tree"],
                        help="特征匹配模式（推荐 sequential）")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    parser.add_argument("--skip-fix-cameras", action="store_true",
                        help="跳过相机参数修正（如果已修正）")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    
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
    logger.info(f"匹配模式: {args.matching}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 0
    cam123_count, cam4_count = step0_prepare_images(args.output_dir, args.cam4_dir)
    
    database_path = str(output_path / "database.db")
    image_path = str(output_path / "images")
    sparse_path = str(output_path / "sparse")
    
    # Step 1: 特征提取
    if not args.skip_extraction:
        ok = step1_feature_extraction(database_path, image_path, colmap_cmd)
        if not ok:
            logger.error("特征提取失败")
            sys.exit(1)
    elif not args.skip_fix_cameras:
        # 只修正相机参数
        fix_camera_params(database_path)
    
    # Step 2: 特征匹配
    if not args.skip_matching:
        ok = step2_matching(database_path, colmap_cmd, args.matching)
        if not ok:
            logger.error("特征匹配失败")
            sys.exit(1)
    
    # Step 3: 稀疏重建
    ok = step3_mapper(database_path, image_path, sparse_path, colmap_cmd)
    if not ok:
        logger.error("稀疏重建失败")
        sys.exit(1)
    
    # Step 4: 分析结果
    results = analyze_results(sparse_path)
    
    # 保存元数据
    meta = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "4cam_colmap_v2",
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
