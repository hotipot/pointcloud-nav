# SuGaR Ubuntu 部署指南

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 20.04 / 22.04 | 推荐 22.04 |
| GPU | NVIDIA RTX 3090 / 4090 / A100 | 最低 8GB VRAM，推荐 24GB |
| CUDA | 11.8 | 必须匹配 PyTorch |
| Python | 3.9 | Conda 管理 |
| Conda | Miniconda / Anaconda | 环境管理 |

## 1. 系统依赖安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git cmake wget curl

# 安装 CUDA 11.8（如果没有）
# 方法 A: 通过 NVIDIA 官方 runfile
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent

# 方法 B: 通过 apt（Ubuntu 22.04）
sudo apt install -y nvidia-cuda-toolkit-11-8

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证 CUDA
nvcc --version
nvidia-smi
```

## 2. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init bash
source ~/.bashrc
```

## 3. 克隆 SuGaR 代码

```bash
cd ~/projects  # 或你的工作目录
git clone https://github.com/Anttwo/SuGaR.git --recursive
cd SuGaR
```

> ⚠️ 如果 git clone 超时，可以下载 zip：
> ```bash
> wget https://github.com/Anttwo/SuGaR/archive/refs/heads/main.zip
> unzip main.zip && mv SuGaR-main SuGaR && cd SuGaR
> ```
> 注意：zip 方式不包含 gaussian_splatting 子模块，需要手动获取：
> ```bash
> cd gaussian_splatting
> git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
> git clone https://github.com/graphdeco-inria/simple-knn submodules/simple-knn
> ```

## 4. 创建 Conda 环境

### 方法 A：自动安装（推荐）

```bash
cd ~/projects/SuGaR
python install.py
conda activate sugar
```

### 方法 B：手动安装

```bash
cd ~/projects/SuGaR

# 创建环境
conda env create -f environment.yml
conda activate sugar

# 安装 PyTorch（如果 environment.yml 安装失败）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d==0.7.4 -c pytorch3d

# 安装 Gaussian Splatting 子模块
cd gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .
cd ../simple-knn/
pip install -e .
cd ../../../

# 安装其他依赖
pip install open3d pymcubes pyquaternion

# （可选）安装 Nvdiffrast 加速纹理提取
cd ~/projects
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ../SuGaR
```

## 5. 安装 COLMAP

COLMAP 是 SfM 前置步骤，必须安装：

```bash
# Ubuntu 22.04（推荐）
sudo apt install -y colmap

# 或从源码编译（获取最新版）
sudo apt install -y \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
    libboost-system-dev libboost-test-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libgoogle-glog-dev libgflags-dev \
    libglew-dev libqt5openglwidgets5-dev qtbase5-dev \
    libmetis-dev libsqlite3-dev libcgal-dev libceres-dev

git clone https://github.com/colmap/colmap.git
cd colmap && mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)
sudo make install
```

验证：
```bash
colmap --version  # 应输出 3.8+ 或更高
```

## 6. 准备 RS30 数据

### 6.1 将数据传输到 Ubuntu 机器

```bash
# 在 Mac 上打包
cd /path/to/pointcloud_data/baoding
tar czf 2026-06-13-BD-GLKGQ.tar.gz 2026-06-13-BD-GLKGQ/

# 通过 scp 传输
scp 2026-06-13-BD-GLKGQ.tar.gz user@ubuntu-host:~/data/

# 在 Ubuntu 上解压
cd ~/data && tar xzf 2026-06-13-BD-GLKGQ.tar.gz
```

### 6.2 使用 rs30_parser 转换数据

```bash
# 安装 rs30_parser
cd ~/projects
git clone https://github.com/hotipot/pointcloud-nav.git
cd pointcloud-nav

# 安装依赖
pip install opencv-python-headless numpy pillow

# 运行转换（一键：提取图像 + 去畸变 + 生成 COLMAP 格式）
python -m rs30_parser to_colmap \
    ~/data/2026-06-13-BD-GLKGQ \
    -o ~/data/2026-06-13-BD-GLKGQ_colmap \
    --cameras Camera1 Camera2 Camera3

# 转换完成后，输出目录结构：
# 2026-06-13-BD-GLKGQ_colmap/
# ├── images/              # 去畸变后的图像（flat，SuGaR 直接用）
# │   ├── Camera1_*.jpg
# │   ├── Camera2_*.jpg
# │   └── Camera3_*.jpg
# ├── Camera1/raw/         # 原始图像备份
# ├── Camera2/raw/
# ├── Camera3/raw/
# ├── sparse/0/
# │   ├── cameras.bin      # PINHOLE 模型（去畸变后）
# │   ├── images.bin       # 空（让 COLMAP 算）
# │   └── points3D.bin     # 空
# ├── cameras.json
# └── conversion_meta.json
```

## 7. 运行 COLMAP SfM

SuGaR 需要 COLMAP 的 SfM 结果（相机位姿 + 稀疏点云）：

```bash
DATA_DIR=~/data/2026-06-13-BD-GLKGQ_colmap

# Step 1: 特征提取
colmap feature_extractor \
    --database_path $DATA_DIR/database.db \
    --image_path $DATA_DIR/images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera_per_folder 1 \
    --SiftExtraction.max_num_features 8192

# Step 2: 特征匹配
colmap exhaustive_matcher \
    --database_path $DATA_DIR/database.db \
    --SiftMatching.max_num_matches 65536

# Step 3: 稀疏重建
mkdir -p $DATA_DIR/sparse
colmap mapper \
    --database_path $DATA_DIR/database.db \
    --image_path $DATA_DIR/images \
    --output_path $DATA_DIR/sparse \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_extra_params 0

# Step 4: 图像去畸变（如果之前没做，COLMAP 也可以做）
# 注意：rs30_parser 已经做了去畸变，这步通常不需要
# colmap image_undistorter \
#     --image_path $DATA_DIR/images \
#     --input_path $DATA_DIR/sparse/0 \
#     --output_path $DATA_DIR/dense \
#     --output_type COLMAP
```

> 💡 **提示**：如果图像数量多（3相机 × 481帧 = 1443张），exhaustive_matcher 会很慢。
> 可以考虑用 sequential_matcher 或 vocab_tree_matcher：
> ```bash
> colmap sequential_matcher --database_path $DATA_DIR/database.db
> ```
> 或者只选一个相机（如 Camera1）先跑通全流程。

## 8. 运行 SuGaR 训练

```bash
cd ~/projects/SuGaR
conda activate sugar

DATA_DIR=~/data/2026-06-13-BD-GLKGQ_colmap

# 完整流程（推荐 dn_consistency 正则化）
python train_full_pipeline.py \
    -s $DATA_DIR \
    -r dn_consistency \
    --high_poly True \
    --export_obj True \
    --eval False

# 参数说明：
# -s: COLMAP 数据集路径
# -r: 正则化方法（dn_consistency 推荐，density 适合物体，sdf 适合大场景）
# --high_poly: 100万顶点网格（更精细）
# --low_poly: 20万顶点网格（更快，6个高斯/三角形）
# --eval False: 不做评估分割（数据量有限时用全部训练）
# --refinement_time short/medium/long: 精炼时间（2k/7k/15k 迭代）
# --white_background: 白色背景（室内可选）
```

### 分步执行（调试用）

```bash
# Step 1: 训练 vanilla 3DGS（7k 迭代，约 10-15 分钟）
python train.py -s $DATA_DIR

# Step 2: SuGaR 正则化
python train_coarse_density.py \
    -s $DATA_DIR \
    -r dn_consistency \
    --gs_output_dir output/xxx  # Step 1 的输出

# Step 3: 提取网格
python extract_mesh.py \
    -s $DATA_DIR \
    --load_coarse \
    --surface_level 0.3

# Step 4: 精炼
python train_refined.py \
    -s $DATA_DIR \
    --load_coarse \
    --load_mesh

# Step 5: 提取纹理网格
python extract_refined_mesh_with_texture.py \
    -s $DATA_DIR \
    --load_refined
```

## 9. 查看结果

```bash
# 结果保存在 output/ 目录
# ├── output/
# │   ├── coarse/           # 粗糙 3DGS + SuGaR
# │   ├── mesh/             # 提取的网格 (.ply)
# │   ├── refined/          # 精炼后的 SuGaR
# │   ├── refined_ply/      # 精炼高斯 (.ply，可用 3DGS 查看器看)
# │   └── refined_mesh/     # 纹理网格 (.obj)

# 使用 SuGaR 查看器
python run_viewer.py -p ./output/refined_ply/<scene_name>/point_cloud.ply

# 或用 SuperSplat 在线查看
# https://playcanvas.com/supersplat/editor
```

## 10. 常见问题

### Q: CUDA out of memory
```bash
# 减少图像分辨率
# 在 rs30_parser 转换时降采样：
python -m rs30_parser extract --max-frames 200 ...

# 或用 --low_poly
python train_full_pipeline.py -s $DATA_DIR -r dn_consistency --low_poly True
```

### Q: COLMAP mapper 失败（注册图像太少）
```bash
# 增加特征数量
colmap feature_extractor ... --SiftExtraction.max_num_features 16384

# 放松匹配阈值
colmap mapper ... --Mapper.init_min_num_inliers 50

# 只用一个相机
python -m rs30_parser to_colmap ... --cameras Camera1
```

### Q: 网格质量差（有洞或噪声）
```bash
# 尝试不同正则化方法
python train_full_pipeline.py -s $DATA_DIR -r sdf ...     # 大场景
python train_full_pipeline.py -s $DATA_DIR -r density ...  # 物体

# 调整 surface_level
python extract_mesh.py ... --surface_level 0.1  # 更低 = 更多细节
python extract_mesh.py ... --surface_level 0.5  # 更高 = 更平滑
```

### Q: RS30 数据的特殊注意事项
- **2 FPS 采样率**：图像间运动较大，COLMAP 匹配可能困难。建议先用单相机 100 帧测试
- **3 相机视角**：Camera1-3 覆盖不同方向，可以一起用增加约束，但也会增加计算量
- **变电站结构**：大量金属表面、重复纹理，COLMAP 匹配可能出错。考虑增加特征数量
- **去畸变是必须的**：RS30 镜头畸变显著（k1≈0.39），不去畸变 SuGaR 无法工作

## 11. 性能参考

| 场景规模 | 图像数 | 3DGS 训练 | SuGaR 正则化 | 网格提取 | 精炼 | 总计 |
|----------|--------|-----------|-------------|---------|------|------|
| 小（1相机 100帧） | 100 | ~10min | ~15min | ~5min | ~5min | ~35min |
| 中（1相机 481帧） | 481 | ~30min | ~30min | ~10min | ~15min | ~85min |
| 大（3相机 481帧） | 1443 | ~90min | ~60min | ~20min | ~30min | ~200min |

> 以上为 RTX 4090 估算时间，3090 大约慢 1.5-2x
