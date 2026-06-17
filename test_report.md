# pointcloud-nav Mac 测试报告

## 环境
- 系统: macOS Darwin 25.5.0 (arm64, Apple Silicon)
- Python: 3.12.13 (venv)
- 虚拟环境: ~/.openclaw/workspace-coder/pointcloud-nav/.venv/
- 依赖: 全部通过清华源安装成功
- 渲染后端: numpy 投影渲染 (macOS 无头环境 fallback)

## 测试结果

### ✅ PLY 加载 — 通过
- AHOLO.ply: 306,764 个高斯椭球体, 59 个属性
- 场景边界: [-9.69, -2.52, -7.10] ~ [4.47, 5.48, 4.27]

### ✅ 轨迹生成 — 通过
- 默认轨迹总长 26.71m, 801 帧

### ✅ render-video — 通过
- 渲染 801 帧, 640x480
- 输出: output/video.mp4 (11.2 MB, 26.70s)
- 渲染后端: numpy 投影渲染

### ✅ collect-vln — 通过
- 采集 801 帧 VLN 数据
- 输出: output/vln_data/
  - frames/ (RGB + 深度图)
  - vln_data.json (元数据)

### ⚠️ visualize — 预期不可用
- 原因: Mac mini 无头环境，Open3D GUI 需要显示器
- 不是 bug，需接显示器或用 VNC

## 修复内容

### src/renderer.py
1. 新增 macOS 无头环境检测逻辑
2. 新增 `_render_numpy()` — 纯 numpy 点云投影渲染
   - 透视投影 + 画家算法深度排序
   - 无需 GPU，无需 OpenGL/EGL
   - 渲染效果比 gsplat 差（点云 vs 3DGS），但可正常使用
3. 后端优先级: gsplat > open3d (有显示器) > numpy (macOS 无头)
4. 原有 gsplat 和 Open3D 代码未修改

### requirements.txt
- 无需新增依赖（numpy 已是核心依赖）

## 输出文件位置
- 视频: ~/.openclaw/workspace-coder/pointcloud-nav/output/video.mp4
- VLN 数据: ~/.openclaw/workspace-coder/pointcloud-nav/output/vln_data/
- 测试报告: ~/.openclaw/workspace-coder/pointcloud-nav/test_report.md
