# 需求：visualize 命令使用 gsplat 渲染

## 背景

当前 `python main.py visualize` 使用 Open3D 的交互式查看器显示点云（普通点云渲染），没有使用 3DGS 的高斯泼溅渲染效果。`render-video` 命令已经支持 gsplat 渲染（`renderer.py` 中的 `_render_gsplat`），但 visualize 命令没有复用。

## 目标

让 `visualize` 命令也能使用 gsplat 进行高质量 3DGS 渲染，提供接近真实的场景可视化效果。

## 需求详细说明

### 1. 新增 `visualize-gsplat` 子命令

在 `main.py` 中添加新子命令 `visualize-gsplat`，使用 gsplat 渲染器逐帧渲染轨迹，输出为视频或图片序列。

功能：
- 复用 `renderer.py` 中已有的 `_render_gsplat` 函数
- 加载 PLY 数据 → 生成轨迹 → 逐帧 gsplat 渲染 → 输出视频
- 本质上是 `render-video` 的别名/变体，但明确使用 gsplat 后端

或者更好的方案：**给现有的 `visualize` 命令添加 `--gsplat` 参数**，当指定时使用 gsplat 渲染模式而非 Open3D 点云模式。

### 2. 交互式 gsplat 可视化（高级，可选）

如果可行，实现一个交互式的 gsplat 渲染窗口：
- 使用 pygame 或 OpenCV 显示渲染帧
- 支持键盘/鼠标控制相机位姿（WASD 移动，鼠标旋转）
- 实时 gsplat 渲染当前视角
- 类似 3DGS viewer 的体验

这需要考虑性能（gsplat 渲染速度是否足够支持实时交互）。

### 3. 渲染预览模式

添加 `visualize --preview` 模式：
- 在轨迹的若干关键帧（起点、1/4、1/2、3/4、终点）渲染 gsplat 图片
- 保存为 PNG，方便快速预览轨迹效果
- 不需要完整渲染所有帧

## 技术约束

- 必须复用 `renderer.py` 中已有的 `render_view()` 函数和 `_render_gsplat()` 函数
- gsplat 渲染需要 CUDA，在 macOS 上不可用，需要 fallback 到 Open3D 或 numpy
- 保持向后兼容：不加 `--gsplat` 参数时，`visualize` 行为不变（Open3D 交互式查看）
- 代码风格与现有代码一致

## 文件修改范围

- `main.py`：添加 `--gsplat` 和 `--preview` 参数
- `src/visualize.py`：添加 gsplat 渲染和预览功能函数
- `src/renderer.py`：无需修改（已有完整 gsplat 渲染逻辑）

## 验收标准

1. `python main.py visualize --gsplat` 能使用 gsplat 渲染轨迹并输出视频
2. `python main.py visualize --preview` 能渲染关键帧预览图片
3. `python main.py visualize`（无参数）行为不变，仍用 Open3D
4. macOS 上无 CUDA 时自动 fallback，不崩溃
