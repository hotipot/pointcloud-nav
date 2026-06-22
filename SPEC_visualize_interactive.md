# 需求：visualize 命令交互式 gsplat 渲染

## 背景

当前 `python main.py visualize` 使用 Open3D 的交互式查看器显示点云（普通点云渲染），没有使用 3DGS 的高斯泼溅渲染效果。用户希望交互式查看时也能看到 gsplat 渲染的真实效果。

## 调研结果

gsplat 官方 `simple_viewer.py` 使用 **viser + nerfview** 方案：
- `viser` 是一个基于浏览器的 3D 可视化库（WebSocket 服务器 + 浏览器前端）
- `nerfview` 是基于 viser 的 nerf/gsplat 查看器框架
- 交互方式：浏览器打开 `http://localhost:端口`，鼠标拖拽旋转/缩放，WASD 移动
- 优点：无需 pygame/OpenCV GUI，跨平台，交互体验好
- 依赖：`pip install viser nerfview`

## 实现方案

### 方案 C（优先）：viser + nerfview 浏览器交互式查看

在 `src/visualize.py` 中新增 `visualize_gsplat_interactive()` 函数：

1. 加载 PLY 数据到 GPU（复用 `ply_loader.py` 的 `GaussianData`）
2. 启动 viser 服务器
3. 注册渲染回调函数：
   - 回调接收 `CameraState`（相机位姿）和 `RenderTabState`（渲染参数）
   - 从 `CameraState` 提取 c2w 矩阵和内参 K
   - 调用 `gsplat.rasterization()` 渲染当前视角
   - 返回 RGB numpy 数组
4. 浏览器打开后用户可自由交互

**关键实现细节：**

参考 gsplat 官方 `simple_viewer.py` 的 `viewer_render_fn` 回调：

```python
def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse().contiguous()

    with torch.inference_mode():
        render_colors, render_alphas, info = rasterization(
            means, quats, scales, opacities, colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width, height=height,
            sh_degree=sh_degree,
            render_mode="RGB",
            packed=False,
        )
    renders = render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
    return renders
```

**与现有代码的对接：**

- 从 `GaussianData` 对象提取 gsplat 所需的 tensor（means, quats, scales, opacities, colors）
- 注意 scales 需要用 `actual_scales`（已 exp 处理），opacities 用 `actual_opacity`（已 sigmoid 处理）
- SH 颜色处理参考 `renderer.py` 中 `_render_gsplat()` 的逻辑
- gsplat 1.5.x API：不传 `backgrounds` 参数

**简化版（不依赖 nerfview）：**

如果 `nerfview` 安装困难，可以只用 `viser` 实现更简单的版本：
- 用 viser 的 `SceneCanvas` + `Camera` 控件
- 每次相机变化时触发渲染回调
- 不需要 nerfview 的 GUI 面板

### 方案 A（备选）：pygame + gsplat

如果 viser/nerfview 方案不可行，使用 pygame：
- pygame 创建窗口，显示渲染帧
- WASD 移动，鼠标拖拽旋转
- 每帧调用 `renderer.render_view()` 渲染
- 实现简单但体验不如浏览器方案

## 代码修改范围

- `src/visualize.py`：新增 `visualize_gsplat_interactive()` 函数
- `main.py`：给 `visualize` 子命令添加 `--gsplat-interactive` 参数
- `requirements.txt` 或文档：添加 `viser` 和 `nerfview` 依赖说明

## 技术约束

1. **必须复用 `GaussianData`** 的数据加载逻辑，不要重新实现 PLY 解析
2. **gsplat 1.5.x API 兼容**：不传 `backgrounds` 参数
3. **SH 颜色处理**：复用 `renderer.py` 中 `_render_gsplat()` 的 SH 逻辑
4. **CUDA 必需**：交互式 gsplat 渲染需要 GPU，macOS 不可用时给出明确提示
5. **端口可配置**：默认 8080，可通过 `--port` 参数指定
6. **分辨率可配置**：默认 640x480，可通过 `--width` / `--height` 指定

## 验收标准

1. `python main.py visualize --gsplat-interactive` 启动 viser 服务器
2. 浏览器打开 `http://localhost:8080` 可看到 gsplat 渲染的 3DGS 场景
3. 鼠标拖拽可旋转视角，滚轮缩放，WASD 移动
4. 渲染效果与 `render-video` 一致（使用相同的 gsplat rasterization）
5. Ctrl+C 可正常退出
6. macOS 无 CUDA 时给出明确错误提示，不崩溃

## 参考文件

- gsplat 官方 viewer: `/tmp/gsplat-repo/examples/simple_viewer.py`
- gsplat viewer 类: `/tmp/gsplat-repo/examples/gsplat_viewer.py`
- 现有渲染器: `src/renderer.py`（`_render_gsplat` 函数）
- 现有 PLY 加载: `src/ply_loader.py`（`GaussianData` 类）
