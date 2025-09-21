# Phong Tessellation Renderer

本项目是基于 Tamy Boubekeur 和 Marc Alexa 在SIGGRAPH Asia 2008发表的论文《Phong Tessellation》的一个Python实现。它利用现代OpenGL的曲面细分着色器（Tessellation Shaders）在GPU上动态地、实时地对3D模型进行几何细分，从而生成平滑的曲面，极大地改善了模型在轮廓处的视觉效果。

该工具集成了交互式查看器和强大的批量渲染模式，旨在成为一个兼具技术验证和实用性的图形学工具。

## ✨ 主要功能

* **核心算法**:
    * 完整实现了Phong Tessellation算法，通过在顶点切平面上进行投影插值来生成平滑的几何体。
    * 利用OpenGL 4.1+的硬件曲面细分管线（Tessellation Pipeline）进行GPU加速，实现高效率实时渲染。

* **交互模式 (`interactive`)**:
    * 一个功能完备的实时3D模型查看器。
    * **模型自动归一化**: 无论模型原始尺寸或位置如何，加载时都会自动居中并缩放到合适的尺寸，确保第一时间看到模型全貌。
    * **Arcball相机控制**: 通过鼠标拖拽和滚轮，实现直观的轨道式旋转和缩放。
    * **实时参数调整**: 通过键盘方向键，可以实时调整细分级别（Tessellation Level）和形状因子（Shape Factor），即时预览不同参数下的曲面效果。
    * **状态显示**: 在窗口标题栏实时显示FPS、细分级别和形状因子等关键参数。

* **批量模式 (`batch`)**:
    * **离屏渲染**: 在后台以不可见的方式运行，专为自动化图像生成设计。
    * **自动相机布局**: 智能计算模型的包围球，并从多个随机视角进行拍摄，保证渲染结果构图合理。
    * **对比图生成**: 针对每一个相机视角，自动生成两张效果图：一张是原始的、未经细分的模型，另一张是经过Phong Tessellation处理的光滑模型，且两者视角严格对齐，便于效果比较。
    * **HDR格式输出**: 渲染结果保存为高动态范围的 `.exr` 浮点数格式，无损保留所有光照细节。

* **网格处理**:
    * **模型简化**: 在渲染前对网格进行简化处理，以控制渲染性能和基础面数。

## 🛠️ 技术栈

* **语言**: Python 3.10
* **图形API**: OpenGL 4.1+ (4.1 here)
* **核心库**:
    * `PyOpenGL`: Python的OpenGL绑定。
    * `GLFW`: 用于创建窗口、处理输入和管理OpenGL上下文。
    * `PyGLM`: 用于图形学相关的数学计算（向量、矩阵）。
    * `Trimesh`: 功能强大的网格处理库，用于加载、分析、简化和归一化模型。
    * `faye-image` / `imageio`: 用于将渲染结果保存为 `.exr` 文件。
    * `NumPy`: 提供高性能的数值计算支持。

## 📂 项目结构
```
phong_tessellation/
├── shaders/                # GLSL着色器文件
│   ├── tess.vert
│   ├── tess.tcs
│   ├── tess.tes
│   └── tess.frag
├── models/                 # 存放 .obj 等模型文件
│   └── teapot.obj
├── output/                 # 批量渲染的输出目录
│   ├── original/           # 存放原始模型的渲染结果
│   └── tessellated/        # 存放细分后模型的渲染结果
├── main.py                 # 主程序入口
└── requirements.txt        # Python依赖库列表
```

## 🚀 安装与设置

1.  **克隆或下载项目**
    将项目文件保存到本地。

2.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows: venv\Scripts\activate
    ```

3.  **安装依赖**
    项目根目录下，运行以下命令安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 如何运行

本项目通过命令行界面进行操作，分为`interactive`和`batch`两种模式。

### 1. 查看帮助信息

您可以随时查看每个模式支持的参数。

```bash
# 查看主帮助和可用模式
python main.py --help

# 查看 batch 模式的具体参数
python main.py batch --help
```

### 2. 运行交互模式

启动实时的模型查看器。

```bash
# 使用默认的茶壶模型
python main.py interactive

# 指定一个不同的模型，并设置不同的简化目标
python main.py interactive --model path/to/your/model.obj --faces 5000
```

**交互操作**:

- 旋转: 按住鼠标左键 + 拖动
- 缩放: 滚动鼠标滚轮
- 调整细分等级: ↑ / ↓ 方向键
- 调整形状因子: ← / → 方向键

## 3. 运行批量模式

启动后台渲染进程，生成对比图。

```Bash
# 使用默认参数进行批量渲染
python main.py batch

# 运行一个自定义的高质量批量渲染任务
python main.py batch --model models/teapot.obj --num_images 50 --width 2048 --height 2048 --tess_level 64 --shape_factor 1.0 --faces 2000
```

渲染任务将在终端显示进度，结束后请到 output/ 目录查看生成的图片。