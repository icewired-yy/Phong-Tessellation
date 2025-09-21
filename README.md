# Phong Tessellation Renderer

This project is a Python implementation based on the paper "***Phong Tessellation***" by Tamy Boubekeur and Marc Alexa, published at SIGGRAPH Asia 2008. It utilizes modern OpenGL tessellation shaders to dynamically and real-time subdivide 3D models on the GPU, generating smooth surfaces that dramatically improve the visual quality of model silhouettes.

This toolkit integrates an interactive viewer and powerful batch rendering capabilities, designed to be both a technical validation tool and a practical graphics utility.

## ✨ Key Features

* **Core Algorithm**:
    * Complete implementation of the Phong Tessellation algorithm, generating smooth geometry through projection interpolation on vertex tangent planes.
    * Leverages OpenGL 4.1+ hardware tessellation pipeline for GPU-accelerated, high-efficiency real-time rendering.

* **Interactive Mode (`interactive`)**:
    * A fully-featured real-time 3D model viewer.
    * **Automatic Model Normalization**: Regardless of the original model size or position, models are automatically centered and scaled to appropriate dimensions upon loading, ensuring immediate visibility of the full model.
    * **Arcball Camera Control**: Intuitive orbit-style rotation and zoom through mouse drag and scroll wheel interactions.
    * **Real-time Parameter Adjustment**: Use keyboard arrow keys to adjust tessellation level and shape factor in real-time, instantly previewing surface effects under different parameters.
    * **Status Display**: Real-time display of FPS, tessellation level, and shape factor in the window title bar.

* **Batch Mode (`batch`)**:
    * **Offscreen Rendering**: Runs invisibly in the background, designed for automated image generation.
    * **Automatic Camera Layout**: Intelligently calculates the model's bounding sphere and captures from multiple random viewpoints, ensuring well-composed rendering results.
    * **Comparison Image Generation**: For each camera viewpoint, automatically generates two images: one original, unsubdivided model and one smooth model processed with Phong Tessellation, with strictly aligned viewpoints for easy effect comparison.
    * **HDR Format Output**: Rendering results are saved in high dynamic range `.exr` floating-point format, preserving all lighting details without loss.

* **Mesh Processing**:
    * **Model Simplification**: Mesh simplification processing before rendering to control rendering performance and base polygon count.

## 🛠️ Technology Stack

* **Language**: Python 3.10
* **Graphics API**: OpenGL 4.1+ (4.1 required here)
* **Core Libraries**:
    * `PyOpenGL`: Python OpenGL bindings.
    * `GLFW`: For creating windows, handling input, and managing OpenGL contexts.
    * `PyGLM`: For graphics-related mathematical computations (vectors, matrices).
    * `Trimesh`: Powerful mesh processing library for loading, analyzing, simplifying, and normalizing models.
    * `faye-image` / `imageio`: For saving rendering results as `.exr` files.
    * `NumPy`: Provides high-performance numerical computation support.

## 📂 Project Structure
```
phong_tessellation/
├── shaders/                # GLSL shader files
│   ├── tess.vert
│   ├── tess.tcs
│   ├── tess.tes
│   └── tess.frag
├── models/                 # Storage for .obj and other model files
│   └── teapot.obj
├── output/                 # Batch rendering output directory
│   ├── original/           # Original model rendering results
│   └── tessellated/        # Tessellated model rendering results
├── main.py                 # Main program entry point
└── requirements.txt        # Python dependency list
```

## 🚀 Installation and Setup

1.  **Clone or Download Project**
    Save the project files locally.

2.  **Create Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    In the project root directory, run the following command to install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run

This project operates through a command-line interface, with `interactive` and `batch` modes.

### 1. View Help Information

You can view supported parameters for each mode at any time.

```bash
# View main help and available modes
python main.py --help

# View specific parameters for batch mode
python main.py batch --help
```

### 2. Run Interactive Mode

Launch the real-time model viewer.

```bash
# Use default teapot model
python main.py interactive

# Specify a different model and set different simplification target
python main.py interactive --model path/to/your/model.obj --faces 5000
```

**Interactive Controls**:

- Rotate: Hold left mouse button + drag
- Zoom: Scroll mouse wheel
- Adjust tessellation level: ↑ / ↓ arrow keys
- Adjust shape factor: ← / → arrow keys

## 3. Run Batch Mode

Launch background rendering process to generate comparison images.

```bash
# Use default parameters for batch rendering
python main.py batch

# Run a custom high-quality batch rendering task
python main.py batch --model models/teapot.obj --num_images 50 --width 2048 --height 2048 --tess_level 64 --shape_factor 1.0 --faces 2000
```

The rendering task will display progress in the terminal. After completion, check the `output/` directory for generated images.
