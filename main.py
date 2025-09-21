# ===== IMPORTS =====
# OpenGL and graphics libraries
import glfw
from OpenGL.GL import *
import glm

# Mesh processing and rendering
import trimesh
import numpy as np

# Image processing and file I/O
import faye_image as fy
import os

# System and utilities
import sys
import time
import argparse 


def create_shader_program(vs_path, tcs_path, tes_path, fs_path):
    """
    Create and compile a complete OpenGL shader program with tessellation shaders.
    
    Args:
        vs_path: Path to vertex shader file
        tcs_path: Path to tessellation control shader file  
        tes_path: Path to tessellation evaluation shader file
        fs_path: Path to fragment shader file
    
    Returns:
        OpenGL program ID
    """
    # Define shader types and their corresponding file paths
    shaders = {
        GL_VERTEX_SHADER: vs_path,
        GL_TESS_CONTROL_SHADER: tcs_path,
        GL_TESS_EVALUATION_SHADER: tes_path,
        GL_FRAGMENT_SHADER: fs_path
    }
    
    shader_ids = []
    
    # Compile each shader
    for s_type, path in shaders.items():
        # Read shader source code from file
        with open(path, 'r') as f:
            source = f.read()
        
        # Create and compile shader
        s_id = glCreateShader(s_type)
        glShaderSource(s_id, source)
        glCompileShader(s_id)
        
        # Check for compilation errors
        if not glGetShaderiv(s_id, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(s_id).decode()
            glDeleteShader(s_id)
            raise RuntimeError(f"Shader {path} compilation failed: {error}")
        
        shader_ids.append(s_id)
    
    # Create and link shader program
    program = glCreateProgram()
    for s_id in shader_ids:
        glAttachShader(program, s_id)
    
    glLinkProgram(program)
    
    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    
    # Clean up individual shaders (they're now part of the program)
    for s_id in shader_ids:
        glDeleteShader(s_id)
    
    return program

class Camera:
    """
    Arcball camera controller for interactive 3D viewing.
    Provides orbit-style camera movement around a target point.
    """
    
    def __init__(self, position=glm.vec3(0, 3, 5)):
        """Initialize camera with default position and target."""
        self.position = position
        self.target = glm.vec3(0, 0, 0)  # Look at origin
        self.up = glm.vec3(0, 1, 0)      # World up vector
        
        # Spherical coordinates for orbiting
        self.yaw = -90.0      # Horizontal rotation
        self.pitch = -30.0    # Vertical rotation  
        self.distance = 5.0   # Distance from target
        
        self.update_camera_vectors()
    
    def get_view_matrix(self):
        """Get the view matrix for rendering."""
        return glm.lookAt(self.position, self.target, self.up)
    
    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Process mouse movement for camera rotation."""
        sensitivity = 0.2
        self.yaw += xoffset * sensitivity
        self.pitch -= yoffset * sensitivity
        
        # Constrain pitch to avoid flipping
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
        
        self.update_camera_vectors()
    
    def process_mouse_scroll(self, yoffset):
        """Process mouse scroll for camera zoom."""
        self.distance -= yoffset
        
        # Clamp distance to reasonable bounds
        if self.distance < 1.0:
            self.distance = 1.0
        if self.distance > 45.0:
            self.distance = 45.0
        
        self.update_camera_vectors()
    
    def update_camera_vectors(self):
        """Update camera position based on spherical coordinates."""
        # Convert spherical coordinates to Cartesian position
        x = self.target.x + self.distance * glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        y = self.target.y + self.distance * glm.sin(glm.radians(self.pitch))
        z = self.target.z + self.distance * glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.position = glm.vec3(x, y, z)

class Renderer:
    """
    Main renderer class implementing Phong Tessellation using OpenGL.
    Supports both interactive viewing and batch rendering modes.
    """
    
    def __init__(self, width=1024, height=1024, is_batch=False):
        """
        Initialize the renderer with OpenGL context and shaders.
        
        Args:
            width: Render target width
            height: Render target height  
            is_batch: If True, create invisible window for batch rendering
        """
        self.width, self.height = width, height
        
        # Initialize GLFW
        if not glfw.init():
            sys.exit(-1)
        
        # Set window to invisible for batch mode
        if is_batch:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        
        # Configure OpenGL context (require 4.1+ for tessellation shaders)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        # Create window and OpenGL context
        self.window = glfw.create_window(width, height, "Phong Tessellation", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit(-1)
        
        glfw.make_context_current(self.window)
        
        # Load and compile tessellation shader program
        self.shader = create_shader_program(
            "shaders/tess.vert", 
            "shaders/tess.tcs", 
            "shaders/tess.tes", 
            "shaders/tess.frag"
        )
        
        # Create vertex array and buffer objects
        self.vao, self.vbo = glGenVertexArrays(1), glGenBuffers(1)
        self.vertex_count = 0
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Initialize framebuffer for batch rendering
        if is_batch:
            self._init_fbo()
    
    def simplify(self, mesh, target_faces):
        """
        Simplify mesh to reduce polygon count if needed.
        
        Args:
            mesh: Input trimesh object
            target_faces: Target number of faces after simplification
            
        Returns:
            Simplified mesh or original if already below target
        """
        if len(mesh.faces) <= target_faces:
            return mesh
        
        try:
            print(f"Simplifying mesh from {len(mesh.faces)} to {target_faces} faces...")
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces, aggression=4)
            print("Simplification successful.")
            return simplified
        except Exception as e:
            print(f"Simplification failed: {e}")
            return mesh

    def load_model(self, filepath, target_faces=1000):
        """
        Load and prepare a 3D model for tessellation rendering.
        
        Args:
            filepath: Path to the .obj model file
            target_faces: Target face count for mesh simplification
        """
        try:
            print(f"Loading model: {filepath}")
            mesh = trimesh.load(filepath, process=True)

            # === MESH NORMALIZATION ===
            # Center the mesh at the origin
            center = mesh.bounds.mean(axis=0)
            mesh.apply_translation(-center)

            # Scale the mesh to fit within a 2x2x2 cube (radius of 1)
            scale = mesh.extents.max()
            mesh.apply_scale(2.0 / scale)
            print("Model normalized (centered and scaled).")

            # Simplify mesh if needed
            mesh = self.simplify(mesh, target_faces=target_faces)
            
            # Ensure vertex normals are computed
            mesh.vertex_normals
            
            # Flatten mesh data for OpenGL (each triangle vertex gets its own data)
            vertices = mesh.vertices[mesh.faces.flatten()]
            normals = mesh.vertex_normals[mesh.faces.flatten()]
            
            # Interleave vertex positions and normals [x,y,z, nx,ny,nz, ...]
            interleaved_data = np.hstack([vertices, normals]).astype(np.float32).flatten()
            self.vertex_count = len(vertices)
            
            # Upload vertex data to GPU
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, interleaved_data.nbytes, interleaved_data, GL_STATIC_DRAW)
            
            # Configure vertex attributes
            stride = 6 * sizeof(GLfloat)  # 3 position + 3 normal floats per vertex
            
            # Position attribute (location 0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            
            # Normal attribute (location 1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * sizeof(GLfloat)))
            glEnableVertexAttribArray(1)
            
            # Cleanup
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(-1)

    def _init_fbo(self):
        """
        Initialize framebuffer object for offscreen rendering (batch mode).
        Creates a high-precision RGBA16F color buffer and depth buffer.
        """
        # Create and bind framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        # Create color texture (RGBA16F for HDR rendering)
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)
        
        # Create depth renderbuffer
        self.rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
        
        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete!")
        
        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _render_scene(self, view, proj, camera_pos, tess_level, shape_factor):
        """
        Render the loaded mesh with tessellation shaders.
        
        Args:
            view: View matrix (camera transformation)
            proj: Projection matrix  
            camera_pos: Camera position for lighting calculations
            tess_level: Tessellation subdivision level
            shape_factor: Shape interpolation factor (0=flat, 1=curved)
        """
        # Clear framebuffer with dark blue background
        glClearColor(0.1, 0.2, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use tessellation shader program
        glUseProgram(self.shader)
        
        # Set transformation matrices
        model = glm.mat4(1.0)  # Identity matrix (no model transformation)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "proj"), 1, GL_FALSE, glm.value_ptr(proj))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        
        # Set tessellation parameters
        glUniform1f(glGetUniformLocation(self.shader, "tessLevel"), tess_level)
        glUniform1f(glGetUniformLocation(self.shader, "shapeFactor"), shape_factor)
        
        # Set camera position for lighting
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, glm.value_ptr(camera_pos))
        
        # Draw tessellated mesh
        glBindVertexArray(self.vao)
        glDrawArrays(GL_PATCHES, 0, self.vertex_count)  # Render as patches for tessellation
        glBindVertexArray(0)

    def run_interactive(self, model_path, target_faces):
        """
        Run interactive mode with real-time 3D viewer.
        
        Args:
            model_path: Path to the 3D model file
            target_faces: Target face count for mesh simplification
        """
        # Initialize interactive mode variables
        self.camera = Camera()
        self.tess_level, self.shape_factor = 16.0, 0.75
        self.last_mouse_x, self.last_mouse_y = self.width / 2, self.height / 2
        self.mouse_button_down = False
        self.last_time, self.frame_count, self.fps = time.time(), 0, 0
        
        # Set up input callbacks
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        
        # Initialize window title and load model
        self.update_window_title()
        self.load_model(model_path, target_faces)
        
        # Configure tessellation (3 vertices per triangle patch)
        glPatchParameteri(GL_PATCH_VERTICES, 3)
        
        # Main render loop
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            self.frame_count += 1
            
            # Update FPS counter every second
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
                self.update_window_title()
            
            # Process input events
            glfw.poll_events()
            
            # Set up camera matrices
            proj = glm.perspective(glm.radians(45), self.width / self.height, 0.1, 100.0)
            view = self.camera.get_view_matrix()
            
            # Render scene with current parameters
            self._render_scene(view, proj, self.camera.position, self.tess_level, self.shape_factor)
            
            # Present rendered frame
            glfw.swap_buffers(self.window)
        
        self.shutdown()

    # === INPUT CALLBACK METHODS ===
    def cursor_pos_callback(self, window, xpos, ypos):
        """Handle mouse cursor movement for camera rotation."""
        if self.mouse_button_down:
            xoffset, yoffset = xpos - self.last_mouse_x, ypos - self.last_mouse_y
            self.camera.process_mouse_movement(xoffset, yoffset)
        self.last_mouse_x, self.last_mouse_y = xpos, ypos
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for camera zoom."""
        self.camera.process_mouse_scroll(yoffset)
    
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button press/release."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_button_down = (action == glfw.PRESS)
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input for tessellation parameter adjustment."""
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Adjust tessellation level with up/down arrows
            if key == glfw.KEY_UP:
                self.tess_level += 1
            if key == glfw.KEY_DOWN:
                self.tess_level = max(1.0, self.tess_level - 1)
            
            # Adjust shape factor with left/right arrows
            if key == glfw.KEY_RIGHT:
                self.shape_factor = min(1.0, self.shape_factor + 0.05)
            if key == glfw.KEY_LEFT:
                self.shape_factor = max(0.0, self.shape_factor - 0.05)
    
    def update_window_title(self):
        """Update window title with current rendering parameters."""
        title = f"FPS: {self.fps:.1f} | Tess Level: {self.tess_level:.1f} | Shape Factor: {self.shape_factor:.2f}"
        glfw.set_window_title(self.window, title)

    def run_batch(self, model_path, num_images, fov, tess_level, shape_factor, target_faces, output_dir):
        """
        Run batch rendering mode to generate comparison images.
        
        Args:
            model_path: Path to the 3D model file
            num_images: Number of image pairs to generate
            fov: Camera field of view in degrees
            tess_level: Tessellation level for subdivided images
            shape_factor: Shape interpolation factor
            target_faces: Target face count for mesh simplification
        """
        print("--- Starting Batch Render ---")
        
        # Load and prepare model
        self.load_model(model_path, target_faces)
        glPatchParameteri(GL_PATCH_VERTICES, 3)
        
        # Calculate camera positioning based on model bounding sphere
        # mesh = trimesh.load(model_path)
        # center, radius = mesh.bounding_sphere.center, mesh.bounding_sphere.primitive.radius
        
        # --- FIX: Calculate camera distance based on a NORMALIZED model ---
        # The normalized model has a center at (0,0,0) and a radius of ~1.0
        center = glm.vec3(0, 0, 0)
        radius = 1.0 
        distance = (radius / np.tan(glm.radians(fov) / 2.0)) * 1.5  # Add 50% margin for better framing
        print(f"Using normalized camera distance: {distance:.2f}")

        distance = (radius / np.tan(glm.radians(fov) / 2.0)) * 1.2  # Add 20% margin
        proj = glm.perspective(glm.radians(fov), self.width / self.height, 0.1, distance * 2)
        
        # Create output directories
        output_original_dir, output_tessellated_dir = os.path.join(output_dir, "original"), os.path.join(output_dir, "tessellated")
        os.makedirs(output_original_dir, exist_ok=True)
        os.makedirs(output_tessellated_dir, exist_ok=True)
        
        # Generate image pairs from random viewpoints
        for i in range(num_images):
            # Generate random camera position on sphere around model
            random_vec = np.random.randn(3)
            random_vec /= np.linalg.norm(random_vec)  # Normalize to unit sphere
            cam_pos_np = center + random_vec * distance
            cam_pos = glm.vec3(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])
            view = glm.lookAt(cam_pos, glm.vec3(center[0], center[1], center[2]), glm.vec3(0, 1, 0))
            
            # Render original (non-tessellated) version
            print(f"Rendering pair {i+1}/{num_images} (Original)...")
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.width, self.height)
            self._render_scene(view, proj, cam_pos, tess_level=1.0, shape_factor=0.0)
            
            # Read and save original image
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT)
            image_data = np.flipud(np.frombuffer(pixels, dtype=np.float32).reshape(self.height, self.width, 4))
            fy.From(image_data, fy.NUMPY_RT).Save(f"{output_original_dir}/render_{i:03d}.exr", fy.EXR_FILE)
            
            # Render tessellated version
            print(f"Rendering pair {i+1}/{num_images} (Tessellated)...")
            self._render_scene(view, proj, cam_pos, tess_level, shape_factor)
            
            # Read and save tessellated image
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT)
            image_data = np.flipud(np.frombuffer(pixels, dtype=np.float32).reshape(self.height, self.width, 4))
            fy.From(image_data, fy.NUMPY_RT).Save(f"{output_tessellated_dir}/render_{i:03d}.exr", fy.EXR_FILE)
        
        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        print("--- Batch Render Finished ---")
        self.shutdown()

    def shutdown(self):
        """Clean up and terminate GLFW."""
        glfw.terminate()

# ===== MAIN FUNCTION =====
def main():
    """
    Main function with command-line argument parsing.
    Supports two modes: interactive (real-time viewer) and batch (image generation).
    """
    parser = argparse.ArgumentParser(description="Phong Tessellation Renderer")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Rendering mode")

    # === Interactive Mode Arguments ===
    parser_interactive = subparsers.add_parser("interactive", help="Run in interactive mode")
    parser_interactive.add_argument(
        "--model", type=str, default="models/teapot.obj", 
        help="Path to the model file"
    )
    parser_interactive.add_argument(
        "--faces", type=int, default=1000, 
        help="Target face count for simplification in interactive mode"
    )

    # === Batch Mode Arguments ===
    parser_batch = subparsers.add_parser("batch", help="Run in batch rendering mode")
    parser_batch.add_argument(
        "--model", type=str, default="models/teapot.obj", 
        help="Path to the model file"
    )
    parser_batch.add_argument(
        "--num_images", type=int, default=10, 
        help="Number of images to generate"
    )
    parser_batch.add_argument(
        "--width", type=int, default=1920, 
        help="Image width"
    )
    parser_batch.add_argument(
        "--height", type=int, default=1080, 
        help="Image height"
    )
    parser_batch.add_argument(
        "--fov", type=float, default=30.0, 
        help="Camera field of view in degrees"
    )
    parser_batch.add_argument(
        "--tess_level", type=float, default=32.0, 
        help="Tessellation level for subdivided output"
    )
    parser_batch.add_argument(
        "--shape_factor", type=float, default=0.75, 
        help="Shape factor (alpha) for tessellation"
    )
    parser_batch.add_argument(
        "--faces", type=int, default=1000, 
        help="Target face count for mesh simplification"
    )
    parser_batch.add_argument(
        "--output_dir", type=str, default="output", 
        help="Directory to save output images"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Run appropriate mode based on arguments
    if args.mode == "interactive":
        app = Renderer()
        app.run_interactive(model_path=args.model, target_faces=args.faces)
    elif args.mode == "batch":
        app = Renderer(width=args.width, height=args.height, is_batch=True)
        app.run_batch(
            model_path=args.model,
            num_images=args.num_images,
            fov=args.fov,
            tess_level=args.tess_level,
            shape_factor=args.shape_factor,
            target_faces=args.faces,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()