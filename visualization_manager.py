import os
import json
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import math # Added for mandala calculations
# Remove Matplotlib imports for geometric rendering, will be replaced by Pyglet
# import matplotlib.pyplot as plt 
# import matplotlib.patches as patches 
import pyglet
import pyglet.shapes as pyglet_shapes # Alias to avoid name conflicts
import pyglet.graphics
import logging

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for pattern visualization."""
    grid_size: int = 8  # E8 lattice size
    zoom_range: tuple = (0.5, 2.0)  # Min/max zoom levels
    rotation_step: int = 15  # Degrees per rotation step
    pattern_threshold: float = 0.3  # Threshold for lattice cell activation
    color_scheme: Dict[str, str] = None  # Color mapping for features

    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'background': '#1a1a1a',
                'card': '#2a2a2a',
                'text': '#ffffff',
                'accent': '#666666',
                'highlight': '#888888'
            }

class PatternVisualizer:
    """Handles pattern visualization and interaction."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(self.template_dir, exist_ok=True)
        
    def _prepare_pattern_data(self, pattern: torch.Tensor) -> Dict[str, Any]:
        """Convert pattern tensor to visualization-friendly format."""
        pattern_np = pattern.detach().cpu().numpy()
        return {
            'values': pattern_np.tolist(),
            'stats': {
                'depth': float(pattern_np[2]) if len(pattern_np) > 2 else 0.0,
                'complexity': float(pattern_np[1]) if len(pattern_np) > 1 else 0.0,
                'harmony': float(pattern_np[3]) if len(pattern_np) > 3 else 0.0
            },
            'lattice': self._generate_lattice_grid(pattern_np)
        }
    
    def _generate_lattice_grid(self, pattern: np.ndarray) -> List[List[bool]]:
        """Generate E8 lattice representation of pattern."""
        grid = []
        for i in range(self.config.grid_size):
            row = []
            for j in range(self.config.grid_size):
                value = pattern[i % len(pattern)]
                row.append(abs(value) > self.config.pattern_threshold)
            grid.append(row)
        return grid
    
    def _load_template(self, template_name: str) -> str:
        """Load HTML template with framework-specific optimizations."""
        template_path = os.path.join(self.template_dir, f'{template_name}.html')
        if not os.path.exists(template_path):
            self._create_default_templates()
        with open(template_path, 'r') as f:
            return f.read()
    
    def _create_default_templates(self):
        """Create default HTML templates with framework optimizations."""
        # Main template
        main_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pattern Explorer</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="styles.css">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js"></script>
        </head>
        <body>
            <div id="app"></div>
            <script src="app.js"></script>
        </body>
        </html>
        """
        
        # Styles template
        styles_template = """
        :root {
            --background: {{background}};
            --card: {{card}};
            --text: {{text}};
            --accent: {{accent}};
            --highlight: {{highlight}};
        }
        
        body {
            background: var(--background);
            color: var(--text);
            font-family: monospace;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
            perspective: 1000px;
        }
        
        /* Additional styles will be added by the framework */
        """
        
        # JavaScript template
        js_template = """
        // Framework-specific optimizations
        const app = {
            state: {
                patterns: {{patterns}},
                currentView: 'both',
                selectedPatterns: [],
                zoomLevel: 1,
                rotation: 0
            },
            
            init() {
                this.setupThreeJS();
                this.setupTensorFlow();
                this.render();
            },
            
            setupThreeJS() {
                // Three.js setup for 3D visualization
            },
            
            setupTensorFlow() {
                // TensorFlow.js setup for pattern analysis
            },
            
            render() {
                // Framework-optimized rendering
            }
        };
        
        // Initialize app when DOM is ready
        document.addEventListener('DOMContentLoaded', () => app.init());
        """
        
        # Write templates
        with open(os.path.join(self.template_dir, 'main.html'), 'w') as f:
            f.write(main_template)
        with open(os.path.join(self.template_dir, 'styles.css'), 'w') as f:
            f.write(styles_template.format(**self.config.color_scheme))
        with open(os.path.join(self.template_dir, 'app.js'), 'w') as f:
            f.write(js_template)
    
    def generate_visualization(self, 
                             patterns: List[torch.Tensor],
                             mandalas: List[str],
                             concretions: List[str],
                             output_path: str = 'pattern_explorer.html') -> str:
        """Generate interactive visualization with framework optimizations."""
        # Prepare pattern data
        pattern_data = [self._prepare_pattern_data(p) for p in patterns]
        
        # Load and fill template
        template = self._load_template('main')
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write visualization files
        with open(output_path, 'w') as f:
            f.write(template)
        
        # Write pattern data
        with open(output_path.replace('.html', '_data.json'), 'w') as f:
            json.dump({
                'patterns': pattern_data,
                'mandalas': mandalas,
                'concretions': concretions
            }, f)
        
        return output_path

class PatternAnalyzer:
    """Analyzes patterns for visualization optimization."""
    
    def __init__(self):
        self.feature_weights = {
            'depth': 0.4,
            'complexity': 0.3,
            'harmony': 0.3
        }
    
    def analyze_pattern(self, pattern: torch.Tensor) -> Dict[str, float]:
        """Analyze pattern features for visualization optimization."""
        pattern_np = pattern.detach().cpu().numpy()
        return {
            'depth': float(pattern_np[2]) if len(pattern_np) > 2 else 0.0,
            'complexity': float(pattern_np[1]) if len(pattern_np) > 1 else 0.0,
            'harmony': float(pattern_np[3]) if len(pattern_np) > 3 else 0.0
        }
    
    def optimize_visualization(self, patterns: List[torch.Tensor]) -> Dict[str, Any]:
        """Optimize visualization parameters based on pattern analysis."""
        analyses = [self.analyze_pattern(p) for p in patterns]
        
        # Calculate optimal parameters
        avg_depth = np.mean([a['depth'] for a in analyses])
        avg_complexity = np.mean([a['complexity'] for a in analyses])
        avg_harmony = np.mean([a['harmony'] for a in analyses])
        
        return {
            'grid_size': max(4, min(8, int(avg_complexity * 8))),
            'pattern_threshold': max(0.2, min(0.4, avg_harmony)),
            'zoom_range': (0.5, 1.0 + avg_depth),
            'rotation_step': max(10, min(30, int(avg_complexity * 30)))
        }

class VectorDrivenVisualizer:
    """Generates vector-driven visual representations like mandalas and concretions."""

    def __init__(self, mandala_size: int = 21, concretion_size: int = 15):
        self.mandala_size = mandala_size  # Conceptual size, influences scaling
        self.concretion_size = concretion_size
        # ASCII symbols kept if old methods are somehow still used
        self.mandala_symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        self.concretion_symbols = ['o', '.', ':', '-', '>', '<', '^', 'v', '*']
        
        # For Pyglet shapes
        self.mandala_pyglet_shapes = []
        self.concretion_pyglet_shapes = [] # For later
        self.previous_driving_vector: Optional[torch.Tensor] = None # To track dv change
        self.debug_star: Optional[pyglet_shapes.Star] = None # For visual update diagnostic

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple[int, int, int]:
        """Converts HSV color (all components in [0, 1]) to RGB (components in [0, 255])."""
        if s == 0.0:
            r = g = b = int(v * 255)
            return r, g, b
        i = int(h * 6.)
        f = (h * 6.) - i
        p = v * (1. - s)
        q = v * (1. - s * f)
        t = v * (1. - s * (1. - f))
        i %= 6
        if i == 0: r_float, g_float, b_float = v, t, p
        elif i == 1: r_float, g_float, b_float = q, v, p
        elif i == 2: r_float, g_float, b_float = p, v, t
        elif i == 3: r_float, g_float, b_float = p, q, v
        elif i == 4: r_float, g_float, b_float = t, p, v
        else: r_float, g_float, b_float = v, p, q # i == 5
        return int(r_float * 255), int(g_float * 255), int(b_float * 255)

    def _get_driving_vector(self, pattern: torch.Tensor, num_elements: int = 5) -> torch.Tensor:
        """Extracts or derives a driving vector from the pattern."""
        if pattern.numel() == 0:
            return torch.zeros(num_elements)
        
        # Use tanh to bring values into a somewhat predictable range [-1, 1]
        # Then scale to [0,1] for easier use in parameter mapping
        pattern_flat = torch.tanh(pattern.flatten()) * 0.5 + 0.5 

        if pattern_flat.numel() >= num_elements:
            return pattern_flat[:num_elements]
        else:
            # Repeat and slice if the pattern is too short
            return pattern_flat.repeat((num_elements // pattern_flat.numel()) + 1)[:num_elements]

    def create_vector_mandala(self, pattern: torch.Tensor) -> str:
        """Creates an ASCII mandala driven by the input pattern tensor."""
        dv = self._get_driving_vector(pattern, num_elements=5)

        # Map driving vector components to mandala parameters
        # Ensure values are in reasonable ranges
        num_segments = max(3, int(dv[0] * 7) + 3)  # e.g., 3 to 10 segments
        layers = max(2, int(dv[1] * (self.mandala_size / 4)) + 1) # e.g., 2 to ~6 layers
        symbol_offset = int(dv[2] * (len(self.mandala_symbols) - 5)) # Start symbol index
        hue_rotation_factor = dv[3] * 360  # For potential color mapping later, now affects symbol choice
        density = dv[4] * 0.5 + 0.3  # Base density 0.3 to 0.8

        grid = [[' ' for _ in range(self.mandala_size)] for _ in range(self.mandala_size)]
        center_x, center_y = self.mandala_size // 2, self.mandala_size // 2

        for l in range(1, layers + 1):
            radius = l * (center_x / layers) * 0.9 # Scale radius by number of layers
            points_in_layer = int(num_segments * l * density) # More points in outer, denser layers

            for i in range(points_in_layer):
                angle_rad = (2 * math.pi / points_in_layer) * i + (hue_rotation_factor * l /180 * math.pi) # Rotate layers
                
                x_offset = radius * math.cos(angle_rad)
                y_offset = radius * math.sin(angle_rad)
                
                x = center_x + int(x_offset)
                y = center_y + int(y_offset)

                if 0 <= x < self.mandala_size and 0 <= y < self.mandala_size:
                    symbol_index = (l + i + symbol_offset + int(hue_rotation_factor/36)) % len(self.mandala_symbols)
                    grid[y][x] = self.mandala_symbols[symbol_index]
        
        # Reflect to create symmetry (optional, can be driven by dv too)
        # For simplicity, basic reflection for now
        for r in range(self.mandala_size):
            for c in range(center_x):
                grid[r][self.mandala_size - 1 - c] = grid[r][c]
        for c in range(self.mandala_size):
            for r in range(center_y):
                 grid[self.mandala_size - 1 - r][c] = grid[r][c]

        return "\n".join("".join(row) for row in grid)

    def update_geometric_mandala_pyglet(self, 
                                     pattern: torch.Tensor, 
                                     batch: pyglet.graphics.Batch,
                                     group_foreground: Optional[pyglet.graphics.Group] = None,
                                     group_background: Optional[pyglet.graphics.Group] = None,
                                     window_width: int = 800, 
                                     window_height: int = 800,
                                     mandala_shape_bias: float = 0.0,
                                     overall_rotation_angle_deg: float = 0.0):
        """
        Updates or re-creates Pyglet shapes for a geometric mandala and adds them to the batch.
        Shapes are stored in self.mandala_pyglet_shapes.
        """
        # Clear existing shapes
        for shape in self.mandala_pyglet_shapes:
            shape.delete() 
        self.mandala_pyglet_shapes.clear()

        plot_center_x = window_width / 2
        plot_center_y = window_height / 2
        base_plot_radius = min(window_width, window_height) / 2.2

        error_occurred = False
        error_message = ''
        shapes_created = 0
        try:
            # --- MINIMAL EGG: Always-visible Easter egg for rendering test ---
            egg_width = min(window_width, window_height) * 0.25
            egg_height = egg_width * 1.3
            pastel_egg = pyglet_shapes.Ellipse(
                x=plot_center_x,
                y=plot_center_y,
                a=egg_width,
                b=egg_height,
                segments=32,
                color=(255, 230, 180),  # Pastel yellow
                batch=batch,
                group=group_foreground
            )
            self.mandala_pyglet_shapes.append(pastel_egg)
            logger.info(f"Created pastel egg at ({plot_center_x}, {plot_center_y}) size=({egg_width}, {egg_height})")
            shapes_created += 1
            # --- END MINIMAL EGG ---
        except Exception as e:
            logger.error(f"Error creating pastel egg: {e}", exc_info=True)
            error_occurred = True
            error_message = f"Egg error: {e}"
        try:
            # Add radial gradient lines
            num_radial_lines = 5  # Five-fold symmetry for origami
            for i in range(num_radial_lines):
                angle = (2 * math.pi * i) / num_radial_lines + math.radians(overall_rotation_angle_deg)
                end_x = plot_center_x + base_plot_radius * 1.5 * math.cos(angle)
                end_y = plot_center_y + base_plot_radius * 1.5 * math.sin(angle)
                for j in range(3):
                    try:
                        gradient_line = pyglet_shapes.Line(
                            x=plot_center_x,
                            y=plot_center_y,
                            x2=end_x * (0.8 + j * 0.1),
                            y2=end_y * (0.8 + j * 0.1),
                            color=(100, 100, 200, 30 - j * 10),
                            batch=batch,
                            group=group_background
                        )
                        self.mandala_pyglet_shapes.append(gradient_line)
                        logger.info(f"Created gradient line {i}-{j} from ({plot_center_x},{plot_center_y}) to ({end_x},{end_y})")
                        shapes_created += 1
                    except Exception as e:
                        logger.error(f"Error creating gradient line {i}-{j}: {e}", exc_info=True)
                        error_occurred = True
                        error_message = f"Line error: {e}"
        except Exception as e:
            logger.error(f"Error in radial lines block: {e}", exc_info=True)
            error_occurred = True
            error_message = f"Radial lines error: {e}"
        try:
            # Add origami-inspired planes
            num_planes = 5
            for i in range(num_planes):
                plane_radius = base_plot_radius * (1.0 + i * 0.15)
                plane_opacity = int(80 * (1.0 - i * 0.15))
                try:
                    gradient_circle = pyglet_shapes.Circle(
                        x=plot_center_x,
                        y=plot_center_y,
                        radius=plane_radius,
                        color=(50, 50, 100, plane_opacity),
                        batch=batch,
                        group=group_background
                    )
                    self.mandala_pyglet_shapes.append(gradient_circle)
                    logger.info(f"Created gradient circle {i} at ({plot_center_x},{plot_center_y}) r={plane_radius}")
                    shapes_created += 1
                except Exception as e:
                    logger.error(f"Error creating gradient circle {i}: {e}", exc_info=True)
                    error_occurred = True
                    error_message = f"Circle error: {e}"
                for j in range(5):
                    triangle_angle = (2 * math.pi * j) / 5 + math.radians(overall_rotation_angle_deg)
                    triangle_size = plane_radius * 0.2
                    center_x = plot_center_x + plane_radius * 0.7 * math.cos(triangle_angle)
                    center_y = plot_center_y + plane_radius * 0.7 * math.sin(triangle_angle)
                    try:
                        triangle = pyglet_shapes.Triangle(
                            x=center_x,
                            y=center_y - triangle_size / math.sqrt(3),
                            x2=center_x + triangle_size / 2,
                            y2=center_y + triangle_size / (2 * math.sqrt(3)),
                            x3=center_x - triangle_size / 2,
                            y3=center_y + triangle_size / (2 * math.sqrt(3)),
                            color=(100, 100, 200, plane_opacity),
                            batch=batch,
                            group=group_background
                        )
                        triangle.rotation = math.degrees(triangle_angle)
                        self.mandala_pyglet_shapes.append(triangle)
                        logger.info(f"Created triangle {i}-{j} at ({center_x},{center_y}) size={triangle_size}")
                        shapes_created += 1
                    except Exception as e:
                        logger.error(f"Error creating triangle {i}-{j}: {e}", exc_info=True)
                        error_occurred = True
                        error_message = f"Triangle error: {e}"
        except Exception as e:
            logger.error(f"Error in origami planes block: {e}", exc_info=True)
            error_occurred = True
            error_message = f"Planes error: {e}"
        logger.info(f"Total shapes created this frame: {shapes_created}")
        if error_occurred and batch is not None:
            # Display a visible error message in the window
            try:
                error_label = pyglet.text.Label(
                    text=error_message,
                    font_name='Arial',
                    font_size=18,
                    x=plot_center_x,
                    y=window_height * 0.1,
                    anchor_x='center',
                    anchor_y='center',
                    color=(255, 0, 0, 255),
                    batch=batch,
                    group=group_foreground
                )
                self.mandala_pyglet_shapes.append(error_label)
            except Exception as e:
                logger.error(f"Error displaying error label: {e}", exc_info=True)

        # Get driving vector and continue with existing mandala generation
        dv_elements = 10
        dv = self._get_driving_vector(pattern, num_elements=dv_elements)
        # Log the driving vector
        logger.info(f"Driving Vector (dv): {[f'{x:.3f}' for x in dv.tolist()]}")

        # Track and log change in driving_vector
        if self.previous_driving_vector is not None and \
           isinstance(self.previous_driving_vector, torch.Tensor) and \
           self.previous_driving_vector.shape == dv.shape and \
           self.previous_driving_vector.device == dv.device:
            dv_change_magnitude = torch.norm(dv - self.previous_driving_vector).item()
            logger.info(f"Driving Vector Change (L2 Norm): {dv_change_magnitude:.4f}")
        else:
            if self.previous_driving_vector is None:
                logger.info("Driving Vector: Initialized (no previous dv to compare).")
            else:
                logger.warning("Skipping dv change calculation due to shape/device mismatch or invalid type.")
        
        self.previous_driving_vector = dv.clone()

        # Map driving vector components to mandala parameters
        # dv[0] for layers, dv[8] for size modulation
        dv0_raw = dv[0].item()
        dv8_raw = dv[8].item() # Used for size_modulation_factor AND now layer complexity modulation

        # "Bent frame" detour: dv[8] (size tendency) also influences layer complexity range
        layer_dynamic_range_modulator = 2.0 + 8.0 * dv8_raw # Modulator from 2.0 to 10.0 (was 2.0 + 3.0 * dv8_raw)
        num_layers = max(2, int(dv0_raw * layer_dynamic_range_modulator) + 2)
        
        segments_per_layer_base = max(3, int(dv[1].item() * 17) + 3) # 3 to 20 segments (was * 9)
        
        start_hue = dv[2].item() # 0-1 for hue
        hue_increment_per_layer = dv[3].item() * 0.8 - 0.4 # Range -0.4 to 0.4 (was * 0.2 - 0.1)
        saturation_base = dv[4].item() * 0.4 + 0.6 # 0.6 to 1.0
        value_base = dv[5].item() * 0.3 + 0.5 # Brightness: 0.5 to 0.8
        alpha_base = dv[6].item() * 0.3 + 0.4 # Opacity: 0.4 to 0.7 (scaled to 0-255 later)

        shape_type_factor = dv[7].item() # Determines shape type
        size_modulation_factor = dv8_raw * 1.5 + 0.25 # Range 0.25 to 1.75 (was * 0.5 + 0.75)
        rotation_pattern_factor = dv[9].item() # For rotational effects

        # Log some key derived parameters
        logger.info(f"Derived Params: Layers={num_layers}, Hue={start_hue:.3f}, ShapeFactor={shape_type_factor:.3f}")

        for l_idx in range(num_layers):
            layer_radius_norm = (l_idx + 1) / num_layers
            # Modulated radius for the current layer
            radius = layer_radius_norm * base_plot_radius * size_modulation_factor
            
            current_segments = segments_per_layer_base + int(l_idx * dv[1].item() * 1.5) # More segments on outer layers
            if current_segments <= 0: continue

            segment_angle_deg = 360.0 / current_segments
            
            layer_hue = (start_hue + l_idx * hue_increment_per_layer) % 1.0
            layer_saturation = saturation_base * (0.9 + layer_radius_norm * 0.1)  # Increased base saturation
            layer_saturation = min(1.0, max(0.0, layer_saturation))
            layer_value = value_base * (1.0 - layer_radius_norm * 0.3)  # Adjusted value range
            layer_value = min(1.0, max(0.0, layer_value))
            
            rgb_color = self._hsv_to_rgb(layer_hue, layer_saturation, layer_value)
            # Pyglet opacity for shapes is 0-255
            opacity = int((alpha_base * (1.0 - layer_radius_norm * 0.4)) * 255)  # Adjusted opacity range
            opacity = min(255, max(0, opacity))

            for s_idx in range(current_segments):
                angle_offset_deg = rotation_pattern_factor * l_idx * 90 
                # Apply the overall_rotation_angle_deg here
                current_angle_rad = math.radians(s_idx * segment_angle_deg + angle_offset_deg + overall_rotation_angle_deg)
                
                # Position for the center of the element
                element_center_x = plot_center_x + radius * math.cos(current_angle_rad)
                element_center_y = plot_center_y + radius * math.sin(current_angle_rad)

                # Base size for elements in this layer, amplified
                element_size_base = (base_plot_radius / (num_layers * 3.5)) * (0.1 + dv[s_idx % dv_elements].item() * 1.9) # Range 0.1 to 2.0 multiplier (was 0.5 + dv_val * 0.8)
                element_size = max(1, element_size_base) # Ensure minimum size, changed from max(2,...) to allow smaller elements if base is small

                pyglet_shape = None
                current_group = group_foreground

                # Adjust thresholds based on mandala_shape_bias
                # A positive bias leans towards more complex shapes (triangles), negative towards simpler (circles).
                # The effect of the bias is scaled (e.g., by 0.1) to make it a gentle nudge.
                threshold_adjustment = mandala_shape_bias * 0.1
                
                # Initial thresholds for a three-way split (0.33, 0.66)
                base_threshold1 = 1/3
                base_threshold2 = 2/3

                adjusted_threshold1 = max(0.01, min(base_threshold1 - threshold_adjustment, 0.98))
                adjusted_threshold2 = max(adjusted_threshold1 + 0.01, min(base_threshold2 - threshold_adjustment, 0.99))
                
                # Ensure threshold2 is always greater than threshold1
                if adjusted_threshold2 <= adjusted_threshold1:
                    adjusted_threshold2 = adjusted_threshold1 + 0.01 # Maintain a small gap
                    # And re-clamp threshold2 just in case
                    adjusted_threshold2 = min(adjusted_threshold2, 0.99)

                current_shape_factor = shape_type_factor # dv[7].item()

                if current_shape_factor < adjusted_threshold1: # Circles
                    pyglet_shape = pyglet_shapes.Circle(
                        x=element_center_x, y=element_center_y, radius=element_size,
                        color=rgb_color, batch=batch, group=current_group
                    )
                elif current_shape_factor < adjusted_threshold2: # Rectangles
                    # Rectangles anchored at bottom-left, so adjust x,y
                    pyglet_shape = pyglet_shapes.Rectangle(
                        x=element_center_x - element_size / 2, 
                        y=element_center_y - element_size / 2,
                        width=element_size, height=element_size,
                        color=rgb_color, batch=batch, group=current_group
                    )
                    # Rotation for rectangles (and other shapes that support it)
                    if hasattr(pyglet_shape, 'rotation'):
                         pyglet_shape.rotation = dv[(s_idx+1) % dv_elements].item() * 360
                else: # Triangles (equilateral for simplicity)
                    pyglet_shape = pyglet_shapes.Triangle(
                        x=element_center_x, y=element_center_y - element_size / math.sqrt(3),
                        x2=element_center_x + element_size / 2, y2=element_center_y + element_size / (2 * math.sqrt(3)),
                        x3=element_center_x - element_size / 2, y3=element_center_y + element_size / (2 * math.sqrt(3)),
                        color=rgb_color, batch=batch, group=current_group
                    )
                    if hasattr(pyglet_shape, 'rotation'):
                        pyglet_shape.rotation = dv[(s_idx+2) % dv_elements].item() * 360
                
                if pyglet_shape:
                    pyglet_shape.opacity = opacity
                    self.mandala_pyglet_shapes.append(pyglet_shape)

    def create_vector_concretion(self, pattern: torch.Tensor) -> str:
        """Creates an ASCII concretion (symbolic growth) driven by the pattern."""
        dv = self._get_driving_vector(pattern, num_elements=5)

        # Map driving vector components to concretion parameters
        base_symbol_idx = int(dv[0] * (len(self.concretion_symbols)-1))
        growth_iterations = max(3, int(dv[1] * (self.concretion_size / 2))) # e.g., 3 to ~7 iterations
        angle_step_rad = dv[2] * math.pi / 2  # 0 to pi/2 radians step
        scale_factor_mod = dv[3] * 0.5 + 0.8 # 0.8 to 1.3
        symbol_variation_chance = dv[4] # 0 to 1

        grid = [[' ' for _ in range(self.concretion_size)] for _ in range(self.concretion_size)]
        center_x, center_y = self.concretion_size // 2, self.concretion_size // 2

        current_pos = [(center_x, center_y)]
        current_angle = 0.0
        current_scale = 1.0 

        grid[center_y][center_x] = self.concretion_symbols[base_symbol_idx]

        for i in range(growth_iterations):
            new_positions = []
            current_scale *= scale_factor_mod
            if current_scale < 0.5 : current_scale = 0.5 # Min scale
            if current_scale > 2.0 : current_scale = 2.0 # Max scale

            for x_orig, y_orig in current_pos:
                num_branches = max(1, int(dv[1] * 2) + 1) # 1 to 3 branches
                for branch in range(num_branches):
                    branch_angle_offset = (math.pi / 4) * (branch - num_branches // 2) # Spread branches
                    angle = current_angle + angle_step_rad * i + branch_angle_offset
                    
                    # Distance scales with iteration and overall scale_factor_mod
                    distance = (i + 1) * 0.5 * current_scale 

                    x_new = x_orig + int(distance * math.cos(angle))
                    y_new = y_orig + int(distance * math.sin(angle))

                    if 0 <= x_new < self.concretion_size and 0 <= y_new < self.concretion_size:
                        if grid[y_new][x_new] == ' ': # Only place if empty
                            symbol_idx_to_use = base_symbol_idx
                            if torch.rand(1).item() < symbol_variation_chance:
                                symbol_idx_to_use = (base_symbol_idx + i + branch) % len(self.concretion_symbols)
                            grid[y_new][x_new] = self.concretion_symbols[symbol_idx_to_use]
                            new_positions.append((x_new, y_new))
            
            current_pos = list(set(new_positions)) # Unique new positions become origins for next step
            if not current_pos: # Stop if no new positions were added
                break
            current_angle += angle_step_rad / 2 # Slightly alter global angle each iteration

        return "\n".join("".join(row) for row in grid)

# Example Usage (for testing, to be removed or placed in a main block):
if __name__ == '__main__':
    test_pattern = torch.randn(20) # Example pattern tensor
    visualizer = VectorDrivenVisualizer(mandala_size=25, concretion_size=25)
    
    print("Vector-Driven Mandala:")
    mandala_art = visualizer.create_vector_mandala(test_pattern)
    print(mandala_art)
    
    print("\nVector-Driven Concretion:")
    concretion_art = visualizer.create_vector_concretion(test_pattern)
    print(concretion_art) 