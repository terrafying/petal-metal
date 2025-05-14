import torch
import asyncio
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager, SecurityConfig, ResonantMode
from flock_generator import FlockConfig, FlockGenerator, IterationCallback
from shape_visualizer import ShapeVisualizer
from visualization_manager import VectorDrivenVisualizer
import numpy as np
from storyteller import Storyteller
import logging
import pyglet
import pyglet.shapes as pyglet_shapes
import pyglet.graphics
import random # For randomizing star properties
import math # For atan2, degrees for tail rotation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import tempfile
from pathlib import Path
import json
import psutil
import pythonosc.udp_client
import time

logger = logging.getLogger(__name__)

# --- Intriguing Keywords for EntendreGlyphs ---
INTRIGUING_KEYWORDS = [
    "echo", "mirror", "paradox", "cycle", "illusion", "reflect",
    "shadow", "play", "riddle", "enigma", "ephemeral", "trace",
    "whisper", "dream", "flow", "tide", "wave", "hidden"
]

# --- EntendreGlyph Class ---
class EntendreGlyph:
    def __init__(self, text, x, y, batch, group, window_width, window_height, lifetime=3.0, color=(200, 200, 255)):
        self.text = text
        self.x = x
        self.y = y
        self.batch = batch
        self.group = group
        self.window_width = window_width
        self.window_height = window_height
        self.initial_lifetime = lifetime
        self.lifetime = lifetime
        self.color = color # Base color (r, g, b)

        self.label = pyglet.text.Label(
            self.text,
            font_name='Arial',
            font_size=random.uniform(12, 20),
            x=self.x, y=self.y,
            anchor_x='center', anchor_y='center',
            color=(*self.color, 255), # Initial full opacity
            batch=self.batch,
            group=self.group
        )
        logger.debug(f"New EntendreGlyph: '{self.text}' at ({self.x:.0f},{self.y:.0f}), lifetime={self.lifetime:.2f}s")

    def update(self, dt):
        self.lifetime -= dt
        if self.lifetime <= 0:
            return False

        # Fade out opacity based on lifetime
        current_opacity = int(max(0, (self.lifetime / self.initial_lifetime)) * 255)
        self.label.color = (*self.color, current_opacity)
        
        # Optional: subtle drift or other animation could be added here
        # self.label.y -= 5 * dt # Example: slow downward drift

        return True

    def delete(self):
        if self.label:
            self.label.delete()
            self.label = None # Avoid trying to delete again
        logger.info(f"Deleted EntendreGlyph: '{self.text}'")

# --- ShootingStar Class ---
class ShootingStar:
    def __init__(self, x, y, batch, group, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.batch = batch
        self.group = group

        self.x = x
        self.y = y
        
        self.velocity_x = random.uniform(100, 300) * random.choice([-1, 1])
        self.velocity_y = random.uniform(100, 300) * random.choice([-1, 1]) # Made y-velocity potentially faster
        self.lifetime = random.uniform(1.5, 3.5)  # seconds
        self.initial_star_lifetime = self.lifetime # Store initial lifetime for opacity
        
        star_radius = random.uniform(4, 10)
        self.tail_length = random.uniform(50, 150) # Longer tails
        self.tail_width = star_radius * 0.8
        
        r, g, b = random.randint(200, 255), random.randint(200, 255), random.randint(150, 255)
        self.color_head = (r, g, b)
        self.color_tail = (r, g, b) # Tail will use opacity for fading

        self.star_shape = pyglet_shapes.Star(
            self.x, self.y,
            outer_radius=star_radius,
            inner_radius=star_radius * 0.5,
            num_spikes=5,
            color=self.color_head,
            batch=self.batch,
            group=self.group
        )
        
        # Tail is a Rectangle
        self.tail_shape = pyglet_shapes.Rectangle(
            self.x, self.y - self.tail_width / 2, # Initial position, anchor adjusted later
            width=self.tail_length, 
            height=self.tail_width,
            color=self.color_tail, 
            batch=self.batch, 
            group=self.group
        )
        # Set anchor point to the middle of the short edge that connects to the star
        self.tail_shape.anchor_x = 0 # Anchor at the start of the rectangle (length-wise)
        self.tail_shape.anchor_y = self.tail_width / 2 # Anchor in the middle (height-wise)
        
        # Initial rotation based on velocity
        angle_rad = math.atan2(self.velocity_y, self.velocity_x)
        self.tail_shape.rotation = math.degrees(angle_rad)
        self.star_shape.rotation = math.degrees(angle_rad) + 90 # Point star in direction of travel
        logger.debug(f"New Star: ({self.x:.1f},{self.y:.1f}) vx={self.velocity_x:.1f}, vy={self.velocity_y:.1f}, lifetime={self.lifetime:.2f}")

    def update(self, dt):
        self.lifetime -= dt
        if self.lifetime <= 0:
            logger.debug(f"Star lifetime ended: ({self.x:.1f},{self.y:.1f})")
            return False

        dx = self.velocity_x * dt
        dy = self.velocity_y * dt
        
        prev_x, prev_y = self.x, self.y
        self.x += dx
        self.y += dy

        self.star_shape.x = self.x
        self.star_shape.y = self.y
        
        self.tail_shape.x = self.x 
        self.tail_shape.y = self.y 

        # Update rotation for both star and tail
        angle_rad = math.atan2(self.velocity_y, self.velocity_x)
        current_rotation_deg = math.degrees(angle_rad)
        self.tail_shape.rotation = current_rotation_deg
        self.star_shape.rotation = current_rotation_deg + 90 # Keep star pointed
        
        # Fade out opacity based on lifetime
        # Opacity for shapes is 0-255
        # Normalize over its initial lifetime for a more consistent fade
        current_opacity = int(max(0, (self.lifetime / self.initial_star_lifetime)) * 255) 
        self.star_shape.opacity = current_opacity
        self.tail_shape.opacity = int(current_opacity * 0.7) # Tail slightly more transparent

        logger.debug(f"Star @({prev_x:.1f},{prev_y:.1f}) update: dt={dt:.4f}, v=({self.velocity_x:.1f},{self.velocity_y:.1f}), " \
                     f"d=({dx:.2f},{dy:.2f}) -> new_pos=({self.x:.1f},{self.y:.1f}), life={self.lifetime:.2f}")

        if self.x > self.window_width + self.tail_length or self.x < -self.tail_length or \
           self.y > self.window_height + self.tail_length or self.y < -self.tail_length:
            logger.debug(f"Star @({self.x:.1f},{self.y:.1f}) out of bounds.")
            return False

        return True

    def delete(self):
        self.star_shape.delete()
        self.tail_shape.delete()
        logger.info(f"Deleted shooting star at ({self.x:.0f}, {self.y:.0f})")

# Callback function to observe flock state during iterations
def log_flock_iteration_state(iteration_num: int, texts: List[str], tensors: List[torch.Tensor]):
    logger.info(f"--- Flock Iteration {iteration_num} State (Metronome Tick) ---")
    for i, (text, tensor) in enumerate(zip(texts, tensors)):
        # Simulate voicing the line
        logger.info(f"  VOICING (Beat {i+1}): '{text[:120]}...'")
        # Original logging for text and tensor stats
        # logger.info(f"  Item {i}: Text: '{text[:80]}...'") # Keep or remove original if VOICING replaces it
        if isinstance(tensor, torch.Tensor):
            logger.info(f"    Tensor shape: {list(tensor.shape)}, Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        else:
            logger.info(f"    Tensor: Not a valid tensor (type: {type(tensor)})")
    logger.info(f"--- End Flock Iteration {iteration_num} ---")

# --- Pyglet Window and Global State (for pattern_evolution_swarm) ---
# These are global for simplicity in this example. In a larger app, you might encapsulate.
pyglet_window = None
pyglet_batch = None
pyglet_fps_display = None
# Groups for rendering order (optional, but good practice)
pyglet_group_background = None
pyglet_group_foreground = None
pyglet_group_stars = None # New group for stars, potentially rendered on top
pyglet_group_glyphs = None # For EntendreGlyphs

# Shared state for the async update loop and Pyglet
current_pattern_tensor = None
generation_cycle_count = 0
mandala_overall_rotation_angle = 0.0 # For constant global rotation
stop_event = asyncio.Event() # For cleanly stopping the async loop
active_stars = [] # List to hold active ShootingStar objects
active_glyphs = [] # List to hold active EntendreGlyph objects

# Vignette 1: Minimal Language-Specific Swarm
async def language_specific_swarm():
    """Demonstrates a minimal swarm focused on language-specific generation."""
    config = FlockConfig(
        num_models=2,
        language_weights={"en": 1.0, "zh": 0.8},
        language_specialization="en"
    )
    
    generator = FlockGenerator(
        model_names=["gpt2", "distilgpt2"],
        config=config
    )
    
    if generator.models:
        generator.pattern_manager = PatternManager(
            model=generator.models[0].model,
            pattern_depth=3,
            base_seed=42
        )
    else:
        raise ValueError("FlockGenerator has no models loaded.")
    
    prompts = ["Hello", "你好"]
    languages = ["en", "zh"]
    
    # Generate results - expecting List[Tuple[str, torch.Tensor]]
    results_tuples = await generator.generate_batch(
        prompts=prompts,
        languages=languages,
        num_iterations=2
    )
    
    # Extract the text part for original processing
    text_results = [text for text, tensor in results_tuples]
    
    # Visualize language patterns using text_results
    visualizer = ShapeVisualizer()
    embeddings = []
    for result_text in text_results:
        # Process the text result as before
        tokens = generator.models[0].tokenizer(result_text, return_tensors="pt")
        with torch.no_grad():
            emb = generator.models[0].model.get_input_embeddings()(tokens.input_ids)
            emb = emb.squeeze(0).mean(dim=0).unsqueeze(0)
        embeddings.append(emb)
    
    embeddings_tensor = torch.cat(embeddings, dim=0)
    embedding_fig = visualizer.visualize_embedding_space(embeddings_tensor, languages)
    
    # Ensure coupling_matrix exists on pattern_manager before accessing
    coupling_matrix_tensor = None
    if hasattr(generator.pattern_manager, 'coupling_matrix') and generator.pattern_manager.coupling_matrix is not None:
        coupling_matrix_tensor = generator.pattern_manager.coupling_matrix
    else:
        logger.warning("Coupling matrix not found in PatternManager for language_specific_swarm visualization.")
        # Create a dummy matrix if needed for visualization, or handle absence
        # coupling_matrix_tensor = torch.zeros(len(config.language_weights), len(config.language_weights))
    
    language_fig = visualizer.visualize_language_matrix(
        config.language_weights,
        coupling_matrix=coupling_matrix_tensor # Pass the tensor or None
    )
    
    # Return the text results and figures
    return text_results, embedding_fig, language_fig

# --- Async Core Logic for Pattern Evolution ---
async def pattern_evolution_update_cycle(flock_generator: FlockGenerator, 
                                         storyteller: Storyteller, 
                                         vector_viz: VectorDrivenVisualizer,
                                         semantic_prompts_sequence: List[Dict]):
    global current_pattern_tensor, generation_cycle_count, active_glyphs, pyglet_window, pyglet_batch, pyglet_group_glyphs
    
    logger.debug(f"pattern_evolution_update_cycle - START - Gen Cycle {generation_cycle_count + 1}")
    current_batch_generated_items: List[Tuple[str, torch.Tensor]] = []

    # Select the current prompt from the sequence based on generation_cycle_count
    current_prompt_index = generation_cycle_count % len(semantic_prompts_sequence)
    mantra_config = semantic_prompts_sequence[current_prompt_index]
    
    logger.info(f"  Processing mantra: {mantra_config['mantra']} (Cycle {generation_cycle_count + 1})")

    if hasattr(flock_generator.pattern_manager, 'resonant_modes'):
        current_resonant_modes = [
            ResonantMode(
                frequency=np.random.uniform(0.2, 2.5), amplitude=np.random.uniform(0.4, 0.9),
                phase=np.random.uniform(0, 2 * np.pi), decay=np.random.uniform(0.1, 0.7),
                temporal_scale=np.random.uniform(0.5, 2.0), modulation_depth=0.4,
                phase_coupling=0.3, harmonic_order=np.random.randint(1, 4),
                freq_mod_scale=0.2, amp_mod_scale=0.2, depth_mod_scale=0.2, time_mod_scale=0.1
            ) for _ in range(1) 
        ]
        flock_generator.pattern_manager.resonant_modes = current_resonant_modes
    
    generated_items_tuples = await flock_generator.generate_batch(
        prompts=[mantra_config['mantra']],
        num_iterations=1,
        iteration_callback=log_flock_iteration_state
    )
    current_batch_generated_items.extend(generated_items_tuples)

    if current_batch_generated_items:
        generated_text, pattern_tensor_candidate = current_batch_generated_items[-1]
        
        # --- EntendreGlyph Creation ---
        if pyglet_window: # Ensure window exists before creating glyphs
            for keyword in INTRIGUING_KEYWORDS:
                if keyword in generated_text.lower(): # Case-insensitive check
                    glyph_x = random.uniform(pyglet_window.width * 0.1, pyglet_window.width * 0.9)
                    glyph_y = random.uniform(pyglet_window.height * 0.1, pyglet_window.height * 0.9)
                    new_glyph = EntendreGlyph(
                        text=keyword,
                        x=glyph_x, y=glyph_y,
                        batch=pyglet_batch, group=pyglet_group_glyphs,
                        window_width=pyglet_window.width, window_height=pyglet_window.height
                    )
                    active_glyphs.append(new_glyph)
                    logger.info(f"Created EntendreGlyph for keyword: '{keyword}' in text.")
        # --- End EntendreGlyph Creation ---

        if not isinstance(pattern_tensor_candidate, torch.Tensor):
            logger.error(f"Pattern is not a tensor ({type(pattern_tensor_candidate)}), using zero tensor.")
            # Ensure a correctly sized zero tensor if possible
            hidden_size = flock_generator.models[0].model.config.hidden_size if flock_generator.models else 768
            current_pattern_tensor = torch.zeros(hidden_size, device=flock_generator.models[0].device if flock_generator.models else 'cpu')
        else:
            alpha = flock_generator.config.mandala_flow_alpha
            # Ensure current_pattern_tensor is compatible for interpolation
            if current_pattern_tensor is None or \
               not isinstance(current_pattern_tensor, torch.Tensor) or \
               current_pattern_tensor.shape != pattern_tensor_candidate.shape or \
               current_pattern_tensor.device != pattern_tensor_candidate.device:
                # This is the first-time assignment or a reset due to incompatibility
                current_pattern_tensor = pattern_tensor_candidate.clone() 
                logger.info("Pattern Tensor: Initialized/Reset.")
            else:
                # Regular update with interpolation
                tensor_at_cycle_start = current_pattern_tensor.clone()
                
                current_pattern_tensor = (1 - alpha) * current_pattern_tensor + alpha * pattern_tensor_candidate
                
                pattern_change_magnitude = torch.norm(current_pattern_tensor - tensor_at_cycle_start).item()
                logger.info(f"Pattern Tensor Change (L2 Norm): {pattern_change_magnitude:.4f}")
        
        logger.debug(f"  Final generated text for pattern: {generated_text[:100]}...")
        
        # Storyteller update (mandala/concretion are now visual, not ASCII)
        storyteller.add_thread(
            pattern=current_pattern_tensor,
            mandala="[Pyglet Geometric Mandala Displayed]",
            concretion="[Pyglet Concretion Pending]",
            context={
                "generation_cycle": generation_cycle_count,
                "mantra_config": mantra_config,
                "pattern_shape": list(current_pattern_tensor.shape) if isinstance(current_pattern_tensor, torch.Tensor) else "N/A",
                "generated_text_preview": generated_text[:50]
            }
        )
    else:
        logger.warning("No items generated in this cycle to display.")
        # If no tensor, keep the previous one or use a default for continuous viz
        if current_pattern_tensor is None and flock_generator.models:
             hidden_size = flock_generator.models[0].model.config.hidden_size
             current_pattern_tensor = torch.zeros(hidden_size, device=flock_generator.models[0].device)


    # Narrative generation and logging (as before)
    narrative = storyteller.generate_narrative()
    stats = storyteller.get_narrative_stats()
    logger.info(f"Narrative stats for Generation Cycle {generation_cycle_count + 1}:")
    logger.info(f"  Total threads: {stats['total_threads']}")
    logger.info(f"  Thematic distribution: {stats.get('thematic_distribution', {})}")
    logger.info(f"  Emotional range: {stats.get('emotional_range', [])}")
    logger.info(f"  Average complexity: {stats.get('average_complexity', 0.0):.2f}")
    logger.info(f"  Average depth: {stats.get('average_depth', 0.0):.2f}")
    logger.info(f"  Temporal coherence: {stats.get('temporal_coherence', 0.0):.2f}")

    generation_cycle_count += 1
    if generation_cycle_count >= 100: # Max cycles condition
        stop_event.set()

    logger.debug(f"pattern_evolution_update_cycle - END - Gen Cycle {generation_cycle_count}")

# Vignette 2: Pattern Evolution Swarm (Pyglet Version)
async def pattern_evolution_swarm():
    """Run pattern evolution swarm with Pyglet-driven visuals."""
    global pyglet_window, pyglet_batch, pyglet_fps_display, current_pattern_tensor, generation_cycle_count
    global pyglet_group_background, pyglet_group_foreground, pyglet_group_stars, pyglet_group_glyphs, stop_event, active_stars, active_glyphs

    # Get the currently running asyncio event loop
    loop = asyncio.get_running_loop()

    stop_event.clear() # Ensure cleared at the very start
    generation_cycle_count = 0
    active_stars = []
    active_glyphs = [] # Initialize our new list

    model_configs = [
        {"name": "distilgpt2", "params": 82, "memory": 500, "speed": "fast"},
        {"name": "gpt2", "params": 124, "memory": 800, "speed": "medium"}
    ]
    selected_models = [config["name"] for config in model_configs]
    flock_config = FlockConfig(
        num_models=len(selected_models), pattern_depth=8, batch_size=53, max_length=41,
        flock_cohesion=0.3, flock_alignment=0.2,
        flock_separation=0.4, temperature=0.85,
        mandala_flow_alpha=0.3
    )
    flock_gen = FlockGenerator(selected_models, config=flock_config)
    story_tell = Storyteller(flock_gen.pattern_manager)
    vector_vis = VectorDrivenVisualizer(mandala_size=20, concretion_size=20)
    
    semantic_flow_prompts = [
        {"mantra": "dinosaur", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "meteor", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "apotheosis", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "plastic toy", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "chicken", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "egg", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "concubine", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "harlem", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "shaker", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "mover", "temporal_scale": 1.0, "frequency": 1.0},
        {"mantra": "candlestick", "temporal_scale": 1.0, "frequency": 1.0}
    ]

    # --- Pyglet Setup ---
    try:
        # Configure OpenGL settings before creating window
        
        pyglet_window = pyglet.window.Window(
            width=800, 
            height=800, 
            caption='Pattern Evolution (Pyglet)', 
            resizable=True,
            vsync=True  # Enable vertical sync
        )
        
        # Enable alpha blending and set clear color
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.gl.glClearColor(0.02, 0.02, 0.05, 1.0)  # Very dark blue-black
        
        # Create batch and groups
        pyglet_batch = pyglet.graphics.Batch()
        pyglet_fps_display = pyglet.window.FPSDisplay(window=pyglet_window)
        
        pyglet_group_background = pyglet.graphics.Group(order=0)
        pyglet_group_foreground = pyglet.graphics.Group(order=1)
        pyglet_group_stars = pyglet.graphics.Group(order=2)
        pyglet_group_glyphs = pyglet.graphics.Group(order=3)

        mouse_is_dragging = False

        hidden_size = flock_gen.models[0].model.config.hidden_size if flock_gen.models else 768
        device_to_use = flock_gen.models[0].device if flock_gen.models else 'cpu'
        current_pattern_tensor = torch.zeros(hidden_size, device=device_to_use)
        
        vector_vis.update_geometric_mandala_pyglet(
            current_pattern_tensor, pyglet_batch,
            group_foreground=pyglet_group_foreground,
            group_background=pyglet_group_background,
            window_width=pyglet_window.width, window_height=pyglet_window.height,
            mandala_shape_bias=flock_gen.config.mandala_shape_bias
        )

        if pyglet_window and pyglet_batch and pyglet_group_stars:
            initial_star_x = random.uniform(0, pyglet_window.width)
            initial_star_y = pyglet_window.height * 0.8 # Lowered initial Y spawn point
            new_star = ShootingStar(initial_star_x, initial_star_y, pyglet_batch, pyglet_group_stars, pyglet_window.width, pyglet_window.height)
            active_stars.append(new_star)
            logger.info(f"Spawned initial shooting star at ({initial_star_x:.0f}, {initial_star_y:.0f})")

            # --- TEMPORARY TEST: Force a glyph to be created ---
            if pyglet_group_glyphs: # Ensure glyph group exists
                test_glyph = EntendreGlyph(
                    text="TEST_GLYPH",
                    x=pyglet_window.width / 2,
                    y=pyglet_window.height / 2,
                    batch=pyglet_batch, group=pyglet_group_glyphs,
                    window_width=pyglet_window.width, window_height=pyglet_window.height,
                    lifetime=10.0 # Long lifetime for testing
                )
                active_glyphs.append(test_glyph)
                logger.info("Spawned TEMPORARY TEST_GLYPH")
            # --- END TEMPORARY TEST ---

    except Exception as e:
        logger.error(f"Pyglet window initialization failed: {e}", exc_info=e)
        raise e

    @pyglet_window.event
    def on_draw():
        logger.debug("on_draw event fired.")
        try:
            pyglet_window.clear()
            pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
            pyglet_batch.draw()
            pyglet_fps_display.draw()
            logger.debug("Frame drawn successfully.")
        except Exception as e:
            logger.error(f"Error in on_draw: {e}", exc_info=True)

    @pyglet_window.event
    def on_resize(width, height):
        logger.info(f"Pyglet window resized to: {width}x{height}")
        if current_pattern_tensor is not None:
             vector_vis.update_geometric_mandala_pyglet(
                current_pattern_tensor, pyglet_batch,
                group_foreground=pyglet_group_foreground,
                group_background=pyglet_group_background,
                window_width=width, window_height=height,
                mandala_shape_bias=flock_gen.config.mandala_shape_bias
            )
        return pyglet.event.EVENT_HANDLED

    @pyglet_window.event
    def on_mouse_press(x, y, button, modifiers):
        nonlocal mouse_is_dragging
        if button == pyglet.window.mouse.LEFT:
            mouse_is_dragging = True
            logger.info(f"Mouse Press: LEFT at ({x},{y})")

    @pyglet_window.event
    def on_mouse_release(x, y, button, modifiers):
        nonlocal mouse_is_dragging
        if button == pyglet.window.mouse.LEFT:
            mouse_is_dragging = False
            logger.info(f"Mouse Release: LEFT at ({x},{y})")

    @pyglet_window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        nonlocal mouse_is_dragging
        global current_pattern_tensor
        if mouse_is_dragging and (buttons & pyglet.window.mouse.LEFT):
            if current_pattern_tensor is not None and isinstance(current_pattern_tensor, torch.Tensor):
                perturb_scale = 0.005
                idx1, idx2 = 0, 1 # Indices to perturb along the last dimension

                # Check if the last dimension is large enough for the chosen indices
                if current_pattern_tensor.shape[-1] > max(idx1, idx2):
                    perturbation_addition_tensor = torch.zeros_like(current_pattern_tensor)
                    applied_perturbation = False
                    
                    if current_pattern_tensor.ndim == 1:
                        # Tensor is 1D, e.g. (H,)
                        perturbation_addition_tensor[idx1] += dx * perturb_scale
                        perturbation_addition_tensor[idx2] += dy * perturb_scale
                        applied_perturbation = True
                    elif current_pattern_tensor.ndim == 2 and current_pattern_tensor.shape[0] == 1:
                        # Tensor is 2D with shape (1, H)
                        perturbation_addition_tensor[0, idx1] += dx * perturb_scale
                        perturbation_addition_tensor[0, idx2] += dy * perturb_scale
                        applied_perturbation = True
                    
                    if applied_perturbation:
                        current_pattern_tensor.add_(perturbation_addition_tensor)
                        logger.info(f"Mouse Drag: Perturbed tensor by ({dx*perturb_scale:.4f}, {dy*perturb_scale:.4f}) on indices {idx1}, {idx2} of the pattern.")
                    else:
                        logger.warning(f"Mouse Drag: Tensor has unhandled shape {current_pattern_tensor.shape} for perturbation. current_pattern_tensor was not modified.")
                else:
                    logger.warning(f"Mouse Drag: Tensor's last dimension (size {current_pattern_tensor.shape[-1]}) too small to perturb chosen indices {idx1}, {idx2}. current_pattern_tensor was not modified.")
            else:
                logger.warning("Mouse Drag: current_pattern_tensor is None or not a Tensor.")

    @pyglet_window.event
    def on_key_press(symbol, modifiers):
        global active_stars
        if symbol == pyglet.window.key.S:
            if pyglet_window and pyglet_batch and pyglet_group_stars:
                spawn_x = random.uniform(0, pyglet_window.width)
                spawn_y = random.uniform(pyglet_window.height * 0.7, pyglet_window.height)
                new_star = ShootingStar(spawn_x, spawn_y, pyglet_batch, pyglet_group_stars, pyglet_window.width, pyglet_window.height)
                active_stars.append(new_star)
                logger.info(f"Spawned new star via keypress at ({spawn_x:.0f}, {spawn_y:.0f})")

    async def scheduled_update(dt):
        global mandala_overall_rotation_angle, active_stars, active_glyphs
        logger.debug(f"Scheduled_update - START - dt_received: {dt:.4f}, cycle: {generation_cycle_count}, Active Stars: {len(active_stars)}, Active Glyphs: {len(active_glyphs)}")
        if stop_event.is_set():
            logger.info("Stop event received in scheduled_update, closing Pyglet window.")
            if pyglet_window and not pyglet_window.has_exit:
                pyglet_window.close()
            return

        # Track memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"CPU Memory - RSS: {memory_info.rss / 1024**2:.1f}MB, VMS: {memory_info.vms / 1024**2:.1f}MB")

        mandala_overall_rotation_angle += 0.5
        if mandala_overall_rotation_angle > 360: mandala_overall_rotation_angle -= 360

        # Run pattern evolution in a separate task to prevent blocking
        try:
            pattern_task = asyncio.create_task(
                pattern_evolution_update_cycle(flock_gen, story_tell, vector_vis, semantic_flow_prompts)
            )
            await asyncio.wait_for(pattern_task, timeout=0.1)  # 100ms timeout
        except asyncio.TimeoutError:
            logger.warning("Pattern evolution cycle took too long, continuing with visualization")
        except Exception as e:
            logger.error(f"Error in pattern evolution cycle: {e}", exc_info=True)
        
        if current_pattern_tensor is not None:
            tensor_mean = current_pattern_tensor.mean().item() if isinstance(current_pattern_tensor, torch.Tensor) else 'N/A'
            tensor_std = current_pattern_tensor.std().item() if isinstance(current_pattern_tensor, torch.Tensor) else 'N/A'
            logger.debug(f"Scheduled_update - Updating visuals with tensor (mean: {tensor_mean:.4f}, std: {tensor_std:.4f}), Overall Rotation: {mandala_overall_rotation_angle:.2f}")
            
            # Update visualization in a non-blocking way
            try:
                vector_vis.update_geometric_mandala_pyglet(
                    current_pattern_tensor, pyglet_batch,
                    group_foreground=pyglet_group_foreground,
                    group_background=pyglet_group_background,
                    window_width=pyglet_window.width, window_height=pyglet_window.height,
                    mandala_shape_bias=flock_gen.config.mandala_shape_bias,
                    overall_rotation_angle_deg=mandala_overall_rotation_angle
                )
            except Exception as e:
                logger.error(f"Error updating mandala visualization: {e}", exc_info=True)
        
        # Update stars and glyphs with error handling
        try:
            stars_to_remove = []
            for star in active_stars:
                if not star.update(dt):
                    stars_to_remove.append(star)
            
            for star in stars_to_remove:
                star.delete()
                active_stars.remove(star)
            
            glyphs_to_remove = []
            for glyph in active_glyphs:
                if not glyph.update(dt):
                    glyphs_to_remove.append(glyph)
            
            for glyph in glyphs_to_remove:
                glyph.delete()
                active_glyphs.remove(glyph)
        except Exception as e:
            logger.error(f"Error updating stars/glyphs: {e}", exc_info=True)
            
        logger.debug(f"Scheduled_update - END - cycle: {generation_cycle_count-1}, Active Stars: {len(active_stars)}, Active Glyphs: {len(active_glyphs)}")

    # Add memory tracking to the main loop
    try:
        while not pyglet_window.has_exit and not stop_event.is_set():
            # Process Pyglet events first
            pyglet.clock.tick()
            pyglet_window.dispatch_events()
            
            # Run our async update with timeout
            try:
                await asyncio.wait_for(scheduled_update(0.016), timeout=0.05)  # 50ms timeout
            except asyncio.TimeoutError:
                logger.warning("Scheduled update took too long, skipping frame")
                continue
            except Exception as e:
                logger.error(f"Error in scheduled update: {e}", exc_info=True)
                continue
            
            # Draw the frame
            try:
                pyglet_window.clear()
                pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
                pyglet_batch.draw()
                pyglet_fps_display.draw()
                pyglet_window.flip()
            except Exception as e:
                logger.error(f"Error drawing frame: {e}", exc_info=True)
                continue
            
            # Adaptive sleep based on frame time
            frame_time = pyglet.clock.get_fps()
            if frame_time > 0:
                sleep_time = max(0.001, 1.0/frame_time - 0.016)  # Target 60 FPS
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error in Pyglet/asyncio main loop: {e}", exc_info=True)
        stop_event.set()
    finally:
        # Ensure Pyglet window is closed if it hasn't been already (e.g., if loop exited due to stop_event)
        if pyglet_window and not pyglet_window.has_exit:
            pyglet_window.close() # This should make pyglet_window.has_exit true
        
        logger.info("Pyglet/asyncio integrated event loop terminated.")
    
    logger.info("Pattern evolution swarm (Pyglet) completed or exited main loop section.") # Clarified log
    # Ensure any remaining asyncio tasks are cancelled if pyglet exits
    # and stop_event wasn't the cause (e.g. window closed manually)
    if not stop_event.is_set():
        stop_event.set() # Signal any lingering async tasks to stop
    
    # Cancel all outstanding asyncio tasks
    tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
        for task in tasks:
            task.cancel()
        # Allow cancellations to propagate
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Asyncio tasks cancelled.")

# Vignette 3: Resource-Aware Swarm
async def resource_aware_swarm():
    """Demonstrates a swarm that adapts to resource constraints."""
    config = FlockConfig(
        num_models=2,
        batch_size=4,
        max_length=128,
        temperature=0.6
    )
    
    generator = FlockGenerator(
        model_names=["distilgpt2", "gpt2"],
        config=config
    )
    
    if generator.models:
        generator.pattern_manager = PatternManager(
            model=generator.models[0].model,
            pattern_depth=3,
            base_seed=42
        )
    else:
        raise ValueError("FlockGenerator has no models loaded.")
    
    prompts = ["Efficient generation:", "Resource aware:", "Optimized:", "Minimal:"]
    languages = ["en"] * len(prompts)
    
    # Generate results - expecting List[Tuple[str, torch.Tensor]]
    results_tuples = await generator.generate_batch(
        prompts=prompts,
        num_iterations=2
        # languages defaults to config.language_specialization which is "en"
    )
    
    # Extract the text part
    text_results = [text for text, tensor in results_tuples]

    # Get swarm dynamics (this part seems unrelated to generate_batch results)
    positions = []
    velocities = []
    if hasattr(generator.pattern_manager, 'swarm_state'):
        for node in generator.pattern_manager.swarm_state.nodes:
            pos = node.position[:3] if len(node.position) > 3 else node.position
            vel = node.velocity[:3] if len(node.velocity) > 3 else node.velocity
            positions.append(pos)
            velocities.append(vel)
    else:
        logger.warning("Swarm state not found in PatternManager for resource_aware_swarm.")
    
    # Visualize swarm dynamics
    visualizer = ShapeVisualizer()
    # Pass languages used for generation if needed by visualizer
    swarm_fig = visualizer.visualize_swarm_dynamics(positions, velocities, languages) 
    
    # Return text results and figure
    return text_results, swarm_fig

# Vignette 4: Emotional Resonance Swarm
async def emotional_resonance_swarm():
    """Demonstrates a swarm that focuses on emotional resonance and coherence."""
    config = FlockConfig(
        num_models=12,
        flock_alignment=0.87,
        flock_cohesion=0.68,
        flock_separation=0.13
    )
    
    generator = FlockGenerator(
        model_names=["gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M"],
        config=config
    )
    
    if generator.models:
        generator.pattern_manager = PatternManager(
            model=generator.models[0].model,
            pattern_depth=8,
            base_seed=53
        )
    else:
        raise ValueError("FlockGenerator has no models loaded.")

    emotional_prompts = [
        "The joy of discovery:", "The wonder of creation:",
        "The harmony of nature:", "The beauty of simplicity:"
    ]
    
    emotional_states = []
    timesteps = []
    exegetical_states = [] # Store final text results across iterations
    
    for i in range(3):
        # Generate results - expecting List[Tuple[str, torch.Tensor]]
        results_tuples = await generator.generate_batch(
            prompts=emotional_prompts,
            num_iterations=2 # Single flocking iteration per loop
        )
        
        # Extract text results for this iteration
        current_text_results = [text for text, tensor in results_tuples]
        if i == 2: # Keep results from the last iteration
            exegetical_states = current_text_results

        # Record emotional state (this part seems independent of generate_batch results)
        if hasattr(generator.pattern_manager, 'emotional_state'):
            # Ensure emotional_state components are tensors before calling item()
            current_emotional_state = {}
            for key, value in generator.pattern_manager.emotional_state.items():
                if isinstance(value, torch.Tensor):
                     # Decide how to store the state: mean, first element, or full tensor?
                     # Storing mean for compatibility with visualizer
                    current_emotional_state[key] = value.mean().item()
                else:
                    current_emotional_state[key] = float(value) # Assume float otherwise
            emotional_states.append(current_emotional_state)
        else:
             logger.warning("Emotional state not found in PatternManager for emotional_resonance_swarm.")
             # Append dummy state if needed
             emotional_states.append({"intensity": 0.5, "coherence": 0.5, "complexity": 0.5})

        timesteps.append(i)
    
    # Visualize emotional evolution
    visualizer = ShapeVisualizer()
    # visualize_emotional_state expects list of dicts with float values, which we prepared
    emotional_fig = visualizer.visualize_emotional_state(emotional_states, timesteps)
    
    # Return final text results and figure
    return exegetical_states, emotional_fig

# Vignette 5: Cross-Language Harmony Swarm
async def cross_language_harmony_swarm():
    """Demonstrates a swarm that creates harmony between different languages."""
    config = FlockConfig(
        num_models=3,
        language_weights={
            "en": 1.0,
            "zh": 0.9,
            "ja": 0.9
        },
        flock_alignment=0.7,
        flock_cohesion=0.6
    )
    
    generator = FlockGenerator(
        model_names=["gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M"],
        config=config
    )
    
    if generator.models:
        generator.pattern_manager = PatternManager(
            model=generator.models[0].model,
            pattern_depth=3,
            base_seed=42
        )
    else:
        raise ValueError("FlockGenerator has no models loaded.")
    
    multilingual_prompts = ["Harmony in diversity:", "Unity in language:", "Cross-cultural bridge:"]
    languages = ["en", "zh", "ja"]
    
    # Generate results - expecting List[Tuple[str, torch.Tensor]]
    results_tuples = await generator.generate_batch(
        prompts=multilingual_prompts,
        languages=languages,
        num_iterations=3
    )
    
    # Extract text results
    text_results = [text for text, tensor in results_tuples]
    
    # Get embeddings for visualization from text_results
    embeddings = []
    for result_text in text_results:
        # Process the text result as before
        tokens = generator.models[0].tokenizer(result_text, return_tensors="pt")
        with torch.no_grad():
            emb = generator.models[0].model.get_input_embeddings()(tokens.input_ids)
            emb = emb.squeeze(0).mean(dim=0).unsqueeze(0)
        embeddings.append(emb)
        
    embeddings_tensor = torch.cat(embeddings, dim=0)
    
    visualizer = ShapeVisualizer()
    embedding_fig = visualizer.visualize_embedding_space(embeddings_tensor, languages)
    
    coupling_matrix_tensor = None
    if hasattr(generator.pattern_manager, 'coupling_matrix') and generator.pattern_manager.coupling_matrix is not None:
        coupling_matrix_tensor = generator.pattern_manager.coupling_matrix
    else:
        logger.warning("Coupling matrix not found in PatternManager for cross_language_harmony_swarm visualization.")
        # coupling_matrix_tensor = torch.zeros(len(config.language_weights), len(config.language_weights))

    language_fig = visualizer.visualize_language_matrix(
        config.language_weights,
        coupling_matrix=coupling_matrix_tensor # Pass the tensor or None
    )
    
    # Return text results and figures
    return text_results, embedding_fig, language_fig

class OSCVisualizationBridge:
    def __init__(self, ip="127.0.0.1", port=8000):
        self.client = pythonosc.udp_client.SimpleUDPClient(ip, port)
        self.last_send_time = 0
        self.send_interval = 1/30  # 30 Hz update rate
        
    def send_visualization_data(self, 
                              voice_states: List[torch.Tensor],
                              voice_projections: List[torch.Tensor],
                              connection_strengths: np.ndarray,
                              reflection_magnitudes: List[float],
                              overall_rotation: float):
        """Send visualization data to Max/MSP/Jitter via OSC."""
        current_time = time.time()
        if current_time - self.last_send_time < self.send_interval:
            return
            
        # Send voice states as matrices
        for i, state in enumerate(voice_states):
            state_np = state.detach().cpu().numpy()
            self.client.send_message(f"/petals/voice/{i}/state", state_np.tolist())
            
        # Send projections
        for i, proj in enumerate(voice_projections):
            self.client.send_message(f"/petals/voice/{i}/projection", float(proj))
            
        # Send connection strengths matrix
        self.client.send_message("/petals/connections", connection_strengths.tolist())
        
        # Send reflection magnitudes
        for i, mag in enumerate(reflection_magnitudes):
            self.client.send_message(f"/petals/voice/{i}/reflection", float(mag))
            
        # Send overall rotation
        self.client.send_message("/petals/rotation", float(overall_rotation))
        
        self.last_send_time = current_time

async def geometric_reflection_game(save_video: bool = True, video_path: Optional[str] = None):
    """Implements a geometric reflection game using an icosahedral lattice of voices."""
    # Initialize OSC bridge
    osc_bridge = OSCVisualizationBridge()
    
    # Initialize with a single model for the base generation
    config = FlockConfig(
        num_models=1,  # We'll handle the 30 voices geometrically
        batch_size=1,
        max_length=128,
        temperature=0.7,
        pattern_depth=8,  # Deeper pattern depth for more complex reflections
        flock_alignment=0.87,  # High alignment for coherent reflections
        flock_cohesion=0.68,
        flock_separation=0.13
    )
    
    generator = FlockGenerator(
        model_names=["gpt2"],  # Single base model
        config=config
    )
    
    if not generator.models:
        raise ValueError("FlockGenerator has no models loaded.")
    
    # Initialize pattern manager with resonant modes for geometric structure
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=config.pattern_depth,
        base_seed=42
    )
    
    # Create icosahedral lattice of 30 voices
    # Using golden ratio for icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    
    # Generate the 12 vertices of an icosahedron
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices.append(np.array([x, y, z]))
    
    for i in range(3):
        for j in range(2):
            v = np.zeros(3)
            v[i] = phi
            v[(i + 1) % 3] = (-1) ** j
            vertices.append(v)
            v = np.zeros(3)
            v[i] = -phi
            v[(i + 1) % 3] = (-1) ** j
            vertices.append(v)
    
    # Normalize vertices to unit sphere
    vertices = [v / np.linalg.norm(v) for v in vertices]
    
    # Create 30 voices by interpolating between vertices
    voices = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if np.dot(vertices[i], vertices[j]) > 0.5:  # Only connect nearby vertices
                mid = (vertices[i] + vertices[j]) / 2
                voices.append(mid / np.linalg.norm(mid))
    
    # Ensure we have exactly 30 voices
    while len(voices) < 30:
        # Add interpolated points between existing voices
        i, j = np.random.choice(len(voices), 2, replace=False)
        mid = (voices[i] + voices[j]) / 2
        voices.append(mid / np.linalg.norm(mid))
    voices = voices[:30]  # Trim to exactly 30
    
    # Initialize voice states with random vectors in the embedding space
    hidden_size = generator.models[0].model.config.hidden_size
    voice_states = []
    for _ in range(30):
        # Create random vector in the embedding space
        random_vec = torch.randn(hidden_size)
        random_vec = random_vec / random_vec.norm()  # Normalize
        voice_states.append(random_vec)
    
    # Initialize dynamic connections
    # Each voice can have up to 5 connections, but they can change over time
    voice_connections = [[] for _ in range(30)]
    connection_strengths = np.zeros((30, 30))  # Track connection strengths
    
    # Initial connections based on geometric proximity
    for i, voice in enumerate(voices):
        distances = [np.arccos(np.clip(np.dot(voice, other), -1.0, 1.0)) for other in voices]
        # Get indices of 5 nearest neighbors (excluding self)
        neighbor_indices = np.argsort(distances)[1:6]  # Skip first (self)
        voice_connections[i] = neighbor_indices.tolist()
        # Initialize connection strengths based on geometric proximity
        for j in neighbor_indices:
            connection_strengths[i, j] = 1.0 - distances[j] / np.pi
    
    # Run IX tokens of generation with reflections
    IX = 10  # Number of iterations
    reflection_results = []
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots for different visualizations
    ax1 = fig.add_subplot(221, projection='3d')  # Voice positions
    ax2 = fig.add_subplot(222)  # Projection strengths
    ax3 = fig.add_subplot(223)  # Reflection magnitudes
    ax4 = fig.add_subplot(224)  # Connection evolution
    
    # Initialize animation writer if saving video
    writer = None
    if save_video:
        if video_path is None:
            video_path = "geometric_reflection_game.mp4"
        writer = FFMpegWriter(fps=12, bitrate=2000)
        writer.setup(fig, video_path, dpi=100)
    
    try:
        for iteration in range(IX):
            logger.info(f"Geometric Reflection Game - Iteration {iteration + 1}/{IX}")
            
            # Generate base text from the initial random vector
            base_prompt = "Reflect on the nature of consciousness:"
            results = await generator.generate_batch(
                prompts=[base_prompt],
                num_iterations=1
            )
            
            if not results:
                logger.warning("No results generated in iteration {iteration + 1}")
                continue
                
            base_text, base_tensor = results[0]
            
            # Project base tensor onto each voice's space
            voice_projections = []
            for i, voice in enumerate(voices):
                # Debug original shapes
                logger.debug(f"base_tensor shape: {base_tensor.shape}")
                logger.debug(f"voice shape: {voice.shape}")
                
                # Convert voice direction to tensor space, but keep it raw
                voice_tensor = torch.tensor(voice, dtype=base_tensor.dtype, device=base_tensor.device)
                logger.debug(f"voice_tensor shape after conversion: {voice_tensor.shape}")
                
                # Work with the raw base tensor - no PCA, no averaging
                # Just reshape to match dimensions for the dot product
                base_tensor_flat = base_tensor.reshape(-1)  # Flatten everything
                
                # Ensure voice tensor is properly sized for projection
                if base_tensor_flat.shape[0] % 3 == 0:
                    # If base tensor is divisible by 3, repeat voice tensor accordingly
                    voice_tensor_flat = voice_tensor.repeat(base_tensor_flat.shape[0] // 3)
                else:
                    # If not divisible by 3, pad the voice tensor to match
                    padding_size = base_tensor_flat.shape[0] - (base_tensor_flat.shape[0] // 3) * 3
                    voice_tensor_flat = torch.cat([
                        voice_tensor.repeat(base_tensor_flat.shape[0] // 3),
                        voice_tensor[:padding_size]
                    ])
                
                # Debug shapes
                logger.debug(f"base_tensor_flat shape: {base_tensor_flat.shape}")
                logger.debug(f"voice_tensor_flat shape: {voice_tensor_flat.shape}")
                
                try:
                    # Compute raw projection
                    projection = torch.dot(base_tensor_flat, voice_tensor_flat)
                    logger.debug(f"projection shape: {projection.shape}")
                    voice_projections.append(projection)
                except RuntimeError as e:
                    logger.error(f"Error in projection calculation: {e}")
                    # Fallback to a simple cosine similarity if dot product fails
                    projection = torch.nn.functional.cosine_similarity(
                        base_tensor_flat.unsqueeze(0),
                        voice_tensor_flat.unsqueeze(0)
                    )
                    voice_projections.append(projection)
            
            # Update connections based on state similarity
            for i in range(30):
                for j in range(30):
                    if i != j:
                        # Compute cosine similarity between voice states
                        similarity = torch.dot(voice_states[i], voice_states[j]).item()
                        # Update connection strength with momentum
                        connection_strengths[i, j] = 0.7 * connection_strengths[i, j] + 0.3 * similarity
            
            # Prune weak connections and form new ones
            for i in range(30):
                # Get current connections
                current_connections = voice_connections[i]
                
                # Prune weak connections
                current_connections = [j for j in current_connections 
                                    if connection_strengths[i, j] > 0.3]
                
                # Find potential new connections
                potential_connections = []
                for j in range(30):
                    if j != i and j not in current_connections:
                        potential_connections.append((j, connection_strengths[i, j]))
                
                # Sort by strength and add strongest new connections
                potential_connections.sort(key=lambda x: x[1], reverse=True)
                while len(current_connections) < 5 and potential_connections:
                    new_conn, strength = potential_connections.pop(0)
                    if strength > 0.5:  # Only form strong connections
                        current_connections.append(new_conn)
                
                # Update connections
                voice_connections[i] = current_connections
            
            # Perform reflections between neighboring voices
            new_voice_states = []
            reflection_magnitudes = []
            
            for i, voice in enumerate(voices):
                # Get current voice state
                current_state = voice_states[i]
                
                # Get states of neighbors
                neighbor_states = [voice_states[j] for j in voice_connections[i]]
                
                # Calculate reflection with weighted connections
                reflection = torch.zeros_like(current_state)
                total_weight = 0
                for j, neighbor_state in enumerate(neighbor_states):
                    weight = connection_strengths[i, voice_connections[i][j]]
                    reflection += weight * (current_state + neighbor_state) / 2
                    total_weight += weight
                
                if total_weight > 0:
                    reflection = reflection / total_weight
                
                # Calculate reflection magnitude
                reflection_magnitude = torch.norm(reflection - current_state).item()
                reflection_magnitudes.append(reflection_magnitude)
                
                # Normalize reflection
                reflection = reflection / reflection.norm()
                
                # Update voice state with reflection
                new_state = (current_state + reflection) / 2
                new_state = new_state / new_state.norm()
                new_voice_states.append(new_state)
            
            # Update voice states
            voice_states = new_voice_states
            
            # Store results
            reflection_results.append({
                'iteration': iteration + 1,
                'base_text': base_text,
                'voice_projections': voice_projections,
                'voice_states': voice_states,
                'reflection_magnitudes': reflection_magnitudes,
                'connection_strengths': connection_strengths.copy()
            })
            
            # Update visualizations
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # Plot 1: Voice positions and connections
            for i, voice in enumerate(voices):
                # Color based on projection strength
                color = plt.cm.viridis(voice_projections[i].item())
                ax1.scatter(*voice, c=[color], alpha=0.6)
                # Draw connections to neighbors with strength-based opacity
                for j in voice_connections[i]:
                    strength = connection_strengths[i, j]
                    ax1.plot([voice[0], voices[j][0]],
                            [voice[1], voices[j][1]],
                            [voice[2], voices[j][2]], 
                            'gray', alpha=0.3 * strength)
            ax1.set_title('Voice Positions and Connections')
            
            # Plot 2: Projection strengths
            ax2.bar(range(len(voice_projections)), [p.item() for p in voice_projections])
            ax2.set_title('Projection Strengths')
            ax2.set_xlabel('Voice Index')
            ax2.set_ylabel('Projection Strength')
            
            # Plot 3: Reflection magnitudes
            ax3.bar(range(len(reflection_magnitudes)), reflection_magnitudes)
            ax3.set_title('Reflection Magnitudes')
            ax3.set_xlabel('Voice Index')
            ax3.set_ylabel('Reflection Magnitude')
            
            # Plot 4: Connection evolution
            if iteration > 0:
                prev_strengths = reflection_results[-2]['connection_strengths']
                strength_changes = np.abs(connection_strengths - prev_strengths)
                ax4.imshow(strength_changes, cmap='viridis')
                ax4.set_title('Connection Strength Changes')
                ax4.set_xlabel('Voice Index')
                ax4.set_ylabel('Voice Index')
                plt.colorbar(ax4.images[0], ax=ax4, label='Change Magnitude')
            
            plt.tight_layout()
            
            # Save frame if writing video
            if writer is not None:
                writer.grab_frame()
            
            # Send visualization data via OSC
            osc_bridge.send_visualization_data(
                voice_states=voice_states,
                voice_projections=voice_projections,
                connection_strengths=connection_strengths,
                reflection_magnitudes=reflection_magnitudes,
                overall_rotation=mandala_overall_rotation_angle
            )
            
            logger.info(f"Completed iteration {iteration + 1} with {len(voice_states)} voice reflections")
    
    finally:
        # Clean up video writer
        if writer is not None:
            writer.finish()
    
    # Return the reflection results and final visualization
    return reflection_results, fig

@ray.remote
class GeometricReflectionWorker:
    """Ray actor for running geometric reflection games in parallel."""
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def run_game(self, 
                      num_iterations: int = 10,
                      save_video: bool = True,
                      video_path: Optional[str] = None) -> Dict:
        """Run a single geometric reflection game."""
        config = FlockConfig(
            num_models=1,
            batch_size=1,
            max_length=128,
            temperature=0.7,
            pattern_depth=8,
            flock_alignment=0.87,
            flock_cohesion=0.68,
            flock_separation=0.13
        )
        
        generator = FlockGenerator(
            model_names=[self.model_name],
            config=config
        )
        
        # Run the game and return results
        results, fig = await geometric_reflection_game(
            save_video=save_video,
            video_path=video_path
        )
        
        return {
            'results': results,
            'video_path': video_path if save_video else None
        }

def run_parallel_reflection_games(num_games: int = 4, output_dir: str = "reflection_games"):
    """Run multiple geometric reflection games in parallel using Ray."""
    # Initialize Ray
    ray.init()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create workers
    workers = [GeometricReflectionWorker.remote() for _ in range(num_games)]
    
    # Run games in parallel
    futures = []
    for i in range(num_games):
        video_path = os.path.join(output_dir, f"reflection_game_{i}.mp4")
        future = workers[i].run_game.remote(
            num_iterations=10,
            save_video=True,
            video_path=video_path
        )
        futures.append(future)
    
    # Get results
    results = ray.get(futures)
    
    # Shutdown Ray
    ray.shutdown()
    
    return results

async def run_vignettes_with_video(output_dir: str = "vignette_videos"):
    """Run all vignettes and save their visualizations as videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Run geometric reflection game
    reflection_results, fig = await geometric_reflection_game(
        save_video=True,
        video_path=os.path.join(output_dir, "geometric_reflection.mp4")
    )
    
    # Save the final figure
    plt.savefig(os.path.join(output_dir, "geometric_reflection_final.png"))
    plt.close(fig)
    
    # Run parallel games
    parallel_results = run_parallel_reflection_games(
        num_games=4,
        output_dir=os.path.join(output_dir, "parallel_games")
    )
    
    return {
        'reflection_results': reflection_results,
        'parallel_results': parallel_results,
        'output_dir': output_dir
    }

def create_visualization_notebook(output_dir: str = "vignette_videos"):
    """Create a Jupyter notebook to display the results."""
    notebook_content = f"""# Geometric Reflection Game Visualization

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import os
import json
import numpy as np

# Set up the output directory
output_dir = "{output_dir}"

# Display the final state visualization
plt.figure(figsize=(15, 10))
img = plt.imread(os.path.join(output_dir, "geometric_reflection_final.png"))
plt.imshow(img)
plt.axis('off')
plt.title('Final State of Geometric Reflection Game')
plt.show()

# Display the main video
from IPython.display import Video
video_path = os.path.join(output_dir, "geometric_reflection.mp4")
display(Video(video_path, embed=True))

# Display parallel game videos
parallel_dir = os.path.join(output_dir, "parallel_games")
print("\\nParallel Game Results:")
for i in range(4):
    video_path = os.path.join(parallel_dir, f"reflection_game_{i}.mp4")
    print(f"\\nGame {i+1}:")
    display(Video(video_path, embed=True))

# Create an interactive visualization of the reflection results
def plot_reflection_metrics(results):
    iterations = [r['iteration'] for r in results]
    avg_projections = [np.mean([p.item() for p in r['voice_projections']]) for r in results]
    avg_reflections = [np.mean(r['reflection_magnitudes']) for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(iterations, avg_projections, 'b-', label='Average Projection')
    ax1.set_title('Average Projection Strength Over Time')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Projection Strength')
    ax1.grid(True)
    
    ax2.plot(iterations, avg_reflections, 'r-', label='Average Reflection')
    ax2.set_title('Average Reflection Magnitude Over Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reflection Magnitude')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Load and plot the reflection results
with open(os.path.join(output_dir, 'reflection_results.json'), 'r') as f:
    results = json.load(f)
plot_reflection_metrics(results)

# Display some example generated texts
print("\\nExample Generated Texts:")
for i, result in enumerate(results):
    print(f"\\nIteration {i+1}:")
    print(result['base_text'][:200] + "...")
"""
    
    # Save the notebook
    notebook_path = os.path.join(output_dir, "visualization.ipynb")
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    return notebook_path

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the vignettes with video saving
    output_dir = "vignette_videos"
    results = asyncio.run(run_vignettes_with_video(output_dir))
    
    # Save the reflection results for the notebook
    with open(os.path.join(output_dir, 'reflection_results.json'), 'w') as f:
        json.dump(results['reflection_results'], f)
    
    # Create the visualization notebook
    notebook_path = create_visualization_notebook(output_dir)
    print(f"\nVisualization notebook created at: {notebook_path}")
    print("You can now run this notebook to view the results interactively.") 