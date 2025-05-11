import torch
import asyncio
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager, SecurityConfig, ResonantMode
from flock_generator import FlockConfig, FlockGenerator, IterationCallback
from visualization_manager import VectorDrivenVisualizer
import numpy as np
from storyteller import Storyteller
import logging
import pyglet
import pyglet.shapes as pyglet_shapes
import pyglet.graphics
import random # For randomizing star properties
import math # For atan2, degrees for tail rotation

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
        num_models=len(selected_models), pattern_depth=3, batch_size=2, max_length=64,
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
        pyglet_window = pyglet.window.Window(width=800, height=800, caption='Pattern Evolution (Pyglet)', resizable=True)
        pyglet_batch = pyglet.graphics.Batch()
        pyglet_fps_display = pyglet.window.FPSDisplay(window=pyglet_window)
        
        pyglet_group_background = pyglet.graphics.Group(order=0)
        pyglet_group_foreground = pyglet.graphics.Group(order=1)
        pyglet_group_stars = pyglet.graphics.Group(order=2)
        pyglet_group_glyphs = pyglet.graphics.Group(order=3) # Glyphs on top of stars, or adjust as needed

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
        logger.error(f"Pyglet window initialization failed: {e}", exc_info=True)
        print(f"ERROR: Could not initialize Pyglet window. Ensure you have an X server or equivalent. {e}")
        return

    @pyglet_window.event
    def on_draw():
        logger.debug("on_draw event fired.")
        pyglet_window.clear()
        pyglet_batch.draw()
        pyglet_fps_display.draw()

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

        mandala_overall_rotation_angle += 0.5
        if mandala_overall_rotation_angle > 360: mandala_overall_rotation_angle -= 360

        await pattern_evolution_update_cycle(flock_gen, story_tell, vector_vis, semantic_flow_prompts)
        
        if current_pattern_tensor is not None:
            tensor_mean = current_pattern_tensor.mean().item() if isinstance(current_pattern_tensor, torch.Tensor) else 'N/A'
            logger.debug(f"Scheduled_update - Updating visuals with tensor (mean: {tensor_mean}), Overall Rotation: {mandala_overall_rotation_angle:.2f}")
            vector_vis.update_geometric_mandala_pyglet(
                current_pattern_tensor, pyglet_batch,
                group_foreground=pyglet_group_foreground,
                group_background=pyglet_group_background,
                window_width=pyglet_window.width, window_height=pyglet_window.height,
                mandala_shape_bias=flock_gen.config.mandala_shape_bias,
                overall_rotation_angle_deg=mandala_overall_rotation_angle
            )
        
        stars_to_remove = []
        for star in active_stars:
            if not star.update(dt):
                stars_to_remove.append(star)
        
        for star in stars_to_remove:
            star.delete()
            active_stars.remove(star)
        
        # Update and remove EntendreGlyphs
        glyphs_to_remove = []
        for glyph in active_glyphs:
            if not glyph.update(dt):
                glyphs_to_remove.append(glyph)
        
        for glyph in glyphs_to_remove:
            glyph.delete()
            active_glyphs.remove(glyph)
            
        logger.debug(f"Scheduled_update - END - cycle: {generation_cycle_count-1}, Active Stars: {len(active_stars)}, Active Glyphs: {len(active_glyphs)}")

    pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1.0)
    stop_event.clear() # stop_event is already cleared at the top, but re-clearing after setup is fine.
    logger.info("Starting Pyglet app with integrated asyncio/Pyglet event loop...")

    try:
        while not pyglet_window.has_exit and not stop_event.is_set():
            dt = pyglet.clock.tick() # Get DT, advances pyglet clock, allows event polling

            # Process Pyglet platform events (OS messages, input, etc.)
            pyglet.app.platform_event_loop.step(timeout=1/200.0) 

            # Explicitly dispatch Pyglet window events (like on_draw, on_resize, etc.)
            pyglet_window.dispatch_events() # This should ensure on_draw is called if needed

            if pyglet_window.has_exit or stop_event.is_set(): 
                break

            await scheduled_update(dt) # Run our main async update logic

            pyglet_window.flip() # Ensure the frame is displayed

            await asyncio.sleep(0.001) 

    except Exception as e:
        logger.error(f"Error in Pyglet/asyncio main loop: {e}", exc_info=True)
        stop_event.set() # Ensure stop_event is set to facilitate graceful shutdown
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
        num_models=2,
        flock_alignment=0.8,
        flock_cohesion=0.6,
        flock_separation=0.1
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

    emotional_prompts = [
        "The joy of discovery:", "The wonder of creation:",
        "The harmony of nature:", "The beauty of simplicity:"
    ]
    
    emotional_states = []
    timesteps = []
    final_text_results = [] # Store final text results across iterations
    
    for i in range(3):
        # Generate results - expecting List[Tuple[str, torch.Tensor]]
        results_tuples = await generator.generate_batch(
            prompts=emotional_prompts,
            num_iterations=1 # Single flocking iteration per loop
        )
        
        # Extract text results for this iteration
        current_text_results = [text for text, tensor in results_tuples]
        if i == 2: # Keep results from the last iteration
            final_text_results = current_text_results

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
    return final_text_results, emotional_fig

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

async def run_vignettes():
    """Run all vignettes and display results."""
    vignettes = {
        "Language-Specific Swarm": language_specific_swarm,
        "Pattern Evolution Swarm": pattern_evolution_swarm,
        "Resource-Aware Swarm": resource_aware_swarm,
        "Emotional Resonance Swarm": emotional_resonance_swarm,
        "Cross-Language Harmony Swarm": cross_language_harmony_swarm
    }
    
    for name, vignette in vignettes.items():
        print(f"\n=== Running Vignette: {name} ===")
        try:
            # Run the async vignette function
            results = await vignette()
            
            # Process results based on expected return type (text list + figures)
            if isinstance(results, tuple) and len(results) >= 1:
                text_results = results[0]
                figures = results[1:] # Can be one or more figures
                
                print("\nGenerated Text:")
                if isinstance(text_results, list):
                    for i, result in enumerate(text_results):
                        print(f"\nResult {i + 1}:")
                        print(result)
                else:
                    print(f"\nResult 1:") # Handle if only one text result is returned
                    print(text_results)
                
                if figures:
                    print(f"\nVisualizations generated ({len(figures)} figure(s)). Check the plots.")
                    # Note: matplotlib figures might show automatically or need plt.show() depending on environment
                else:
                    print("\nNo visualizations returned.")

            else:
                 # Handle cases where only text results might be returned (though unlikely based on vignette structure)
                print("\nGenerated Text (raw result):")
                print(results)

        except Exception as e:
            logger.error(f"Error in vignette '{name}': {e}", exc_info=True) # Log full traceback
            print(f"\n--- ERROR in vignette '{name}': {e} ---")
            # Optionally re-raise or continue to next vignette
            # raise Exception(f"Error in vignette {name}: {e}") from e
            print(f"--- Skipping vignette '{name}' due to error. ---")

if __name__ == "__main__":
    # Configure logging to see the output from the callback
    # Set level to DEBUG to see the new detailed star update logs
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # To run the Pyglet based pattern evolution:
    # asyncio.run(pattern_evolution_swarm()) 
    
    # To run all vignettes (be mindful of mixed graphics backends):
    # asyncio.run(run_vignettes())

    # Defaulting to pattern_evolution_swarm for testing Pyglet
    # Ensure any previous matplotlib windows are closed if running in interactive environment.
    try:
        asyncio.run(pattern_evolution_swarm())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down.")
        stop_event.set() # Signal the loop to stop
        if pyglet_window:
            pyglet_window.close()
    except Exception as e:
        logger.error(f"Unhandled error in main execution: {e}", exc_info=True)
    finally:
        # Ensure stop_event is set so any scheduled async tasks can see it
        stop_event.set() 
        logger.info("Application terminated.") 