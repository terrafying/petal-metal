import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Set
import hashlib
import logging
from dataclasses import dataclass, field
import threading
import time
from contextlib import contextmanager
import concurrent.futures
import asyncio
import traceback
from logging_utils import get_logger

logger = get_logger(__name__)

def log_error_with_traceback(logger, error: Exception, context: str = ""):
    """Log error with limited traceback (3 steps) and context."""
    tb = traceback.extract_tb(error.__traceback__)
    # Get last 3 frames of traceback
    relevant_frames = tb[-3:] if len(tb) > 3 else tb
    
    error_info = {
        "error_type": type(error).__name__,
        "error_msg": str(error),
        "context": context,
        "traceback": [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
                "code": frame.line
            } for frame in relevant_frames
        ]
    }
    
    logger.error(
        "Error occurred: %s\nContext: %s\nTraceback (last 3 steps):\n%s",
        error_info["error_type"],
        error_info["context"],
        "\n".join([
            f"  File '{frame['file']}', line {frame['line']}, in {frame['function']}\n"
            f"    {frame['code']}"
            for frame in error_info["traceback"]
        ])
    )

@dataclass
class ResonantMode:
    frequency: float
    amplitude: float
    phase: float
    decay: float
    coupling_strength: float = 0.5
    harmonic_order: int = 1
    modulation_depth: float = 0.3
    phase_coupling: float = 0.2
    temporal_scale: float = 1.0
    
    # Scales for modulation by driving_vector
    freq_mod_scale: float = 0.5 
    amp_mod_scale: float = 0.5
    depth_mod_scale: float = 0.3
    time_mod_scale: float = 0.2
    
    def to_tensor(self, embedding_dim: int, dynamic_phase_offset: float = 0.0, driving_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert resonant mode parameters to a structured tensor, modulated by a driving_vector."""
        # Base parameters from self, potentially modulated by driving_vector
        base_freq = self.frequency
        base_amp = self.amplitude
        base_mod_depth = self.modulation_depth
        base_time_scale = self.temporal_scale
        current_phase = self.phase + dynamic_phase_offset

        if driving_vector is not None and driving_vector.numel() > 0:
            # Ensure driving_vector has at least 4 elements, or repeat if fewer
            if driving_vector.numel() < 4:
                # Ensure driving_vector is on the same device as model parameters if applicable
                # For now, assuming it's a CPU tensor or device handling is managed upstream
                dv_expanded = driving_vector.repeat((4 // driving_vector.numel()) + 1)
                dv_final = dv_expanded[:4]
            else:
                dv_final = driving_vector[:4]

            dv_norm = torch.tanh(dv_final)

            base_freq += dv_norm[0] * self.freq_mod_scale * self.frequency 
            base_amp += dv_norm[1] * self.amp_mod_scale * self.amplitude
            base_mod_depth += dv_norm[2] * self.depth_mod_scale * self.modulation_depth
            base_time_scale += dv_norm[3] * self.time_mod_scale * self.temporal_scale

            # Clamp to avoid extreme values
            base_freq = max(0.01, base_freq.item() if isinstance(base_freq, torch.Tensor) else base_freq)
            base_amp = max(0.01, base_amp.item() if isinstance(base_amp, torch.Tensor) else base_amp)
            base_mod_depth = max(0.0, min(1.0, base_mod_depth.item() if isinstance(base_mod_depth, torch.Tensor) else base_mod_depth))
            base_time_scale = max(0.1, base_time_scale.item() if isinstance(base_time_scale, torch.Tensor) else base_time_scale)
        
        # Create base tensor with geometric structure
        base_output_tensor = torch.zeros(embedding_dim)
        
        # Create temporal axis based on modulated time scale
        t = torch.linspace(0, 2 * np.pi * base_time_scale, embedding_dim)
        
        # Apply frequency modulation using modulated frequency and current_phase
        freq_mod_wave = torch.sin(t * base_freq + current_phase)
        base_output_tensor += base_amp * freq_mod_wave
        
        # Apply harmonic structure using modulated parameters
        for h_order in range(1, self.harmonic_order + 1):
            harmonic_t = t * h_order
            # Harmonic frequency uses base_freq, phase includes current_phase
            harmonic_wave = torch.sin(harmonic_t * base_freq + current_phase) 
            # Modulation wave for harmonics can use base_mod_depth
            modulation_wave_for_harmonics = torch.sin(harmonic_t * base_mod_depth) 
            
            harmonic_combined = harmonic_wave * modulation_wave_for_harmonics
            # Amplitude scaling for harmonics uses base_amp
            base_output_tensor += (base_amp / (h_order + 1)) * harmonic_combined 
        
        # Apply phase coupling (original phase_coupling, not current_phase directly here as it's a different effect)
        # This phase_coupling might need to be re-thought if current_phase already covers all phase aspects.
        # For now, let's assume self.phase_coupling is a separate modulation component as per original.
        # It uses self.phase (base phase of the mode) for its wave.
        phase_coupling_wave = torch.sin(t * self.phase_coupling + self.phase) 
        base_output_tensor += self.phase_coupling * phase_coupling_wave
        
        # Normalize and apply decay (self.decay is the base decay rate)
        norm_val = base_output_tensor.norm()
        if norm_val > 1e-6:
            base_output_tensor = base_output_tensor / norm_val
        
        temporal_decay_values = torch.exp(-t * self.decay / base_time_scale)
        base_output_tensor *= temporal_decay_values
        
        return base_output_tensor

@dataclass
class SecurityConfig:
    """Configuration for pattern generation security and resource management."""
    # Resource limits
    max_pattern_size: int = 1024 * 1024  # 1MB
    max_recursion_depth: int = 10
    pattern_timeout: float = 5.0  # seconds
    
    # Rate limiting
    rate_limit_requests: int = 100  # requests per second
    rate_limit_window: float = 1.0  # seconds
    
    # Concurrency control
    max_concurrent_operations: int = 5
    
    # Dimensionality constraints
    min_embedding_dim: int = 64
    max_embedding_dim: int = 4096
    position_dim: int = 3  # Fixed dimension for swarm positions
    language_pattern_dim: int = 3  # Fixed dimension for language patterns
    
    def validate_tensor_dimensions(self, tensor: torch.Tensor, expected_dims: Optional[int] = None) -> None:
        """Validate tensor dimensions against security constraints."""
        if tensor.numel() > self.max_pattern_size:
            raise SecurityError(f"Tensor size {tensor.numel()} exceeds maximum allowed {self.max_pattern_size}")
        
        if not torch.isfinite(tensor).all():
            raise SecurityError("Tensor contains non-finite values")
        
        if expected_dims is not None and tensor.dim() != expected_dims:
            raise SecurityError(f"Expected {expected_dims} dimensions, got {tensor.dim()}")
        
        if tensor.dim() > 0 and tensor.size(-1) > self.max_embedding_dim:
            raise SecurityError(f"Embedding dimension {tensor.size(-1)} exceeds maximum {self.max_embedding_dim}")
        
        if tensor.dim() > 0 and tensor.size(-1) < self.min_embedding_dim:
            raise SecurityError(f"Embedding dimension {tensor.size(-1)} below minimum {self.min_embedding_dim}")

@dataclass
class SwarmNode:
    node_id: int
    position: torch.Tensor
    velocity: torch.Tensor
    best_position: torch.Tensor
    best_fitness: float
    resource_budget: float = 1.0
    query_count: int = 0
    success_rate: float = 1.0
    last_query_time: float = 0.0
    thread_id: Optional[int] = None
    experience_level: int = 1
    specializations: Set[str] = field(default_factory=set)
    interaction_history: List[Dict] = field(default_factory=list)
    energy_level: float = 1.0
    creativity_score: float = 0.5
    language_config: Dict[str, float] = field(default_factory=lambda: {
        "en": 1.0,  # English as base language
        "es": 0.8,  # Spanish
        "fr": 0.8,  # French
        "de": 0.8,  # German
        "zh": 0.7,  # Chinese
        "ja": 0.7,  # Japanese
        "ko": 0.7,  # Korean
        "ru": 0.8,  # Russian
        "ar": 0.7,  # Arabic
        "hi": 0.7,  # Hindi
    })
    language_specialization: str = "en"  # Default specialization

class SwarmState:
    def __init__(self, num_particles: int = 10, position_dim: int = 3):
        self.nodes = [
            SwarmNode(
                node_id=i,
                position=torch.randn(position_dim),
                velocity=torch.zeros(position_dim),
                best_position=torch.randn(position_dim),
                best_fitness=float('inf'),
                resource_budget=1.0,
                thread_id=i % threading.active_count(),
                specializations=set(np.random.choice(
                    ["pattern_recognition", "harmony", "rhythm", "melody", "texture"],
                    size=np.random.randint(1, 4),
                    replace=False
                ))
            ) for i in range(num_particles)
        ]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.velocity_clamp = 2.0
        self.adaptive_weights = True
        self.generation = 0
        self.diversity_threshold = 0.1
        self.diversity_history = []
        self.reinforcement_threshold = 0.7
        self.min_resource_budget = 0.1
        self.max_queries_per_second = 10
        self._node_locks = {node.node_id: threading.Lock() for node in self.nodes}
        self.collective_consciousness = torch.zeros(position_dim)  # Shared knowledge state
        self.energy_field = torch.ones(num_particles)  # Energy distribution
        self.creative_potential = torch.ones(num_particles)  # Creative potential
        self.harmony_matrix = torch.ones(num_particles, num_particles)  # Inter-node harmony

    def select_node_for_query(self) -> Optional[SwarmNode]:
        """Select the best node for a new query based on resource availability and success rate."""
        current_time = time.time()
        available_nodes = []
        
        for node in self.nodes:
            with self._node_locks[node.node_id]:
                # Check if node has resources and hasn't exceeded query rate
                if (node.resource_budget >= self.min_resource_budget and
                    current_time - node.last_query_time >= 1.0 / self.max_queries_per_second):
                    # Calculate node score based on success rate and resource budget
                    score = node.success_rate * node.resource_budget
                    available_nodes.append((node, score))
        
        if not available_nodes:
            return None
            
        # Select node with highest score
        return max(available_nodes, key=lambda x: x[1])[0]

    def update_node_stats(self, node_id: int, success: bool, resource_used: float):
        """Update node statistics after a query."""
        with self._node_locks[node_id]:
            node = next(n for n in self.nodes if n.node_id == node_id)
            node.query_count += 1
            node.resource_budget -= resource_used
            node.last_query_time = time.time()
            
            # Update success rate with exponential moving average
            alpha = 0.1
            node.success_rate = (1 - alpha) * node.success_rate + alpha * float(success)
            
            # Reinforce successful nodes
            if success and node.success_rate > self.reinforcement_threshold:
                node.resource_budget = min(1.0, node.resource_budget + 0.1)

    def evolve_node(self, node: SwarmNode, success: bool):
        """Evolve node based on its performance and interactions."""
        with self._node_locks[node.node_id]:
            # Update experience level
            if success:
                node.experience_level += 1
                node.energy_level = min(1.0, node.energy_level + 0.1)
                node.creativity_score = min(1.0, node.creativity_score + 0.05)
            else:
                node.energy_level = max(0.1, node.energy_level - 0.05)
            
            # Update specializations based on performance
            if success and np.random.random() < 0.1:
                new_specialization = np.random.choice([
                    "pattern_recognition", "harmony", "rhythm", "melody", "texture"
                ])
                node.specializations.add(new_specialization)
            
            # Record interaction
            node.interaction_history.append({
                "timestamp": time.time(),
                "success": success,
                "energy_level": node.energy_level,
                "creativity_score": node.creativity_score
            })

    def update_collective_consciousness(self):
        """Update the collective consciousness based on node states."""
        positions = torch.stack([node.position for node in self.nodes])
        energies = torch.tensor([node.energy_level for node in self.nodes])
        creativity = torch.tensor([node.creativity_score for node in self.nodes])
        
        # Update collective consciousness
        self.collective_consciousness = (
            torch.mean(positions, dim=0) * 0.4 +
            torch.mean(energies) * 0.3 +
            torch.mean(creativity) * 0.3
        )
        
        # Update harmony matrix
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j:
                    harmony = torch.cosine_similarity(
                        node1.position.unsqueeze(0),
                        node2.position.unsqueeze(0)
                    )
                    self.harmony_matrix[i, j] = harmony

class ReentrancyGuard:
    def __init__(self, max_concurrent: int):
        self._lock = threading.Lock()
        self._active_operations = 0
        self._max_concurrent = max_concurrent
        self._operation_locks = {}  # Track locks per operation
        
    @contextmanager
    def guard_operation(self, operation_id: str):
        """Guard against reentrancy for a specific operation."""
        with self._lock:
            if self._active_operations >= self._max_concurrent:
                raise SecurityError("Maximum concurrent operations exceeded")
            
            # Check if this specific operation is already running
            if operation_id in self._operation_locks:
                raise SecurityError(f"Operation {operation_id} is already running")
            
            self._operation_locks[operation_id] = True
            self._active_operations += 1
        
        try:
            yield
        finally:
            with self._lock:
                self._active_operations -= 1
                self._operation_locks.pop(operation_id, None)

class PatternManager:
    def __init__(self, model: torch.nn.Module, pattern_depth: int = 3, base_seed: int = 42):
        self.logger = logger
        self.logger.info("Initializing PatternManager with depth %d and seed %d", pattern_depth, base_seed)
        
        self.model = model
        self.pattern_depth = min(pattern_depth, 10)  # Cap recursion depth
        self.base_seed = base_seed
        self.rng = np.random.RandomState(base_seed)
        
        # Initialize operation counter for unique IDs
        self._operation_counter = 0
        
        # Initialize security-related attributes
        self._reentrancy_guard = ReentrancyGuard(max_concurrent=5)
        self._lock = threading.Lock()
        self._request_times = []
        self._query_futures = {}
        self.query_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.time_step = 0
        
        # Get model's embedding dimension
        self.embedding_dim = self.model.get_input_embeddings().weight.shape[1]
        
        # Initialize security config
        self.security_config = SecurityConfig()
        
        # Initialize pattern storage with geometric structure
        self.patterns: Dict[str, torch.Tensor] = {}
        self.resonance_matrix: Optional[torch.Tensor] = None
        
        # Initialize resonant modes with geometric structure
        self.resonant_modes = self._initialize_resonant_modes(num_modes=5)
        self.coupling_matrix = self._initialize_coupling_matrix(num_modes=5)
        
        # Initialize swarm state with proper dimensionality
        self.swarm_state = SwarmState(
            num_particles=10,
            position_dim=self.security_config.position_dim
        )
        
        # Initialize node locks for all nodes
        self._node_locks = {node.node_id: threading.Lock() for node in self.swarm_state.nodes}
        
        # Initialize language patterns with geometric structure
        self.language_patterns = {}
        # Initialize language weights as a dictionary
        self.language_weights = {
            "en": 1.0,  # English as base language
            "es": 0.8,  # Spanish
            "fr": 0.8,  # French
            "de": 0.8,  # German
            "zh": 0.7,  # Chinese
            "ja": 0.7,  # Japanese
            "ko": 0.7,  # Korean
            "ru": 0.8,  # Russian
            "ar": 0.7,  # Arabic
            "hi": 0.7,  # Hindi
        }
        
        # Initialize emotional state with geometric structure
        self.emotional_state = {
            "intensity": torch.zeros(self.embedding_dim),
            "coherence": torch.ones(self.embedding_dim),
            "complexity": torch.ones(self.embedding_dim)
        }
        
        # Add shared memory pool (Restored)
        self.shared_memory = {
            'patterns': [],  # Store pattern embeddings
            'interpretations': [],  # Store pattern interpretations
            'contexts': [],  # Store context information
            'timestamps': [],  # Track when patterns were added
            'usage_count': [],  # Track how often patterns are borrowed
            'affinity_scores': []  # Track pattern compatibility
        }
        
        # Memory pool parameters (Restored)
        self.max_memory_size = 1000  # Maximum number of patterns to store
        self.memory_decay = 0.95  # Decay factor for old patterns
        self.borrow_threshold = 0.7  # Minimum similarity to borrow a pattern
        
        self.logger.info("PatternManager initialized successfully")
    
    def get_memory_stats(self) -> Dict[str, any]:
        """Get statistics about the shared memory pool."""
        with self._lock: # Ensure thread safety if shared_memory is modified elsewhere
            num_patterns = len(self.shared_memory.get('patterns', []))
            utilization = num_patterns / self.max_memory_size if self.max_memory_size > 0 else 0
            
            avg_usage_count = 0
            if self.shared_memory.get('usage_count') and num_patterns > 0:
                avg_usage_count = sum(self.shared_memory['usage_count']) / num_patterns

            avg_affinity_score = 0
            if self.shared_memory.get('affinity_scores') and num_patterns > 0:
                avg_affinity_score = sum(self.shared_memory['affinity_scores']) / num_patterns

            return {
                "total_patterns": num_patterns,
                "max_memory_size": self.max_memory_size,
                "memory_utilization": utilization,
                "average_usage_count": avg_usage_count,
                "average_affinity_score": avg_affinity_score
            }

    def _validate_pattern_dimensions(self, pattern: torch.Tensor) -> None:
        """Validate pattern dimensions and content."""
        self.security_config.validate_tensor_dimensions(pattern)
        
        # Additional pattern-specific validations
        if pattern.dim() < 2:
            raise SecurityError("Pattern must have at least 2 dimensions")
        
        # Check for NaN or Inf values
        if torch.isnan(pattern).any() or torch.isinf(pattern).any():
            raise SecurityError("Pattern contains NaN or Inf values")
    
    def generate_pattern(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate a pattern from input tensor."""
        try:
            self.logger.debug("Generating pattern for input tensor of shape %s", input_tensor.shape)
            
            # Validate input dimensions
            self._validate_pattern_dimensions(input_tensor)
            
            # Pattern generation logic here
            pattern = self._apply_pattern_transformation(input_tensor)
            
            # Validate output dimensions
            self._validate_pattern_dimensions(pattern)
            
            self.logger.debug("Pattern generated successfully with shape %s", pattern.shape)
            return pattern
        except Exception as e:
            self.logger.exception("Failed to generate pattern: %s", e)
            raise
    
    def _apply_pattern_transformation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply pattern transformation with geometric structure preservation."""
        try:
            self.logger.debug("Applying pattern transformation to tensor of shape %s", tensor.shape)
            
            # Ensure proper shape
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            
            # Get current embedding dimension
            current_dim = tensor.shape[-1]
            
            # Project to model's embedding dimension if needed
            if current_dim != self.embedding_dim:
                tensor = torch.nn.functional.linear(
                    tensor,
                    self.model.get_input_embeddings().weight
                )
            
            # Apply geometric transformations
            transformed = tensor.clone()
            
            # Apply resonant modes with geometric structure
            for mode in self.resonant_modes:
                mode_tensor = mode.to_tensor(self.embedding_dim)
                transformed = transformed * (1 + mode_tensor.unsqueeze(0))
            
            # Apply emotional state influence
            emotional_influence = (
                self.emotional_state["intensity"].unsqueeze(0) * 0.4 +
                self.emotional_state["coherence"].unsqueeze(0) * 0.3 +
                self.emotional_state["complexity"].unsqueeze(0) * 0.3
            )
            transformed = transformed * (1 + emotional_influence)
            
            # Preserve geometric structure
            transformed = transformed / (transformed.norm(dim=-1, keepdim=True) + 1e-6)
            
            self.logger.debug("Pattern transformation completed with shape %s", transformed.shape)
            return transformed
        except Exception as e:
            self.logger.exception("Pattern transformation failed: %s", e)
            raise
    
    def compute_resonance(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """Compute resonance between two patterns."""
        try:
            self.logger.debug("Computing resonance between patterns of shapes %s and %s", 
                            pattern1.shape, pattern2.shape)
            
            # Resonance computation logic here
            resonance = self._calculate_resonance_score(pattern1, pattern2)
            
            self.logger.debug("Resonance computed: %.4f", resonance)
            return resonance
        except Exception as e:
            self.logger.exception("Failed to compute resonance: %s", e)
            raise
    
    def _calculate_resonance_score(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """Calculate resonance score between two patterns."""
        try:
            self.logger.debug(f"Calculating resonance score for p1: {p1.shape}, p2: {p2.shape}")

            # Ensure tensors are at least 2D (e.g. [seq_len, hidden_dim] or [batch, hidden_dim])
            # If 3D [batch, seq, hidden], take mean over seq_len.
            # If 1D [hidden_dim], unsqueeze to [1, hidden_dim]
            
            p1_proc = p1.clone()
            p2_proc = p2.clone()

            if p1_proc.ndim == 1:
                p1_proc = p1_proc.unsqueeze(0) # [1, hidden_dim]
            elif p1_proc.ndim == 3 and p1_proc.shape[0] == 1: # [1, seq, hidden]
                p1_proc = p1_proc.mean(dim=1) # [1, hidden_dim]
            elif p1_proc.ndim == 2: # [seq, hidden] or [batch_size > 1, hidden_dim]
                # If it's [seq, hidden], we could mean it. If [batch > 1, hidden], this is tricky.
                # Assuming for now p1 and p2 will be comparable after this, or batch_size is 1.
                # If seq_len is the first dim and hidden_dim is the second.
                if p1_proc.shape[0] > 1 and p1_proc.shape[0] != p2_proc.shape[0] : # Likely [seq, hidden]
                     p1_proc = p1_proc.mean(dim=0, keepdim=True) # [1, hidden_dim]


            if p2_proc.ndim == 1:
                p2_proc = p2_proc.unsqueeze(0)
            elif p2_proc.ndim == 3 and p2_proc.shape[0] == 1: # [1, seq, hidden]
                p2_proc = p2_proc.mean(dim=1)
            elif p2_proc.ndim == 2:
                if p2_proc.shape[0] > 1 and p1_proc.shape[0] != p2_proc.shape[0]: # Likely [seq, hidden]
                    p2_proc = p2_proc.mean(dim=0, keepdim=True)


            # After processing, p1_proc and p2_proc should be [batch_size, hidden_dim]
            # Ensure they are on the same device
            if p1_proc.device != p2_proc.device:
                p2_proc = p2_proc.to(p1_proc.device)

            # Ensure hidden dimensions match. If not, this approach won't work directly.
            if p1_proc.shape[-1] != p2_proc.shape[-1]:
                self.logger.warning(f"Hidden dimensions mismatch for resonance: p1_proc {p1_proc.shape}, p2_proc {p2_proc.shape}. Returning 0.0.")
                return 0.0

            # If batch sizes are different after processing (e.g., one was [seq,hidden] and other [1,seq,hidden])
            # and they are now e.g. [1, hidden] and [N, hidden], we can't directly do cosine similarity for all pairs.
            # For simplicity, if batch sizes differ but one is 1, expand it.
            if p1_proc.shape[0] != p2_proc.shape[0]:
                if p1_proc.shape[0] == 1:
                    p1_proc = p1_proc.expand_as(p2_proc)
                elif p2_proc.shape[0] == 1:
                    p2_proc = p2_proc.expand_as(p1_proc)
                else:
                    self.logger.warning(f"Batch sizes differ after processing and neither is 1: p1_proc {p1_proc.shape}, p2_proc {p2_proc.shape}. Returning 0.0.")
                    return 0.0 # Or handle by taking mean of batch, or pairwise and then mean

            # Cosine similarity expects inputs of shape (B, D) or (D)
            # torch.nn.functional.cosine_similarity computes it along a dimension.
            # If p1_proc is (N, D) and p2_proc is (N, D), it computes element-wise.
            similarity = torch.nn.functional.cosine_similarity(p1_proc, p2_proc, dim=-1)
            
            # If similarity is a tensor with multiple values (e.g. batch > 1), take the mean.
            score = similarity.mean().item()
            
            self.logger.debug(f"Resonance score calculated: {score:.4f}")
            return score
        except Exception as e:
            # Ensure logger is defined in this scope if it's not self.logger
            current_logger = getattr(self, 'logger', logging.getLogger(__name__))
            current_logger.exception("Failed to calculate resonance score: %s", e)
            # It's generally better to raise the exception to allow higher-level error handling,
            # or return a value that indicates error (like NaN or a specific error code if 0.0 is a valid score)
            # For now, returning 0.0 on error to maintain previous behavior of returning a float.
            return 0.0
    
    def update_patterns(self, new_patterns: Dict[str, torch.Tensor]):
        """Update stored patterns."""
        try:
            self.logger.info("Updating patterns with %d new entries", len(new_patterns))
            
            # Pattern update logic here
            self.patterns.update(new_patterns)
            
            self.logger.info("Patterns updated successfully")
        except Exception as e:
            self.logger.exception("Failed to update patterns: %s", e)
            raise
    
    def get_pattern(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a pattern by key."""
        try:
            self.logger.debug("Retrieving pattern for key: %s", key)
            
            pattern = self.patterns.get(key)
            
            if pattern is None:
                self.logger.warning("No pattern found for key: %s", key)
            else:
                self.logger.debug("Pattern retrieved successfully with shape %s", pattern.shape)
            
            return pattern
        except Exception as e:
            self.logger.exception("Failed to retrieve pattern: %s", e)
            raise

    def _initialize_resonant_modes(self, num_modes: int) -> List[ResonantMode]:
        """Initialize resonant modes with random parameters."""
        modes = []
        for i in range(num_modes):
            mode = ResonantMode(
                frequency=np.random.uniform(0.1, 2.0),
                amplitude=np.random.uniform(0.1, 1.0),
                phase=np.random.uniform(0, 2 * np.pi),
                decay=np.random.uniform(0.1, 0.5),
                coupling_strength=np.random.uniform(0.1, 0.5),
                harmonic_order=np.random.randint(1, 5),
                modulation_depth=np.random.uniform(0.1, 0.7),
                phase_coupling=np.random.uniform(0.1, 0.5),
                temporal_scale=np.random.uniform(0.5, 1.5),
                freq_mod_scale=np.random.uniform(0.1, 0.5),
                amp_mod_scale=np.random.uniform(0.1, 0.5),
                depth_mod_scale=np.random.uniform(0.1, 0.5),
                time_mod_scale=np.random.uniform(0.1, 0.3)
            )
            modes.append(mode)
        return modes
    
    def _initialize_coupling_matrix(self, num_modes: int) -> torch.Tensor:
        """Initialize the coupling matrix between resonant modes."""
        matrix = torch.zeros(num_modes, num_modes)
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                coupling = np.random.uniform(0.1, 0.5)
                matrix[i, j] = coupling
                matrix[j, i] = coupling
        return matrix
    
    def _generate_pattern_seed(self, block_idx: int, sequence_idx: int) -> int:
        """Generate a deterministic seed based on block and sequence indices."""
        seed_str = f"{self.base_seed}_{block_idx}_{sequence_idx}"
        return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    
    def _apply_resonant_modes(self, 
                            tensor: torch.Tensor, 
                            time_step: int,
                            language: str = "en") -> torch.Tensor:
        """Apply resonant modes with geometric structure preservation."""
        result = tensor.clone()
        
        # Get language-specific modulation
        lang_weight = self.language_weights.get(language, 0.5)
        
        # Extract a driving vector from the input tensor (e.g., mean of features or a fixed slice)
        # This driving vector will have 4 elements.
        if tensor.dim() > 1 and tensor.shape[-1] > 0 : # Ensure tensor is not empty and has a feature dimension
            driving_vector_source = tensor.mean(dim=list(range(tensor.dim() -1 )))
        elif tensor.dim() == 1 and tensor.numel() > 0: # 1D tensor
             driving_vector_source = tensor
        else: # Fallback if tensor is empty or unsuitable
            driving_vector_source = torch.zeros(4, device=tensor.device if isinstance(tensor, torch.Tensor) else None)

        if driving_vector_source.numel() >= 4:
            control_vector = driving_vector_source[:4]
        elif driving_vector_source.numel() > 0 : # Repeat if too short
             control_vector = driving_vector_source.repeat((4 // driving_vector_source.numel()) + 1)[:4]
        else: # Fallback if tensor was empty initially
            control_vector = torch.zeros(4, device=tensor.device if isinstance(tensor, torch.Tensor) else None)

        # Apply resonant modes with geometric structure
        for i, mode in enumerate(self.resonant_modes):
            # Calculate dynamic phase offset for this mode and timestep
            # mode.frequency here is the base frequency of the mode
            dynamic_phase_offset = mode.frequency * time_step 
            
            # Convert mode to tensor with geometric structure, applying driving vector and dynamic phase
            mode_wave = mode.to_tensor(
                self.embedding_dim, 
                dynamic_phase_offset=dynamic_phase_offset, 
                driving_vector=control_vector
            )
            
            # Current resonance starts with this mode's wave
            current_resonance_effect = mode_wave
            
            # Apply language-specific coupling with geometric structure
            for j, other_mode in enumerate(self.resonant_modes):
                if i != j:
                    coupling = self.coupling_matrix[i, j]
                    # Other modes also get driven by the same control_vector and their own dynamic phase offset
                    other_dynamic_phase_offset = other_mode.frequency * time_step
                    other_mode_wave = other_mode.to_tensor(
                        self.embedding_dim, 
                        dynamic_phase_offset=other_dynamic_phase_offset, 
                        driving_vector=control_vector
                    )
                    # The coupling in the original 1012-line file used torch.sin(other_tensor)
                    # If other_mode_wave is already a complex wave, applying torch.sin again might be distorting.
                    # Let's assume the coupling term should be based on the other_mode_wave directly or a simple transformation.
                    # Sticking to torch.sin for now to match the structure of the 1012-line file's coupling.
                    current_resonance_effect += coupling * torch.sin(other_mode_wave) 
            
            # Apply language modulation with geometric structure
            current_resonance_effect *= lang_weight
            
            # Apply to tensor while preserving geometric structure (multiplicative application)
            # mode.decay is the base decay for the mode.
            result = result * (1 + current_resonance_effect.unsqueeze(0) * mode.decay)
        
        # Normalize to preserve geometric structure
        norm_val = result.norm(dim=-1, keepdim=True)
        # Add a small epsilon to norm_val before division to prevent division by zero with zero tensors
        result = result / (norm_val + 1e-9) 
        
        return result
    
    def _get_operation_id(self, block_idx: int, sequence_idx: int) -> str:
        """Generate a unique operation ID."""
        return f"pattern_{block_idx}_{sequence_idx}_{self._operation_counter}"
    
    def _apply_recursive_pattern(self, tensor: torch.Tensor, pattern_seed: int, depth: int) -> torch.Tensor:
        """Apply recursive pattern to tensor with security checks."""
        if depth <= 0 or depth > self.security_config.max_recursion_depth:
            return tensor
            
        # Set deterministic seed for this pattern level
        torch.manual_seed(pattern_seed)
        np.random.seed(pattern_seed)
        
        # Generate stochastic pattern with size validation
        pattern = torch.randn_like(tensor)
        if pattern.numel() > self.security_config.max_pattern_size:
            raise SecurityError("Generated pattern exceeds size limit")
            
        pattern = pattern / pattern.norm()
        
        # Apply swarm-based modulation with validation
        if self.swarm_state.global_best_position is not None:
            swarm_modulation = torch.tanh(
                torch.matmul(pattern, self.swarm_state.global_best_position)
            )
            if not torch.isfinite(swarm_modulation).all():
                raise SecurityError("Invalid swarm modulation detected")
            pattern = pattern * (1 + swarm_modulation)
        
        # Apply recursive pattern with timeout
        start_time = time.time()
        result = tensor + 0.1 * pattern * self._apply_recursive_pattern(
            torch.ones_like(tensor), 
            pattern_seed + 1,
            depth - 1
        )
        
        if time.time() - start_time > self.security_config.pattern_timeout:
            raise SecurityError("Pattern generation timeout")
        
        return result
    
    def _align_tensor_strategies(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Apply nine different strategies for tensor alignment and transformation."""
        strategies = []
        
        # Ensure tensor has at least 2 dimensions
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        # Handle different tensor dimensions
        if tensor.dim() == 2:
            # For 2D tensors [batch, hidden], add sequence dimension
            batch_size, hidden = tensor.shape
            tensor = tensor.unsqueeze(1)  # Add sequence dimension
            seq_len = 1
        else:
            batch_size, seq_len, hidden = tensor.shape
        
        # Strategy 1: Direct projection
        strategies.append(torch.nn.functional.linear(
            tensor,
            self.model.get_input_embeddings().weight
        ))
        
        # Strategy 2: SVD-based alignment
        # Reshape to 2D for SVD: [batch, seq_len, hidden] -> [batch*seq_len, hidden]
        reshaped = tensor.reshape(-1, hidden)
        U, S, V = torch.linalg.svd(reshaped, full_matrices=False)
        # Ensure dimensions match for multiplication
        U = U[:, :hidden]  # Keep only needed columns
        S = S[:hidden]     # Keep only needed singular values
        # Reshape back to original dimensions
        svd_result = torch.matmul(U, torch.diag(S))
        # Ensure the total size matches before reshaping
        if svd_result.numel() == batch_size * seq_len * hidden:
            strategies.append(svd_result.reshape(batch_size, seq_len, hidden))
        else:
            # Fallback to direct projection if dimensions don't match
            projection = torch.nn.Linear(svd_result.size(-1), hidden)
            strategies.append(projection(svd_result).reshape(batch_size, seq_len, hidden))
        
        # Strategy 3: PCA-based transformation
        centered = tensor - tensor.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered)
        eigenvals, eigenvecs = torch.linalg.eigh(cov, UPLO='L')
        strategies.append(torch.matmul(centered, eigenvecs))
        
        # Strategy 4: Fourier-based alignment
        fft = torch.fft.fft2(tensor)
        strategies.append(torch.fft.ifft2(fft).real)
        
        # Strategy 5: Wavelet-based transformation
        if tensor.dim() == 3:
            # Ensure we have enough dimensions for pooling
            if tensor.size(1) > 1 and tensor.size(2) > 1:
                # Use adaptive pooling to handle varying sizes
                strategies.append(torch.nn.functional.adaptive_avg_pool2d(
                    tensor.unsqueeze(1),
                    output_size=(tensor.size(1), tensor.size(2))
                ).squeeze(1))
            else:
                # For small inputs, use a simple linear projection
                projection = torch.nn.Linear(tensor.size(-1), tensor.size(-1))
                strategies.append(projection(tensor))
        
        # Strategy 6: Attention-based alignment
        attention = torch.matmul(tensor, tensor.transpose(1, 2))
        attention = torch.softmax(attention / torch.sqrt(torch.tensor(tensor.size(-1))), dim=-1)
        strategies.append(torch.matmul(attention, tensor))
        
        # Strategy 7: Geometric structure preservation
        norm = tensor.norm(dim=-1, keepdim=True)
        strategies.append(tensor / (norm + 1e-6))
        
        # Strategy 8: Manifold alignment
        if tensor.dim() == 3:
            laplacian = torch.matmul(tensor.transpose(1, 2), tensor)
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian, UPLO='L')
            strategies.append(torch.matmul(tensor, eigenvecs))
        
        # Strategy 9: Adaptive projection
        if tensor.size(-1) != self.embedding_dim:
            projection = torch.nn.Linear(tensor.size(-1), self.embedding_dim)
            strategies.append(projection(tensor))
        
        return strategies

    def _select_best_strategy(self, strategies: List[torch.Tensor], target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Select the best strategy based on shape compatibility and quality metrics."""
        valid_strategies = []
        for strategy in strategies:
            if strategy.shape[-1] == target_shape[-1]:
                # Calculate quality metrics
                norm = strategy.norm()
                entropy = -torch.sum(torch.softmax(strategy, dim=-1) * 
                                  torch.log_softmax(strategy, dim=-1))
                coherence = torch.mean(torch.abs(torch.fft.fft(strategy)))
                
                # Combine metrics
                quality_score = norm * entropy * coherence
                valid_strategies.append((strategy, quality_score))
        
        if not valid_strategies:
            # Fallback to direct projection if no valid strategies
            return torch.nn.functional.linear(
                strategies[0],
                self.model.get_input_embeddings().weight
            )
        
        # Select strategy with highest quality score
        return max(valid_strategies, key=lambda x: x[1])[0]

    async def process_output(self, 
                           output: torch.Tensor, 
                           block_idx: int, 
                           sequence_idx: int,
                           maintain_consistency: bool = True,
                           language: str = "en") -> torch.Tensor:
        """Process model output with nine-fold strategy alignment."""
        operation_id = self._get_operation_id(block_idx, sequence_idx)
        self._operation_counter += 1
        
        try:
            with self._reentrancy_guard.guard_operation(operation_id):
                self._check_rate_limit()
                self._validate_input(output)
                
                if not maintain_consistency:
                    return output
                
                # Select node for processing
                node = self.swarm_state.select_node_for_query()
                if node is None:
                    raise SecurityError("No available nodes for processing")
                
                # Apply nine-fold strategy alignment
                try:
                    strategies = self._align_tensor_strategies(output)
                    aligned_output = self._select_best_strategy(
                        strategies,
                        (output.size(0), self.embedding_dim)
                    )
                except Exception as e:
                    log_error_with_traceback(
                        logger,
                        e,
                        f"Error in strategy alignment for output shape {output.shape}"
                    )
                    raise
                
                # Submit query to thread pool
                future = self.query_thread_pool.submit(
                    self._process_with_node,
                    node.node_id,
                    aligned_output.clone(),
                    block_idx,
                    sequence_idx,
                    language
                )
                self._query_futures[operation_id] = future
                
                # Wait for result with timeout
                try:
                    result = await asyncio.wrap_future(future)
                    self.swarm_state.update_node_stats(
                        node.node_id,
                        success=True,
                        resource_used=0.1
                    )
                    return result
                except Exception as e:
                    self.swarm_state.update_node_stats(
                        node.node_id,
                        success=False,
                        resource_used=0.1
                    )
                    log_error_with_traceback(
                        logger,
                        e,
                        f"Node processing failed for node {node.node_id}"
                    )
                    raise SecurityError(f"Node processing failed: {str(e)}")
                
        except Exception as e:
            log_error_with_traceback(
                logger,
                e,
                f"Security error in pattern processing for block {block_idx}, sequence {sequence_idx}"
            )
            raise SecurityError(f"Pattern processing failed: {str(e)}")

    def _process_with_node(self, 
                          node_id: int,
                          tensor: torch.Tensor,
                          block_idx: int,
                          sequence_idx: int,
                          language: str = "en") -> torch.Tensor:
        """Process tensor using a specific swarm node with nine-fold strategy alignment."""
        if node_id not in self._node_locks:
            raise SecurityError(f"Invalid node ID: {node_id}")
            
        with self._node_locks[node_id]:
            try:
                node = next(n for n in self.swarm_state.nodes if n.node_id == node_id)
                
                # Apply nine-fold strategy alignment
                try:
                    strategies = self._align_tensor_strategies(tensor)
                    aligned_tensor = self._select_best_strategy(
                        strategies,
                        (tensor.size(0), self.embedding_dim)
                    )
                except Exception as e:
                    log_error_with_traceback(
                        logger,
                        e,
                        f"Error in strategy alignment for tensor shape {tensor.shape}"
                    )
                    raise
                
                # Apply language-specific processing
                lang_weight = node.language_config.get(language, 0.5)
                if language not in self.language_patterns:
                    self.language_patterns[language] = self._generate_language_pattern(language)
                
                # Blend language-specific pattern with base pattern
                pattern_seed = self._generate_pattern_seed(block_idx, sequence_idx)
                base_pattern = self._apply_recursive_pattern(
                    aligned_tensor,
                    pattern_seed,
                    self.pattern_depth
                )
                
                # Ensure language pattern has correct shape
                lang_pattern = self.language_patterns[language]
                if lang_pattern.dim() == 1:
                    lang_pattern = lang_pattern.unsqueeze(0)
                
                # Apply nine-fold strategy alignment to language pattern
                try:
                    lang_strategies = self._align_tensor_strategies(lang_pattern)
                    aligned_lang_pattern = self._select_best_strategy(
                        lang_strategies,
                        (lang_pattern.size(0), self.embedding_dim)
                    )
                except Exception as e:
                    log_error_with_traceback(
                        logger,
                        e,
                        f"Error in language pattern alignment for shape {lang_pattern.shape}"
                    )
                    raise
                
                # Blend patterns
                processed = (1 - lang_weight) * base_pattern + lang_weight * aligned_lang_pattern
                
                # Apply resonant modes with language-aware parameters
                processed = self._apply_resonant_modes(
                    processed,
                    self.time_step,
                    language=language
                )
                
                # Update node's language specialization if successful
                if torch.norm(processed - aligned_tensor).item() < node.best_fitness:
                    node.language_specialization = language
                
                return processed
                
            except Exception as e:
                log_error_with_traceback(
                    logger,
                    e,
                    f"Error in node processing for node {node_id}, language {language}"
                )
                raise

    def _generate_language_pattern(self, language: str) -> torch.Tensor:
        """Generate language-specific pattern with geometric structure."""
        # Language-specific parameters with geometric structure
        lang_params = {
            "en": {"rhythm": 0.8, "harmony": 0.7, "complexity": 0.6},
            "zh": {"rhythm": 0.6, "harmony": 0.8, "complexity": 0.9},
            "ja": {"rhythm": 0.7, "harmony": 0.9, "complexity": 0.8},
            # Add more languages as needed
        }
        
        params = lang_params.get(language, {"rhythm": 0.5, "harmony": 0.5, "complexity": 0.5})
        
        # Generate base pattern with geometric structure
        base = torch.randn(self.embedding_dim)
        
        # Apply language-specific geometric transformations
        rhythm_mod = torch.sin(torch.linspace(0, 2 * np.pi * params["rhythm"], self.embedding_dim))
        harmony_mod = torch.cos(torch.linspace(0, 2 * np.pi * params["harmony"], self.embedding_dim))
        complexity_mod = torch.exp(-torch.linspace(0, params["complexity"], self.embedding_dim))
        
        # Combine transformations
        pattern = base * (1 + rhythm_mod + harmony_mod) * complexity_mod
        
        # Normalize to preserve geometric structure
        pattern = pattern / (pattern.norm() + 1e-6)
        
        return pattern

    def _update_swarm(self, fitness_function):
        """Update swarm state with selective reinforcement."""
        with self._lock:
            self.swarm_state.generation += 1
            
            # Update each node's position and velocity
            for node in self.swarm_state.nodes:
                with self._node_locks[node.node_id]:
                    # Skip nodes that are currently processing
                    if node.thread_id in self._query_futures:
                        continue
                    
                    # Update velocity with adaptive weights
                    r1 = torch.rand_like(node.velocity)
                    r2 = torch.rand_like(node.velocity)
                    
                    cognitive_component = self.swarm_state.cognitive_weight * r1 * (
                        node.best_position - node.position
                    )
                    social_component = self.swarm_state.social_weight * r2 * (
                        self.swarm_state.global_best_position - node.position
                    )
                    
                    node.velocity = (
                        self.swarm_state.inertia_weight * node.velocity +
                        cognitive_component +
                        social_component
                    )
                    
                    # Apply velocity clamping
                    velocity_norm = torch.norm(node.velocity)
                    if velocity_norm > self.swarm_state.velocity_clamp:
                        node.velocity *= self.swarm_state.velocity_clamp / velocity_norm
                    
                    # Update position
                    node.position += node.velocity
                    
                    # Evaluate fitness
                    fitness = fitness_function(node.position.unsqueeze(0))
                    if fitness < node.best_fitness:
                        node.best_position = node.position.clone()
                        node.best_fitness = fitness
                        
                        # Update global best if better
                        if fitness < self.swarm_state.global_best_fitness:
                            self.swarm_state.global_best_fitness = fitness
                            self.swarm_state.global_best_position = node.position.clone()
    
    def _check_rate_limit(self):
        """Implement rate limiting to prevent abuse."""
        current_time = time.time()
        with self._lock:
            # Remove old requests
            self._request_times = [t for t in self._request_times 
                                 if current_time - t < 1.0]
            
            # Check rate limit
            if len(self._request_times) >= self.security_config.rate_limit_requests:
                raise SecurityError("Rate limit exceeded")
            
            self._request_times.append(current_time)
    
    def _validate_input(self, tensor: torch.Tensor):
        """Validate input tensor to prevent resource exhaustion."""
        if tensor.numel() > self.security_config.max_pattern_size:
            raise SecurityError("Pattern size exceeds maximum allowed")
        
        if not torch.isfinite(tensor).all():
            raise SecurityError("Invalid tensor values detected")

    def update_emotional_state(self, success: bool, complexity: float):
        """Update the emotional state based on pattern generation success."""
        # Update emotional state based on success and complexity
        if success:
            self.emotional_state["intensity"] = min(1.0, self.emotional_state["intensity"] + 0.1)
            self.emotional_state["coherence"] = min(1.0, self.emotional_state["coherence"] + 0.05)
        else:
            self.emotional_state["intensity"] = max(0.1, self.emotional_state["intensity"] - 0.05)
            self.emotional_state["coherence"] = max(0.1, self.emotional_state["coherence"] - 0.1)
        
        # Update complexity based on input
        self.emotional_state["complexity"] = complexity
        
        # Update immersion level
        self.immersion_level = (
            self.emotional_state["intensity"] * 0.4 +
            self.emotional_state["complexity"] * 0.3 +
            self.emotional_state["coherence"] * 0.3
        )
        
        # Update creative potential
        self.creative_potential = min(1.0, self.creative_potential + 0.05 * success)

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass 