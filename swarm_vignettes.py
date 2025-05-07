import torch
import asyncio
from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager, SecurityConfig
from flock_generator import FlockConfig, FlockGenerator
from shape_visualizer import ShapeVisualizer, ShapeConfig

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
    
    # Initialize pattern manager with model
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=3,
        base_seed=42
    )
    
    prompts = ["Hello", "你好"]
    languages = ["en", "zh"]
    
    # Generate results
    results = await generator.generate_batch(
        prompts=prompts,
        languages=languages,
        num_iterations=2
    )
    
    # Visualize language patterns
    visualizer = ShapeVisualizer()
    
    # Get embeddings for visualization
    embeddings = []
    for result in results:
        tokens = generator.models[0].tokenizer(result, return_tensors="pt")
        with torch.no_grad():
            # Get input embeddings and ensure correct shape
            emb = generator.models[0].model.get_input_embeddings()(tokens.input_ids)
            # Reshape to (sequence_length, embedding_dim)
            emb = emb.squeeze(0)  # Remove batch dimension
            # Average over sequence length to get (embedding_dim,)
            emb = emb.mean(dim=0)
            # Ensure 2D shape for visualization
            emb = emb.unsqueeze(0)  # Add batch dimension
        embeddings.append(emb)
    
    # Stack embeddings to create (num_samples, embedding_dim) tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    
    # Create visualizations
    embedding_fig = visualizer.visualize_embedding_space(
        embeddings_tensor,
        languages
    )
    
    language_fig = visualizer.visualize_language_matrix(
        config.language_weights,
        generator.pattern_manager.coupling_matrix
    )
    
    return results, embedding_fig, language_fig

# Vignette 2: Pattern Evolution Swarm
async def pattern_evolution_swarm():
    """Demonstrates a swarm focused on pattern evolution and adaptation."""
    config = FlockConfig(
        num_models=3,
        pattern_depth=4,
        flock_cohesion=0.7,
        flock_separation=0.2
    )
    
    generator = FlockGenerator(
        model_names=["gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M"],
        config=config
    )
    
    # Initialize pattern manager with model
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=config.pattern_depth,
        base_seed=42
    )
    
    base_prompt = "The pattern evolves:"
    prompts = [base_prompt] * 3
    
    # Track pattern evolution
    patterns = []
    timesteps = []
    
    # Generate with pattern tracking
    for i in range(4):
        results = await generator.generate_batch(
            prompts=prompts,
            num_iterations=1
        )
        
        # Get patterns
        for result in results:
            tokens = generator.models[0].tokenizer(result, return_tensors="pt")
            with torch.no_grad():
                # Get input embeddings and ensure correct shape
                pattern = generator.models[0].model.get_input_embeddings()(tokens.input_ids)
                # Reshape to (sequence_length, embedding_dim)
                pattern = pattern.squeeze(0)  # Remove batch dimension
                # Average over sequence length to get (embedding_dim,)
                pattern = pattern.mean(dim=0)
            patterns.append(pattern)
            timesteps.append(i)
    
    # Visualize pattern evolution
    visualizer = ShapeVisualizer()
    pattern_fig = visualizer.visualize_pattern_evolution(patterns, timesteps)
    
    return results, pattern_fig

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
    
    # Initialize pattern manager with model
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=3,
        base_seed=42
    )
    
    prompts = ["Efficient generation:", "Resource aware:", "Optimized:", "Minimal:"]
    
    # Track resource usage
    positions = []
    velocities = []
    languages = ["en"] * len(prompts)
    
    results = await generator.generate_batch(
        prompts=prompts,
        num_iterations=2
    )
    
    # Get swarm dynamics
    for node in generator.pattern_manager.swarm_state.nodes:
        # Ensure position and velocity are 3D vectors
        pos = node.position[:3] if len(node.position) > 3 else node.position
        vel = node.velocity[:3] if len(node.velocity) > 3 else node.velocity
        positions.append(pos)
        velocities.append(vel)
    
    # Visualize swarm dynamics
    visualizer = ShapeVisualizer()
    swarm_fig = visualizer.visualize_swarm_dynamics(positions, velocities, languages)
    
    return results, swarm_fig

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
    
    # Initialize pattern manager with model
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=3,
        base_seed=42
    )
    
    emotional_prompts = [
        "The joy of discovery:",
        "The wonder of creation:",
        "The harmony of nature:",
        "The beauty of simplicity:"
    ]
    
    # Track emotional states
    emotional_states = []
    timesteps = []
    
    for i in range(3):
        results = await generator.generate_batch(
            prompts=emotional_prompts,
            num_iterations=1
        )
        
        # Record emotional state
        emotional_states.append(generator.pattern_manager.emotional_state.copy())
        timesteps.append(i)
    
    # Visualize emotional evolution
    visualizer = ShapeVisualizer()
    emotional_fig = visualizer.visualize_emotional_state(emotional_states, timesteps)
    
    return results, emotional_fig

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
    
    # Initialize pattern manager with model
    generator.pattern_manager = PatternManager(
        model=generator.models[0].model,
        pattern_depth=3,
        base_seed=42
    )
    
    multilingual_prompts = [
        "Harmony in diversity:",
        "Unity in language:",
        "Cross-cultural bridge:"
    ]
    
    languages = ["en", "zh", "ja"]
    
    # Generate results
    results = await generator.generate_batch(
        prompts=multilingual_prompts,
        languages=languages,
        num_iterations=3
    )
    
    # Get embeddings for visualization
    embeddings = []
    for result in results:
        tokens = generator.models[0].tokenizer(result, return_tensors="pt")
        with torch.no_grad():
            # Get input embeddings and ensure correct shape
            emb = generator.models[0].model.get_input_embeddings()(tokens.input_ids)
            # Reshape to (sequence_length, embedding_dim)
            emb = emb.squeeze(0)  # Remove batch dimension
            # Average over sequence length to get (embedding_dim,)
            emb = emb.mean(dim=0)
            # Ensure 2D shape for visualization
            emb = emb.unsqueeze(0)  # Add batch dimension
        embeddings.append(emb)
    
    # Stack embeddings to create (num_samples, embedding_dim) tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    
    # Create visualizations
    visualizer = ShapeVisualizer()
    
    # Visualize embedding space
    embedding_fig = visualizer.visualize_embedding_space(
        embeddings_tensor,
        languages
    )
    
    # Visualize language matrix
    language_fig = visualizer.visualize_language_matrix(
        config.language_weights,
        generator.pattern_manager.coupling_matrix
    )
    
    return results, embedding_fig, language_fig

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
        print(f"\n=== {name} ===")
        try:
            results = await vignette()
            if isinstance(results, tuple):
                text_results = results[0]
                figures = results[1:]
                print("\nGenerated Text:")
                for i, result in enumerate(text_results):
                    print(f"\nResult {i + 1}:")
                    print(result)
                print("\nVisualizations generated. Check the plots.")
            else:
                for i, result in enumerate(results):
                    print(f"\nResult {i + 1}:")
                    print(result)
        except Exception as e:
            raise Exception(f"Error in vignette {name}: {e}") from e

if __name__ == "__main__":
    asyncio.run(run_vignettes()) 