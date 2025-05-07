import pytest
import torch
import asyncio
from flock_generator import FlockGenerator, FlockConfig, FlockModel
from pattern_manager import PatternManager, SecurityConfig

@pytest.fixture
def flock_config():
    return FlockConfig(
        num_models=2,
        batch_size=4,
        max_length=64,
        temperature=0.7,
        top_p=0.9,
        pattern_depth=2,
        flock_cohesion=0.5,
        flock_separation=0.3,
        flock_alignment=0.4
    )

@pytest.fixture
def model_names():
    return ["gpt2", "distilgpt2"]

@pytest.fixture
def generator(model_names, flock_config):
    return FlockGenerator(model_names, flock_config)

@pytest.mark.asyncio
async def test_flock_model_generation():
    """Test individual model generation with pattern influence."""
    model = FlockModel("gpt2")
    prompt = "Once upon a time"
    
    result = await model.generate(
        prompt,
        max_length=32,
        temperature=0.7,
        top_p=0.9,
        pattern_influence=0.5
    )
    
    assert isinstance(result, str)
    assert len(result) > len(prompt)
    assert "Once upon a time" in result

@pytest.mark.asyncio
async def test_flock_generator_batch(generator):
    """Test batch generation with flocking behavior."""
    prompts = [
        "Once upon a time",
        "In a world where",
        "The future of AI",
        "When machines learn"
    ]
    
    results = await generator.generate_batch(prompts, num_iterations=2)
    
    assert len(results) == len(prompts)
    for prompt, result in zip(prompts, results):
        assert isinstance(result, str)
        assert len(result) > len(prompt)
        assert prompt in result

@pytest.mark.asyncio
async def test_flock_behavior(generator):
    """Test that flocking behavior affects outputs."""
    prompt = "The future of AI"
    prompts = [prompt] * 4
    
    # Generate without flocking
    generator.config.flock_cohesion = 0.0
    generator.config.flock_separation = 0.0
    generator.config.flock_alignment = 0.0
    results_no_flock = await generator.generate_batch(prompts, num_iterations=1)
    
    # Generate with flocking
    generator.config.flock_cohesion = 0.5
    generator.config.flock_separation = 0.3
    generator.config.flock_alignment = 0.4
    results_with_flock = await generator.generate_batch(prompts, num_iterations=1)
    
    # Results should be different
    assert results_no_flock != results_with_flock

@pytest.mark.asyncio
async def test_pattern_influence(generator):
    """Test that pattern influence affects outputs."""
    prompt = "The future of AI"
    prompts = [prompt] * 4
    
    # Generate with different pattern influences
    generator.models[0].pattern_manager.pattern_depth = 1
    results_shallow = await generator.generate_batch(prompts, num_iterations=1)
    
    generator.models[0].pattern_manager.pattern_depth = 3
    results_deep = await generator.generate_batch(prompts, num_iterations=1)
    
    # Results should be different
    assert results_shallow != results_deep

@pytest.mark.asyncio
async def test_concurrent_generation(generator):
    """Test concurrent generation across multiple models."""
    prompts = ["Test prompt"] * 8
    
    # Should handle concurrent generation without errors
    results = await generator.generate_batch(prompts, num_iterations=2)
    
    assert len(results) == len(prompts)
    assert all(isinstance(r, str) for r in results)

@pytest.mark.asyncio
async def test_flock_forces(generator):
    """Test individual flocking forces."""
    prompt = "The future of AI"
    prompts = [prompt] * 4
    
    # Test cohesion
    generator.config.flock_cohesion = 1.0
    generator.config.flock_separation = 0.0
    generator.config.flock_alignment = 0.0
    results_cohesion = await generator.generate_batch(prompts, num_iterations=1)
    
    # Test separation
    generator.config.flock_cohesion = 0.0
    generator.config.flock_separation = 1.0
    generator.config.flock_alignment = 0.0
    results_separation = await generator.generate_batch(prompts, num_iterations=1)
    
    # Test alignment
    generator.config.flock_cohesion = 0.0
    generator.config.flock_separation = 0.0
    generator.config.flock_alignment = 1.0
    results_alignment = await generator.generate_batch(prompts, num_iterations=1)
    
    # All results should be different
    assert results_cohesion != results_separation
    assert results_separation != results_alignment
    assert results_cohesion != results_alignment

@pytest.mark.asyncio
async def test_error_handling(generator):
    """Test error handling in generation."""
    # Test with invalid prompt
    with pytest.raises(Exception):
        await generator.generate_batch([""], num_iterations=1)
    
    # Test with invalid model
    generator.models[0].model = None
    with pytest.raises(Exception):
        await generator.generate_batch(["Test"], num_iterations=1) 