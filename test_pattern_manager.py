import pytest
import torch
import asyncio
import numpy as np
import time
from pattern_manager import PatternManager, ResonantMode, SecurityConfig, SecurityError
from server import Server, ServerConfig
from client import Client, ClientConfig

@pytest.fixture
def security_config():
    return SecurityConfig(
        max_recursion_depth=5,
        max_pattern_size=1024,
        rate_limit_requests=10,
        pattern_timeout=0.1,
    )

@pytest.fixture
def pattern_manager():
    return PatternManager(
        base_seed=42, 
        pattern_depth=3,
        num_resonant_modes=5,
        swarm_size=10,
    )

@pytest.fixture
def server_config():
    return ServerConfig(
        model_name_or_path="gpt2",
        block_indices=[0, 1],
        pattern_depth=3,
        base_seed=42
    )

@pytest.fixture
def client_config():
    return ClientConfig(
        model_name_or_path="gpt2",
        block_indices=[0, 1],
        pattern_depth=3,
        base_seed=42
    )

def test_pattern_manager_consistency(pattern_manager):
    """Test that pattern manager produces consistent patterns"""
    # Create test input
    input_tensor = torch.randn(2, 3, 768)
    
    # Process through same block and sequence multiple times
    output1 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    output2 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    
    # Should be identical
    assert torch.allclose(output1, output2)
    
    # Different sequence should be different
    output3 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=1)
    assert not torch.allclose(output1, output3)

def test_pattern_manager_block_difference(pattern_manager):
    """Test that different blocks produce different patterns"""
    input_tensor = torch.randn(2, 3, 768)
    
    output1 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    output2 = pattern_manager.process_output(input_tensor, block_idx=1, sequence_idx=0)
    
    assert not torch.allclose(output1, output2)

def test_resonant_modes(pattern_manager):
    """Test that resonant modes affect the output"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Process with different time steps
    output1 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    output2 = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=1)
    
    # Should be different due to resonant modes
    assert not torch.allclose(output1, output2)
    
    # Test mode parameters
    mode = pattern_manager.resonant_modes[0]
    assert 0.1 <= mode.frequency <= 2.0
    assert 0.1 <= mode.amplitude <= 1.0
    assert 0 <= mode.phase <= 2 * np.pi
    assert 0.1 <= mode.decay <= 0.5

def test_swarm_behavior(pattern_manager):
    """Test that swarm optimization affects the output"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Process multiple times to allow swarm to evolve
    outputs = []
    for i in range(5):
        output = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=i)
        outputs.append(output)
    
    # Check that swarm state is being updated
    assert pattern_manager.swarm_state.global_best_position is not None
    assert pattern_manager.swarm_state.global_best_fitness < float('inf')
    
    # Verify that outputs are different due to swarm evolution
    for i in range(len(outputs)-1):
        assert not torch.allclose(outputs[i], outputs[i+1])

def test_stochastic_synthesis(pattern_manager):
    """Test that stochastic synthesis produces diverse patterns"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Generate multiple patterns
    patterns = []
    for i in range(10):
        pattern = pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=i)
        patterns.append(pattern)
    
    # Calculate pattern diversity
    pattern_tensor = torch.stack(patterns)
    pattern_means = pattern_tensor.mean(dim=0)
    pattern_stds = pattern_tensor.std(dim=0)
    
    # Verify that patterns have non-zero variance
    assert torch.all(pattern_stds > 0)
    
    # Verify that patterns are centered around input
    assert torch.allclose(pattern_means, input_tensor, atol=1.0)

@pytest.mark.asyncio
async def test_server_client_pattern_consistency(server_config, client_config):
    """Test that server and client produce consistent patterns"""
    # Initialize server and client
    server = Server(server_config)
    client = Client(client_config)
    
    # Create test input
    input_tensor = torch.randn(2, 3, 768)
    
    # Get outputs from both
    server_output = await server.blocks[0].forward(input_tensor)
    client_output = await client.forward(input_tensor, block_idx=0)
    
    # Should be identical
    assert torch.allclose(server_output, client_output)
    
    # Cleanup
    await server.shutdown()
    await client.close()

@pytest.mark.asyncio
async def test_multi_block_pattern_consistency(server_config, client_config):
    """Test pattern consistency across multiple blocks"""
    server = Server(server_config)
    client = Client(client_config)
    
    input_tensor = torch.randn(2, 3, 768)
    
    # Process through multiple blocks
    server_outputs = []
    client_outputs = []
    
    for block_idx in server_config.block_indices:
        server_output = await server.blocks[block_idx].forward(input_tensor)
        client_output = await client.forward(input_tensor, block_idx)
        
        server_outputs.append(server_output)
        client_outputs.append(client_output)
    
    # Compare outputs
    for s_out, c_out in zip(server_outputs, client_outputs):
        assert torch.allclose(s_out, c_out)
    
    # Cleanup
    await server.shutdown()
    await client.close()

def test_pattern_manager_seed_consistency():
    """Test that different seeds produce different patterns"""
    pm1 = PatternManager(base_seed=42, pattern_depth=3)
    pm2 = PatternManager(base_seed=43, pattern_depth=3)
    
    input_tensor = torch.randn(2, 3, 768)
    
    output1 = pm1.process_output(input_tensor, block_idx=0, sequence_idx=0)
    output2 = pm2.process_output(input_tensor, block_idx=0, sequence_idx=0)
    
    assert not torch.allclose(output1, output2)

def test_pattern_manager_depth_effect():
    """Test that pattern depth affects output"""
    pm1 = PatternManager(base_seed=42, pattern_depth=2)
    pm2 = PatternManager(base_seed=42, pattern_depth=3)
    
    input_tensor = torch.randn(2, 3, 768)
    
    output1 = pm1.process_output(input_tensor, block_idx=0, sequence_idx=0)
    output2 = pm2.process_output(input_tensor, block_idx=0, sequence_idx=0)
    
    assert not torch.allclose(output1, output2)

def test_rate_limiting(pattern_manager):
    """Test that rate limiting prevents abuse"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Should work for first 10 requests
    for _ in range(10):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    
    # Should fail on 11th request
    with pytest.raises(SecurityError, match="Rate limit exceeded"):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)

def test_input_validation(pattern_manager):
    """Test that input validation prevents resource exhaustion"""
    # Test oversized input
    large_tensor = torch.randn(1000, 1000, 1000)
    with pytest.raises(SecurityError, match="Pattern size exceeds maximum allowed"):
        pattern_manager.process_output(large_tensor, block_idx=0, sequence_idx=0)
    
    # Test invalid values
    invalid_tensor = torch.tensor([float('inf'), float('nan')])
    with pytest.raises(SecurityError, match="Invalid tensor values detected"):
        pattern_manager.process_output(invalid_tensor, block_idx=0, sequence_idx=0)

def test_recursion_depth_limit(pattern_manager):
    """Test that recursion depth is limited"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Should work with normal depth
    pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)
    
    # Should fail with excessive depth
    pattern_manager.pattern_depth = 20
    with pytest.raises(SecurityError, match="Pattern generation timeout"):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)

def test_pattern_timeout(pattern_manager):
    """Test that pattern generation times out"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Make pattern generation take too long
    def slow_fitness(positions):
        time.sleep(0.2)  # Longer than timeout
        return torch.norm(positions, dim=1)
    
    pattern_manager._update_swarm = lambda x: slow_fitness(x)
    
    with pytest.raises(SecurityError, match="Swarm optimization timeout"):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)

def test_swarm_validation(pattern_manager):
    """Test that swarm modulation is validated"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Create invalid swarm state
    pattern_manager.swarm_state.global_best_position = torch.tensor([float('inf')])
    
    with pytest.raises(SecurityError, match="Invalid swarm modulation detected"):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0)

def test_output_validation(pattern_manager):
    """Test that output is validated before returning"""
    input_tensor = torch.randn(2, 3, 768)
    
    # Mock pattern generation to produce invalid output
    def mock_recursive_pattern(*args, **kwargs):
        return torch.tensor([float('inf')])
    
    pattern_manager._apply_recursive_pattern = mock_recursive_pattern
    
    with pytest.raises(SecurityError, match="Invalid tensor values detected"):
        pattern_manager.process_output(input_tensor, block_idx=0, sequence_idx=0) 