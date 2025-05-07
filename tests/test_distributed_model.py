import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from petals.client.config import ClientConfig
from petals.client.model import DistributedModelForCausalLM
from petals.data_structures import InferenceSession
from tests.mock_server import MockServer

logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    """Create a test configuration."""
    return ClientConfig(
        model_name_or_path="gpt2",
        initial_peers=[],
        inference_max_length=1024,
        request_timeout=30.0,
        session_timeout=60.0,
        step_timeout=5.0,
        server_timeout=30.0,
        health_check_interval=5.0,
    )

@pytest.fixture
def model(config):
    """Create a test model."""
    model = DistributedModelForCausalLM(
        config=config,
        block_indices=[0, 1, 2],
        max_retries=3,
    )
    model.start()
    yield model
    model.shutdown()

@pytest.fixture
def mock_server():
    """Create a mock server."""
    server = MockServer(
        model_name_or_path="gpt2",
        block_indices=[0, 1, 2],
        host="localhost",
        port=8000,
    )
    with server:
        yield server

def test_forward_pass(model, mock_server):
    """Test forward pass."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)

    # Run forward pass
    outputs = model.forward(hidden_states, attention_mask)

    # Check outputs
    assert outputs.shape == (batch_size, seq_length, hidden_size)
    assert not torch.isnan(outputs).any()
    assert not torch.isinf(outputs).any()

def test_backward_pass(model, mock_server):
    """Test backward pass."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
    attention_mask = torch.ones(batch_size, seq_length)
    grad_outputs = torch.randn(batch_size, seq_length, hidden_size)

    # Run backward pass
    grad_inputs = model.backward(hidden_states, grad_outputs, attention_mask)

    # Check gradients
    assert grad_inputs.shape == (batch_size, seq_length, hidden_size)
    assert not torch.isnan(grad_inputs).any()
    assert not torch.isinf(grad_inputs).any()

def test_multiple_blocks(model, mock_server):
    """Test multiple blocks."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)

    # Run forward pass through each block
    outputs = hidden_states
    for block_idx in range(len(model)):
        block = model[block_idx]
        outputs = block.forward(outputs, attention_mask)

    # Check outputs
    assert outputs.shape == (batch_size, seq_length, hidden_size)
    assert not torch.isnan(outputs).any()
    assert not torch.isinf(outputs).any()

def test_server_management(model, mock_server):
    """Test server management."""
    # Add server
    server_id = "test_server"
    model.add_server(server_id, mock_server.host, mock_server.port, mock_server.block_indices)

    # Check server info
    server_info = model.get_server_info(server_id)
    assert server_info is not None
    assert server_info.host == mock_server.host
    assert server_info.port == mock_server.port
    assert server_info.block_indices == mock_server.block_indices

    # Remove server
    model.remove_server(server_id)
    server_info = model.get_server_info(server_id)
    assert server_info is None

def test_error_handling(model, mock_server):
    """Test error handling."""
    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)

    # Test with no servers
    model.remove_server("test_server")
    with pytest.raises(RuntimeError):
        model.forward(hidden_states, attention_mask)

    # Test with invalid input
    with pytest.raises(ValueError):
        model.forward(torch.randn(1, 1, 1), attention_mask)

def test_session_management(model, mock_server):
    """Test session management."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create session
    session = InferenceSession(
        session_id="test_session",
        block_indices=[0, 1, 2],
        max_length=1024,
        timeout=30.0,
    )

    # Use session
    with model.use_session(session):
        # Create input
        batch_size = 2
        seq_length = 10
        hidden_size = 768
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)

        # Run forward pass
        outputs = model.forward(hidden_states, attention_mask)

        # Check outputs
        assert outputs.shape == (batch_size, seq_length, hidden_size)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

def test_async_operations(model, mock_server):
    """Test async operations."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)

    # Run async forward pass
    async def run_forward():
        return await model.forward_async(hidden_states, attention_mask)

    outputs = asyncio.run(run_forward())

    # Check outputs
    assert outputs.shape == (batch_size, seq_length, hidden_size)
    assert not torch.isnan(outputs).any()
    assert not torch.isinf(outputs).any()

def test_context_manager(model, mock_server):
    """Test context manager."""
    # Add server
    model.add_server("test_server", mock_server.host, mock_server.port, mock_server.block_indices)

    # Create input
    batch_size = 2
    seq_length = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    attention_mask = torch.ones(batch_size, seq_length)

    # Use context manager
    with model as m:
        # Run forward pass
        outputs = m.forward(hidden_states, attention_mask)

        # Check outputs
        assert outputs.shape == (batch_size, seq_length, hidden_size)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

    # Check that model is shut down
    assert not m._running 