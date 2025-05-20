from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import multiprocessing as mp
import sys
from enum import Enum
from itertools import chain
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from async_timeout import timeout
from hivemind.utils.logging import get_logger
from transformers import PreTrainedModel

from petals.data_structures import UID_DELIMITER, InferenceSession
from petals.server.backend import TransformerBackend
from petals.server.block_functions import iterate_rpc_inference, run_rpc_backward, run_rpc_forward
from petals.server.task_prioritizer import DummyTaskPrioritizer, TaskPrioritizerBase
from petals.utils.convert_block import QuantType
from petals.server.config import ServerConfig

logger = get_logger(__name__)

CACHE_TOKENS_AVAILABLE = "cache_tokens_available"

class Event(Enum):
    NEW_SESSION = 0
    END_SESSION = 1
    PUSH = 2
    SHUTDOWN = 3

@dataclass
class RequestType:
    """Request types for the server."""
    FORWARD = "forward"
    BACKWARD = "backward"
    INFO = "info"

@dataclass
class ResponseStatus:
    """Response status codes."""
    OK = "ok"
    ERROR = "error"

class TransformerConnectionHandler:
    """Handles connections for transformer blocks."""

    def __init__(
        self,
        config: ServerConfig,
        module_backends: Dict[int, PreTrainedModel],
    ):
        self.config = config
        self.module_backends = module_backends
        self.sessions: Dict[str, InferenceSession] = {}
        self.num_connections = 0
        self._lock = threading.Lock()
        self._running = False

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a connection."""
        self._running = True
        try:
            while self._running:
                # Read request
                request_data = await reader.read(1024)
                if not request_data:
                    break

                # Parse request
                try:
                    request = json.loads(request_data.decode())
                except json.JSONDecodeError as e:
                    await self._send_error(writer, f"Invalid JSON: {e}")
                    continue

                # Process request
                try:
                    response = await self._process_request(request)
                    await self._send_response(writer, response)
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    await self._send_error(writer, str(e))

        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_request(self, request: dict) -> dict:
        """Process a request."""
        request_type = request.get("type")
        if not request_type:
            raise ValueError("Request type not specified")

        if request_type == RequestType.FORWARD:
            return await self._handle_forward(request)
        elif request_type == RequestType.BACKWARD:
            return await self._handle_backward(request)
        elif request_type == RequestType.INFO:
            return await self._handle_info(request)
        else:
            raise ValueError(f"Unknown request type: {request_type}")

    async def _handle_forward(self, request: dict) -> dict:
        """Handle a forward pass request."""
        # Get request parameters
        session_id = request.get("session_id")
        block_indices = request.get("block_indices", [])
        hidden_states = request.get("hidden_states")
        attention_mask = request.get("attention_mask")

        # Validate parameters
        if not block_indices:
            raise ValueError("Block indices not specified")
        if hidden_states is None:
            raise ValueError("Hidden states not specified")

        # Convert inputs to tensors
        hidden_states = torch.tensor(hidden_states)
        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask)

        # Process each block
        outputs = hidden_states
        for block_idx in block_indices:
            # Get block
            block = self.module_backends.get(block_idx)
            if block is None:
                raise ValueError(f"Block {block_idx} not found")

            # Run forward pass
            with torch.no_grad():
                outputs = block(outputs, attention_mask=attention_mask)

        # Convert outputs to lists
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.tolist()
        elif isinstance(outputs, tuple):
            outputs = [o.tolist() if isinstance(o, torch.Tensor) else o for o in outputs]

        return {
            "status": ResponseStatus.OK,
            "outputs": outputs,
        }

    async def _handle_backward(self, request: dict) -> dict:
        """Handle a backward pass request."""
        # Get request parameters
        session_id = request.get("session_id")
        block_indices = request.get("block_indices", [])
        hidden_states = request.get("hidden_states")
        grad_outputs = request.get("grad_outputs")
        attention_mask = request.get("attention_mask")

        # Validate parameters
        if not block_indices:
            raise ValueError("Block indices not specified")
        if hidden_states is None:
            raise ValueError("Hidden states not specified")
        if grad_outputs is None:
            raise ValueError("Gradient outputs not specified")

        # Convert inputs to tensors
        hidden_states = torch.tensor(hidden_states, requires_grad=True)
        grad_outputs = torch.tensor(grad_outputs)
        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask)

        # Process each block in reverse
        outputs = hidden_states
        for block_idx in reversed(block_indices):
            # Get block
            block = self.module_backends.get(block_idx)
            if block is None:
                raise ValueError(f"Block {block_idx} not found")

            # Run backward pass
            outputs = block(outputs, attention_mask=attention_mask)
            outputs.backward(grad_outputs)

        # Get gradients
        grad_inputs = hidden_states.grad.tolist() if hidden_states.grad is not None else None

        return {
            "status": ResponseStatus.OK,
            "grad_inputs": grad_inputs,
        }

    async def _handle_info(self, request: dict) -> dict:
        """Handle an info request."""
        return {
            "status": ResponseStatus.OK,
            "info": {
                "model_name": self.config.model_name_or_path,
                "block_indices": self.config.block_indices,
                "num_handlers": self.config.num_handlers,
                "inference_max_length": self.config.inference_max_length,
            },
        }

    async def _send_response(self, writer: asyncio.StreamWriter, response: dict):
        """Send a response."""
        try:
            response_data = json.dumps(response).encode()
            writer.write(response_data)
            await writer.drain()
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    async def _send_error(self, writer: asyncio.StreamWriter, error: str):
        """Send an error response."""
        await self._send_response(writer, {
            "status": ResponseStatus.ERROR,
            "error": error,
        })

    def update_metrics(self):
        """Update server metrics."""
        with self._lock:
            # Update session metrics
            current_time = time.time()
            for session_id, session in list(self.sessions.items()):
                if current_time - session.last_seen > self.config.session_timeout:
                    del self.sessions[session_id]

    async def shutdown(self):
        """Shutdown the handler."""
        self._running = False
        with self._lock:
            self.sessions.clear()
