import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class MockServer:
    """A mock server for testing."""

    def __init__(
        self,
        model_name_or_path: str,
        block_indices: List[int],
        host: str = "localhost",
        port: int = 8000,
    ):
        self.model_name_or_path = model_name_or_path
        self.block_indices = block_indices
        self.host = host
        self.port = port

        # Load model config
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)

        # Load model blocks
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.blocks = {}
        for block_idx in block_indices:
            self.blocks[block_idx] = self.model.transformer.h[block_idx]

        # Initialize server
        self.server = None
        self._running = False
        self._lock = threading.Lock()

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a connection."""
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

        if request_type == "forward":
            return await self._handle_forward(request)
        elif request_type == "backward":
            return await self._handle_backward(request)
        elif request_type == "info":
            return await self._handle_info(request)
        else:
            raise ValueError(f"Unknown request type: {request_type}")

    async def _handle_forward(self, request: dict) -> dict:
        """Handle a forward pass request."""
        # Get request parameters
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
            block = self.blocks.get(block_idx)
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
            "status": "ok",
            "outputs": outputs,
        }

    async def _handle_backward(self, request: dict) -> dict:
        """Handle a backward pass request."""
        # Get request parameters
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
            block = self.blocks.get(block_idx)
            if block is None:
                raise ValueError(f"Block {block_idx} not found")

            # Run backward pass
            outputs = block(outputs, attention_mask=attention_mask)
            outputs.backward(grad_outputs)

        # Get gradients
        grad_inputs = hidden_states.grad.tolist() if hidden_states.grad is not None else None

        return {
            "status": "ok",
            "grad_inputs": grad_inputs,
        }

    async def _handle_info(self, request: dict) -> dict:
        """Handle an info request."""
        return {
            "status": "ok",
            "info": {
                "model_name": self.model_name_or_path,
                "block_indices": self.block_indices,
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
            "status": "error",
            "error": error,
        })

    async def start(self):
        """Start the server."""
        if self._running:
            return

        self._running = True
        self.server = await asyncio.start_server(
            self.handle_connection,
            self.host,
            self.port,
        )

        async with self.server:
            await self.server.serve_forever()

    def shutdown(self):
        """Shutdown the server."""
        self._running = False
        if self.server is not None:
            self.server.close()

    def __enter__(self):
        """Start the server."""
        asyncio.run(self.start())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown the server."""
        self.shutdown() 