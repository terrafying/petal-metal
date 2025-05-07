import asyncio
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from petals.client.config import DistributedConfig, ClientConfig
from petals.client.remote_sequential import RemoteSequential
from petals.data_structures import ModelInfo, ServerInfo, InferenceSession

logger = logging.getLogger(__name__)

class DistributedModelForCausalLM(torch.nn.Module):
    """A distributed causal language model."""

    def __init__(
        self,
        config: ClientConfig,
        block_indices: List[int],
        max_retries: int = 3,
    ):
        super().__init__()
        self.config = config
        self.block_indices = block_indices
        self.max_retries = max_retries

        # Load model config
        self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.remote_sequential = RemoteSequential(
            config=config,
            block_indices=block_indices,
            max_retries=max_retries,
        )
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start the model."""
        if self._running:
            return

        self._running = True
        self.remote_sequential.start()

    async def forward_async(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass through the model."""
        if not self._running:
            raise RuntimeError("Model not started")

        # Run forward pass
        try:
            outputs = await self.remote_sequential.forward_async(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass through the model."""
        return asyncio.run(self.forward_async(hidden_states, attention_mask))

    async def backward_async(
        self,
        hidden_states: torch.Tensor,
        grad_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a backward pass through the model."""
        if not self._running:
            raise RuntimeError("Model not started")

        # Run backward pass
        try:
            grad_inputs = await self.remote_sequential.backward_async(
                hidden_states=hidden_states,
                grad_outputs=grad_outputs,
                attention_mask=attention_mask,
            )
            return grad_inputs
        except Exception as e:
            logger.error(f"Error in backward pass: {e}")
            raise

    def backward(
        self,
        hidden_states: torch.Tensor,
        grad_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a backward pass through the model."""
        return asyncio.run(self.backward_async(hidden_states, grad_outputs, attention_mask))

    def add_server(self, server_id: str, host: str, port: int, block_indices: List[int]):
        """Add a server."""
        self.remote_sequential.add_server(server_id, host, port, block_indices)

    def remove_server(self, server_id: str):
        """Remove a server."""
        self.remote_sequential.remove_server(server_id)

    def get_server_info(self, server_id: str):
        """Get server information."""
        return self.remote_sequential.get_server_info(server_id)

    def get_block_info(self, block_idx: int):
        """Get block information."""
        return self.remote_sequential.get_block_info(block_idx)

    def __getitem__(self, idx: int) -> "DistributedModelForCausalLM":
        """Get a sub-sequence of blocks."""
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self.block_indices) + idx
            if not 0 <= idx < len(self.block_indices):
                raise IndexError(f"Block index {idx} out of range")
            return DistributedModelForCausalLM(
                config=self.config,
                block_indices=[self.block_indices[idx]],
                max_retries=self.max_retries,
            )
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.block_indices))
            if step != 1:
                raise ValueError("Step must be 1")
            return DistributedModelForCausalLM(
                config=self.config,
                block_indices=self.block_indices[start:stop],
                max_retries=self.max_retries,
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self) -> int:
        """Get the number of blocks."""
        return len(self.block_indices)

    def __iter__(self):
        """Iterate over the blocks."""
        return iter(self.block_indices)

    def shutdown(self):
        """Shutdown the model."""
        self._running = False
        self.remote_sequential.shutdown()

    def __enter__(self):
        """Start the model."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown the model."""
        self.shutdown() 