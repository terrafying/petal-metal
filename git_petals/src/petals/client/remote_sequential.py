from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple, Union

import torch
from hivemind.utils.logging import get_logger
from transformers import PreTrainedModel

from petals.client.config import DistributedConfig
from petals.client.inference_session import InferenceSession
from petals.client.routing.sequence_manager import RemoteSequenceManager
from petals.client.sequential_autograd import _RemoteSequentialAutogradFunction
from petals.data_structures import UID_DELIMITER, ModelInfo, ServerInfo

logger = get_logger(__name__)

_active_session = ContextVar("active_session", default=None)

class RemoteSequential(torch.nn.Module):
    """
    A sequence of transformer blocks hosted by the swarm.
    """

    def __init__(
        self,
        config: DistributedConfig,
        model_info: ModelInfo,
        max_retries: int = 3,
    ):
        super().__init__()
        self.config = config
        self.model_info = model_info
        self.max_retries = max_retries

        # Initialize sequence manager
        self.sequence_manager = RemoteSequenceManager(
            config=config,
            model_info=model_info,
            max_retries=max_retries,
        )

    def start(self):
        """Start the remote sequential."""
        self.sequence_manager.start()

    async def forward_async(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a forward pass through the sequence."""
        if not self.sequence_manager.running:
            raise RuntimeError("Remote sequential not started")

        # Get active session
        session = _active_session.get()
        if session is None:
            session = InferenceSession(
                session_id=None,
                block_indices=self.model_info.block_indices,
                max_length=self.config.inference_max_length,
                timeout=self.config.request_timeout,
            )

        # Run forward pass
        try:
            outputs = await self.sequence_manager.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                session_id=session.session_id,
            )
            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a forward pass through the sequence."""
        return asyncio.run(self.forward_async(hidden_states, attention_mask))

    @property
    def active_session(self) -> Optional[InferenceSession]:
        """
        If called inside `with model.inference_session(...):` or `with model.use_session(...):`,
        returns an active InferenceSession. Otherwise, returns None.
        """

        return _active_session.get()

    @property
    def position(self) -> int:
        """Returns the prefix length (in tokens) in the active inference session or zero if no session is active."""

        return self.active_session.position if self.active_session is not None else 0

    @contextmanager
    def use_session(self, session: Optional[InferenceSession]) -> InferenceSession:
        """Inside this context, forward() will use an _existing_ InferenceSession provided as the argument."""

        token = _active_session.set(session)
        try:
            yield session
        finally:
            _active_session.reset(token)

    @contextmanager
    def inference_session(self, **kwargs) -> InferenceSession:
        """
        Inside this context, forward() will use a _new_ InferenceSession created with given parameters.

        :param max_length: Maximal expected length of inference results. Servers use this parameter
                           to calculate the size of attention caches allocated to this client.
        """

        with InferenceSession(self.sequence_manager, **kwargs) as session, self.use_session(session):
            yield session

    def __getitem__(self, idx: Union[int, slice]) -> "RemoteSequential":
        """Get a slice of the sequence."""
        if isinstance(idx, int):
            if idx < 0:
                idx = self.model_info.num_blocks + idx
            if not 0 <= idx < self.model_info.num_blocks:
                raise IndexError(f"Block index {idx} out of range [0, {self.model_info.num_blocks})")
            return self.sequence_manager.get_block(idx)
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.model_info.num_blocks
            step = idx.step if idx.step is not None else 1
            if start < 0:
                start = self.model_info.num_blocks + start
            if stop < 0:
                stop = self.model_info.num_blocks + stop
            if not 0 <= start < self.model_info.num_blocks:
                raise IndexError(f"Start index {start} out of range [0, {self.model_info.num_blocks})")
            if not 0 <= stop <= self.model_info.num_blocks:
                raise IndexError(f"Stop index {stop} out of range [0, {self.model_info.num_blocks}]")
            if step != 1:
                raise ValueError("Step must be 1")
            return self.sequence_manager.get_blocks(start, stop)
        else:
            raise TypeError(f"Index must be int or slice, not {type(idx)}")

    def __iter__(self):
        """Iterate over the blocks."""
        for i in range(self.model_info.num_blocks):
            yield self[i]

    def __len__(self) -> int:
        return len(self.sequence_manager)

    def extra_repr(self) -> str:
        return f"modules={self.sequence_manager.block_uids[0]}..{self.sequence_manager.block_uids[-1]}"

    def add_server(self, server_id: str, server_info: ServerInfo):
        """Add a server to the sequence manager."""
        self.sequence_manager.add_server(server_id, server_info)

    def remove_server(self, server_id: str):
        """Remove a server from the sequence manager."""
        self.sequence_manager.remove_server(server_id)

    def get_server_info(self, server_id: str) -> Optional[ServerInfo]:
        """Get information about a server."""
        return self.sequence_manager.get_server_info(server_id)

    async def backward_async(
        self,
        hidden_states: torch.Tensor,
        grad_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a backward pass through the sequence."""
        if not self.sequence_manager.running:
            raise RuntimeError("Remote sequential not started")

        # Get active session
        session = _active_session.get()
        if session is None:
            session = InferenceSession(
                session_id=None,
                block_indices=self.model_info.block_indices,
                max_length=self.config.inference_max_length,
                timeout=self.config.request_timeout,
            )

        # Run backward pass
        try:
            grad_inputs = await self.sequence_manager.backward(
                hidden_states=hidden_states,
                grad_outputs=grad_outputs,
                attention_mask=attention_mask,
                session_id=session.session_id,
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
        """Run a backward pass through the sequence."""
        return asyncio.run(self.backward_async(hidden_states, grad_outputs, attention_mask))

    def shutdown(self):
        """Shutdown the remote sequential."""
        self.sequence_manager.shutdown()

    def __enter__(self):
        """Start the remote sequential."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown the remote sequential."""
        self.shutdown()
