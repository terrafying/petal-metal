from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import random
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Union, Tuple
from weakref import WeakMethod

import dijkstar
import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import PreTrainedModel

from petals.client.config import DistributedConfig, ClientConfig
from petals.client.routing.sequence_info import RemoteSequenceInfo
from petals.client.routing.spending_policy import NoSpendingPolicy
from petals.data_structures import ServerState, UID_DELIMITER, ModelInfo, ServerInfo, BlockInfo, InferenceSession
from petals.server.handler import TransformerConnectionHandler
from petals.utils.ping import PingAggregator
from petals.utils.random import sample_up_to

logger = get_logger(__name__)


class SequenceManagerError(Exception):
    """Base class for sequence manager errors."""
    pass

class ServerError(SequenceManagerError):
    """Error from a server."""
    pass

class NoServersError(SequenceManagerError):
    """No servers available."""
    pass

class RemoteSequenceManager:
    """Manages a sequence of remote transformer blocks."""

    def __init__(
        self,
        config: ClientConfig,
        block_indices: List[int],
        max_retries: int = 3,
    ):
        self.config = config
        self.block_indices = block_indices
        self.max_retries = max_retries
        self.servers: Dict[str, ServerInfo] = {}
        self.blocks: Dict[int, BlockInfo] = {}
        self.sessions: Dict[str, InferenceSession] = {}
        self.ping_aggregator = PingAggregator()
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None

    def start(self):
        """Start the sequence manager."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._health_check_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def _health_check_loop(self):
        """Monitor server health."""
        while self._running:
            try:
                # Ping all servers
                rtts = self.ping_aggregator.ping(list(self.servers.keys()))
                
                # Update server metrics
                with self._lock:
                    for server_id, rtt in rtts.items():
                        if server_id in self.servers:
                            self.servers[server_id].latency = rtt
                            self.servers[server_id].last_seen = time.time()

                    # Remove unresponsive servers
                    current_time = time.time()
                    for server_id, server in list(self.servers.items()):
                        if current_time - server.last_seen > self.config.server_timeout:
                            self._remove_server(server_id)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            time.sleep(self.config.health_check_interval)

    def _add_server(self, server_id: str, host: str, port: int, block_indices: List[int]):
        """Add a server."""
        with self._lock:
            self.servers[server_id] = ServerInfo(
                host=host,
                port=port,
                block_indices=block_indices,
            )
            for block_idx in block_indices:
                self.blocks[block_idx] = BlockInfo(
                    block_idx=block_idx,
                    server_id=server_id,
                )

    def _remove_server(self, server_id: str):
        """Remove a server."""
        with self._lock:
            if server_id in self.servers:
                # Remove server
                del self.servers[server_id]
                
                # Remove blocks
                for block_idx, block in list(self.blocks.items()):
                    if block.server_id == server_id:
                        del self.blocks[block_idx]

    def get_server_info(self, server_id: str) -> Optional[ServerInfo]:
        """Get server information."""
        with self._lock:
            return self.servers.get(server_id)

    def get_block_info(self, block_idx: int) -> Optional[BlockInfo]:
        """Get block information."""
        with self._lock:
            return self.blocks.get(block_idx)

    async def _make_request(
        self,
        server_id: str,
        request_type: str,
        **kwargs
    ) -> dict:
        """Make a request to a server."""
        server = self.get_server_info(server_id)
        if server is None:
            raise ServerError(f"Server {server_id} not found")

        # Create connection
        try:
            reader, writer = await asyncio.open_connection(server.host, server.port)
        except Exception as e:
            raise ServerError(f"Failed to connect to server {server_id}: {e}")

        try:
            # Send request
            request = {
                "type": request_type,
                **kwargs,
            }
            request_data = json.dumps(request).encode()
            writer.write(request_data)
            await writer.drain()

            # Read response
            response_data = await reader.read(1024)
            if not response_data:
                raise ServerError(f"Empty response from server {server_id}")

            # Parse response
            try:
                response = json.loads(response_data.decode())
            except json.JSONDecodeError as e:
                raise ServerError(f"Invalid JSON response from server {server_id}: {e}")

            # Check response status
            if response.get("status") != "ok":
                raise ServerError(f"Error from server {server_id}: {response.get('error')}")

            return response

        finally:
            writer.close()
            await writer.wait_closed()

    async def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Run a forward pass through the sequence."""
        if not self._running:
            raise SequenceManagerError("Sequence manager not started")

        # Get available blocks
        with self._lock:
            available_blocks = list(self.blocks.keys())
            if not available_blocks:
                raise NoServersError("No blocks available")

        # Try each server
        last_error = None
        for _ in range(self.max_retries):
            try:
                # Find server with most blocks
                server_blocks = {}
                for block_idx in available_blocks:
                    block = self.get_block_info(block_idx)
                    if block is not None:
                        server_blocks.setdefault(block.server_id, []).append(block_idx)

                if not server_blocks:
                    raise NoServersError("No servers available")

                # Use server with most blocks
                server_id = max(server_blocks.items(), key=lambda x: len(x[1]))[0]
                block_indices = server_blocks[server_id]

                # Make request
                response = await self._make_request(
                    server_id=server_id,
                    request_type="forward",
                    session_id=session_id,
                    block_indices=block_indices,
                    hidden_states=hidden_states.tolist(),
                    attention_mask=attention_mask.tolist() if attention_mask is not None else None,
                )

                # Convert outputs to tensor
                outputs = torch.tensor(response["outputs"])
                return outputs

            except Exception as e:
                last_error = e
                logger.error(f"Error in forward pass: {e}")
                continue

        raise last_error or SequenceManagerError("Failed to run forward pass")

    async def backward(
        self,
        hidden_states: torch.Tensor,
        grad_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Run a backward pass through the sequence."""
        if not self._running:
            raise SequenceManagerError("Sequence manager not started")

        # Get available blocks
        with self._lock:
            available_blocks = list(self.blocks.keys())
            if not available_blocks:
                raise NoServersError("No blocks available")

        # Try each server
        last_error = None
        for _ in range(self.max_retries):
            try:
                # Find server with most blocks
                server_blocks = {}
                for block_idx in available_blocks:
                    block = self.get_block_info(block_idx)
                    if block is not None:
                        server_blocks.setdefault(block.server_id, []).append(block_idx)

                if not server_blocks:
                    raise NoServersError("No servers available")

                # Use server with most blocks
                server_id = max(server_blocks.items(), key=lambda x: len(x[1]))[0]
                block_indices = server_blocks[server_id]

                # Make request
                response = await self._make_request(
                    server_id=server_id,
                    request_type="backward",
                    session_id=session_id,
                    block_indices=block_indices,
                    hidden_states=hidden_states.tolist(),
                    grad_outputs=grad_outputs.tolist(),
                    attention_mask=attention_mask.tolist() if attention_mask is not None else None,
                )

                # Convert gradients to tensor
                grad_inputs = torch.tensor(response["grad_inputs"]) if response["grad_inputs"] is not None else None
                return grad_inputs

            except Exception as e:
                last_error = e
                logger.error(f"Error in backward pass: {e}")
                continue

        raise last_error or SequenceManagerError("Failed to run backward pass")

    def shutdown(self):
        """Shutdown the sequence manager."""
        self._running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        with self._lock:
            self.servers.clear()
            self.blocks.clear()
            self.sessions.clear()

    def __getitem__(self, idx: int) -> "RemoteSequenceManager":
        """Get a sub-sequence of blocks."""
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self.block_indices) + idx
            if not 0 <= idx < len(self.block_indices):
                raise IndexError(f"Block index {idx} out of range")
            return RemoteSequenceManager(
                config=self.config,
                block_indices=[self.block_indices[idx]],
                max_retries=self.max_retries,
            )
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.block_indices))
            if step != 1:
                raise ValueError("Step must be 1")
            return RemoteSequenceManager(
                config=self.config,
                block_indices=self.block_indices[start:stop],
                max_retries=self.max_retries,
            )
        else:
            raise TypeError(f"Index must be int or slice, not {type(idx)}")

    def __iter__(self):
        """Iterate over the blocks."""
        return iter(self.block_indices)


class _SequenceManagerUpdateThread(threading.Thread):
    def __init__(self, update_period: float, ref_update_manager: WeakMethod):
        super().__init__(daemon=True)
        self.ref_update_manager = ref_update_manager
        self.ready = threading.Event()
        self.trigger = threading.Event()
        self.update_period = update_period
        self.should_shutdown = False

    def run(self) -> None:
        while not self.should_shutdown:
            update_manager = self.ref_update_manager()
            if update_manager is None:
                logger.debug(f"{self.__class__.__name__} exited because the sequence manager no longer exists")
                break

            try:
                self.trigger.clear()
                update_manager()
            except Exception as e:
                logger.exception(e)
            finally:
                del update_manager

            self.trigger.wait(self.update_period)

        logger.debug(f"{self.__class__.__name__} thread exited")

    def shutdown(self, timeout: Optional[float] = None):
        self.should_shutdown = True
        self.trigger.set()
        if self.is_alive():
            self.join(timeout)

    def __del__(self):
        self.shutdown()


def maybe_log_traceback(exc: Exception):
    traceback_level = logging.DEBUG if str(exc) or isinstance(exc, asyncio.TimeoutError) else logging.WARNING
    logger.log(traceback_level, "See detailed traceback below:", exc_info=True)


class MissingBlocksError(RuntimeError):
    def __init__(self, block_indices: Union[int, Sequence[int]]):
        super().__init__(
            f"No servers holding blocks {block_indices} are online. "
            f"You can check the public swarm's state at https://health.petals.dev "
            f"If there are not enough servers, please connect your GPU: "
            f"https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity "
        )
