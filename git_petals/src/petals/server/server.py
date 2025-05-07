from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, PreTrainedModel

from petals.server.config import ServerConfig
from petals.server.handler import TransformerConnectionHandler
from petals.utils.ping import PingAggregator

logger = logging.getLogger(__name__)

class Server:
    """A server that hosts model blocks."""

    def __init__(
        self,
        config: ServerConfig,
        module_backends: Dict[int, PreTrainedModel],
    ):
        self.config = config
        self.module_backends = module_backends
        self.handlers: List[TransformerConnectionHandler] = []
        self._ping_aggregator = PingAggregator()
        self._running = False
        self._lock = threading.Lock()

    def _start_handlers(self):
        """Start the connection handlers."""
        for _ in range(self.config.num_handlers):
            handler = TransformerConnectionHandler(
                config=self.config,
                module_backends=self.module_backends,
            )
            self.handlers.append(handler)

    async def start(self):
        """Start the server."""
        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        self._running = True
        self._start_handlers()

        # Start health check thread
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

        # Start server
        server = await asyncio.start_server(
            self._handle_connection,
            self.config.host,
            self.config.port,
        )

        async with server:
            await server.serve_forever()

    def _health_check_loop(self):
        """Background thread to check server health."""
        while self._running:
            try:
                # Update server metrics
                with self._lock:
                    for handler in self.handlers:
                        handler.update_metrics()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            time.sleep(self.config.health_check_interval)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new connection."""
        try:
            # Get next available handler
            with self._lock:
                handler = min(self.handlers, key=lambda h: h.num_connections)
                handler.num_connections += 1

            # Handle connection
            await handler.handle_connection(reader, writer)
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            # Release handler
            with self._lock:
                handler.num_connections -= 1

    async def shutdown(self):
        """Shutdown the server."""
        logger.info("Shutting down server")
        self._running = False

        # Stop health check thread
        if hasattr(self, "_health_check_thread"):
            self._health_check_thread.join(timeout=1.0)

        # Stop handlers
        for handler in self.handlers:
            await handler.shutdown()
