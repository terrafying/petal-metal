import asyncio
import math
import threading
import time
from functools import partial
from typing import Dict, Sequence

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

async def ping(server_id: str, *, wait_timeout: float = 5) -> float:
    """Ping a server to check its health."""
    try:
        # TODO: Implement actual ping using local discovery
        return 0.0
    except Exception as e:
        logger.debug(f"Failed to ping {server_id}:", exc_info=True)
        return math.inf

class PingAggregator:
    """Aggregates ping results for servers."""
    
    def __init__(self, *, ema_alpha: float = 0.2, expiration: float = 300):
        self.ema_alpha = ema_alpha
        self.expiration = expiration
        self.ping_emas = {}  # server_id -> (rtt, expiration_time)
        self.lock = threading.Lock()

    def ping(self, server_ids: Sequence[str], **kwargs) -> None:
        """Ping multiple servers and update their RTTs."""
        current_rtts = {}
        for server_id in server_ids:
            try:
                rtt = asyncio.run(ping(server_id, **kwargs))
                current_rtts[server_id] = rtt
            except Exception as e:
                logger.error(f"Error pinging {server_id}: {e}")
                current_rtts[server_id] = math.inf

        logger.debug(f"Current RTTs: {current_rtts}")

        with self.lock:
            expiration = time.time() + self.expiration
            for server_id, rtt in current_rtts.items():
                prev_rtt = self.ping_emas.get(server_id)
                if prev_rtt is not None and prev_rtt[0] != math.inf:
                    rtt = self.ema_alpha * rtt + (1 - self.ema_alpha) * prev_rtt[0]  # Exponential smoothing
                self.ping_emas[server_id] = (rtt, expiration)

    def to_dict(self) -> Dict[str, float]:
        """Get current RTTs for all servers."""
        with self.lock:
            current_time = time.time()
            smoothed_rtts = {
                server_id: rtt 
                for server_id, (rtt, exp) in self.ping_emas.items() 
                if exp > current_time
            }
            logger.debug(f"Smoothed RTTs: {smoothed_rtts}")
            return smoothed_rtts
