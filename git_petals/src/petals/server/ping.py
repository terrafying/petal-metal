from typing import Dict, Optional

import torch
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

class PingProtocol:
    """Protocol for checking server health."""

    def __init__(self, *, update_period: float = 30.0):
        self.update_period = update_period
        self.last_ping: Dict[str, float] = {}

    def update_ping(self, peer_id: str):
        """Update the last ping time for a peer."""
        self.last_ping[peer_id] = torch.cuda.current_time()

    def get_last_ping(self, peer_id: str) -> Optional[float]:
        """Get the last ping time for a peer."""
        return self.last_ping.get(peer_id)

    def is_peer_alive(self, peer_id: str) -> bool:
        """Check if a peer is alive based on its last ping time."""
        last_ping = self.get_last_ping(peer_id)
        if last_ping is None:
            return False
        return torch.cuda.current_time() - last_ping < self.update_period * 2

    def shutdown(self):
        """Shutdown the ping protocol."""
        self.last_ping.clear() 