from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch

UID_DELIMITER = "."
CHAIN_DELIMITER = "."

class ServerState(str, Enum):
    """State of a server in the network."""
    OFFLINE = "offline"
    ONLINE = "online"
    JOINING = "joining"

@dataclass
class ModelInfo:
    """Information about a model."""
    repository: str
    num_blocks: int
    torch_dtype: torch.dtype

@dataclass
class ServerInfo:
    """Information about a server."""
    host: str
    port: int
    block_indices: List[int]
    throughput: float = 0.0
    latency: float = 0.0
    last_seen: float = 0.0

@dataclass
class BlockInfo:
    """Information about a model block."""
    block_idx: int
    server_id: str
    throughput: float = 0.0
    latency: float = 0.0
    last_seen: float = 0.0

@dataclass
class InferenceSession:
    """Information about an inference session."""
    session_id: str
    block_indices: List[int]
    max_length: int
    timeout: float
    last_seen: float = 0.0

@dataclass
class InferenceMetadata:
    """Metadata for inference requests."""
    session_id: str
    block_idx: int
    max_length: int
    timeout: float
    last_seen: float = 0.0

@dataclass
class RemoteSpanInfo:
    """Information about a span of blocks on a remote server."""
    peer_id: str
    start: int
    end: int
    throughput: float = 0.0
    latency: float = 0.0
    last_seen: float = 0.0

    @property
    def length(self) -> int:
        return self.end - self.start

@dataclass
class RemoteModuleInfo:
    """Information about a remote module."""
    peer_id: str
    block_indices: List[int]
    throughput: float = 0.0
    latency: float = 0.0
    last_seen: float = 0.0

@dataclass
class RPCInfo:
    """Information about RPC calls."""
    inference_schema: Tuple[Tuple[Any, ...], Dict[str, Any]]
    forward_schema: Tuple[Tuple[Any, ...], Dict[str, Any]]
    backward_schema: Tuple[Tuple[Any, ...], Dict[str, Any]]

ModuleUID = str  # Type alias for module unique identifier
Handle = int    # Type alias for tensor handle in memory cache

def parse_uid(uid: str) -> Tuple[str, int]:
    """Parse a UID into its components."""
    model_name, index = uid.split(UID_DELIMITER)
    return model_name, int(index)
