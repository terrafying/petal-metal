from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ServerConfig:
    """Configuration for a model server."""

    # Model configuration
    model_name_or_path: str
    block_indices: List[int]
    torch_dtype: str = "float32"
    quant_type: str = "none"

    # Server configuration
    host: str = "localhost"
    port: int = 8000
    num_handlers: int = 8
    inference_max_length: int = 2048

    # Timeouts
    request_timeout: float = 30.0
    session_timeout: float = 300.0
    step_timeout: float = 60.0
    health_check_interval: float = 10.0

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.block_indices:
            raise ValueError("block_indices must not be empty")

        if self.num_handlers <= 0:
            raise ValueError("num_handlers must be positive")

        if self.inference_max_length <= 0:
            raise ValueError("inference_max_length must be positive")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        if self.session_timeout <= 0:
            raise ValueError("session_timeout must be positive")

        if self.step_timeout <= 0:
            raise ValueError("step_timeout must be positive")

        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive") 