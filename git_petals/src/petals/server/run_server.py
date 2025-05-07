import argparse
import os
from typing import Optional

import torch
from hivemind.utils.logging import get_logger

from petals.server.server import Server
from petals.server.reachability import ReachabilityProtocol
from petals.server.ping import PingProtocol

logger = get_logger(__name__)

def main(
    *,
    model_name_or_path: str,
    num_blocks: int,
    block_indices: Optional[str] = None,
    host_maddrs: Optional[str] = None,
    public_ip: Optional[str] = None,
    public_port: Optional[int] = None,
    update_period: float = 30.0,
    max_retries: int = 3,
    torch_dtype: str = "float32",
    quant_type: str = "none",
    adapters: Optional[str] = None,
    **kwargs,
):
    """Run a server that uses local file-based discovery."""
    if block_indices is not None:
        block_indices = [int(idx) for idx in block_indices.split(",")]
    else:
        block_indices = list(range(num_blocks))

    if adapters is not None:
        adapters = tuple(adapter for adapter in adapters.split(",") if adapter)
    else:
        adapters = ()

    # Initialize protocols
    reachability_protocol = ReachabilityProtocol()
    ping_protocol = PingProtocol(update_period=update_period)

    # Initialize server
    server = Server(
        model_name_or_path=model_name_or_path,
        block_indices=block_indices,
        host_maddrs=host_maddrs,
        public_ip=public_ip,
        public_port=public_port,
        reachability_protocol=reachability_protocol,
        ping_protocol=ping_protocol,
        max_retries=max_retries,
        torch_dtype=getattr(torch, torch_dtype),
        quant_type=quant_type,
        adapters=adapters,
        **kwargs,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_blocks", type=int, required=True)
    parser.add_argument("--block_indices", type=str)
    parser.add_argument("--host_maddrs", type=str)
    parser.add_argument("--public_ip", type=str)
    parser.add_argument("--public_port", type=int)
    parser.add_argument("--update_period", type=float, default=30.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--torch_dtype", type=str, default="float32")
    parser.add_argument("--quant_type", type=str, default="none")
    parser.add_argument("--adapters", type=str)
    args = parser.parse_args()

    main(**vars(args)) 