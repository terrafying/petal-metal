import argparse
import asyncio
import torch
from transformers import AutoConfig
from typing import List

from server import ServerConfig, Server
from logging_utils import initialize_logging, get_logger

# Initialize logging
logger = initialize_logging()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_blocks", type=int, help="Number of blocks to serve")
    parser.add_argument("--block_indices", type=str, help="Comma-separated list of block indices to serve")
    parser.add_argument("--num_handlers", type=int, default=8, help="Number of connection handlers")
    parser.add_argument("--inference_max_length", type=int, default=2048, help="Maximum sequence length for inference")
    parser.add_argument("--request_timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--session_timeout", type=float, default=300.0, help="Session timeout in seconds")
    parser.add_argument("--step_timeout", type=float, default=60.0, help="Step timeout in seconds")
    parser.add_argument("--quant_type", type=str, default="none", help="Quantization type")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch data type")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--log_dir", type=str, help="Directory to store log files")
    args = parser.parse_args()

    # Initialize logging with custom directory if provided
    if args.log_dir:
        logger = initialize_logging(args.log_dir)
    
    logger.info("Starting server initialization...")
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model)
    num_blocks = args.num_blocks or config.num_hidden_layers

    # Parse block indices
    if args.block_indices:
        block_indices = [int(i) for i in args.block_indices.split(",")]
    else:
        block_indices = list(range(num_blocks))

    logger.info(f"Model: {args.model}")
    logger.info(f"Number of blocks: {len(block_indices)}")
    logger.info(f"Block indices: {block_indices}")

    # Create server config
    server_config = ServerConfig(
        model_name_or_path=args.model,
        block_indices=block_indices,
        num_handlers=args.num_handlers,
        inference_max_length=args.inference_max_length,
        request_timeout=args.request_timeout,
        session_timeout=args.session_timeout,
        step_timeout=args.step_timeout,
        quant_type=args.quant_type,
        torch_dtype=args.torch_dtype,
    )

    # Create and start server
    server = Server(server_config)
    try:
        logger.info(f"Starting server on {args.host}:{args.port}")
        asyncio.run(server.start(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, shutting down server...")
        asyncio.run(server.shutdown())
    except Exception as e:
        logger.exception(f"Server error: {e}")
        asyncio.run(server.shutdown())

if __name__ == "__main__":
    main()
