import os
import sys
import time
import signal
import logging
import threading
from typing import Optional
from process_lock import PetalsProcessLock
from unified_discovery import UnifiedDiscovery
from petals.server.server import Server
import torch
import base58
import hashlib
import uuid

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-7b1-petals"
SERVER_PORT = 31330

def generate_peer_id() -> str:
    """Generate a peer ID in the format expected by multiaddr."""
    # Generate a random UUID and hash it
    random_bytes = uuid.uuid4().bytes
    hash_bytes = hashlib.sha256(random_bytes).digest()
    # Format: <hash-func-code><digest-length><digest-value>
    mh = bytes([0x12]) + bytes([len(hash_bytes)]) + hash_bytes
    # Encode with base58btc
    return base58.b58encode(mh).decode()

def get_device_config():
    """Get device configuration based on available hardware."""
    if os.path.exists('/dev/mps'):
        return {
            "device": "mps",
            "torch_dtype": "float32",
            "dht_prefix": "petals",
            "throughput": 5.0
        }
    else:
        return {
            "device": "cpu",
            "torch_dtype": "float32",
            "dht_prefix": "petals",
            "throughput": 1.0
        }

def run_server(existing_peer: Optional[str] = None):
    """Run a Petals server instance."""
    try:
        # Generate peer ID
        peer_id = generate_peer_id()
        logger.info(f"Generated peer ID: {peer_id}")
        
        # Initialize discovery
        discovery = UnifiedDiscovery()
        discovery.start_discovery()
        logger.info("Started peer discovery service")
        
        # Get device config
        device_config = get_device_config()
        logger.info(f"Using device config: {device_config}")
        
        # Initialize server with minimal required configuration
        # Set default tensor type before server initialization
        torch.set_default_dtype(torch.float32 if device_config["torch_dtype"] == "float32" else torch.float16)
        
        # Create server configuration
        server_config = {
            "converted_model_name_or_path": MODEL_NAME,
            "device": device_config["device"],
            "torch_dtype": device_config["torch_dtype"],
            "initial_peers": [existing_peer] if existing_peer else None,
            "dht_prefix": device_config["dht_prefix"],
            "throughput": device_config["throughput"]
        }
        
        # Initialize server
        server = Server(**server_config)
        
        # Start advertising
        discovery.start_advertising(
            port=SERVER_PORT,
            model_name=MODEL_NAME,
            device=device_config["device"]
        )
        logger.info("Started advertising server")
        
        # Handle cleanup
        def cleanup(signum=None, frame=None):
            logger.info("Shutting down server...")
            try:
                discovery.stop()
                logger.info("Stopped discovery service")
            except Exception as e:
                logger.error(f"Error stopping discovery: {e}")
            
            try:
                server.stop()
                logger.info("Stopped server")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
        
        # Main loop
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in server: {e}", exc_info=e, stack_info=True)
        if 'cleanup' in locals():
            cleanup()
        raise

if __name__ == '__main__':
    try:
        run_server()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
