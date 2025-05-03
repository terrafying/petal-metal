#!/usr/bin/env python3

import os
import sys
import time
import signal
import logging
import threading
import subprocess
from process_lock import PetalsProcessLock
from simple_discovery import SimplePeerDiscovery

# Configure logging with git hash and line numbers
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-7b1-petals"
SERVER_PORT = 31330

def get_git_hash():
    """Get the current git hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        return 'unknown'

def run_server(existing_peer=None):
    """Start the Petals server process."""
    cmd = [
        "python", "-m", "petals.cli.run_server",
        "--converted_model_name_or_path", MODEL_NAME,
        "--device", "mps",
        "--host_maddrs", f"/ip4/0.0.0.0/tcp/{SERVER_PORT}",
        "--torch_dtype", "float16"
    ]
    
    if existing_peer:
        logger.info(f"Joining existing swarm with peer: {existing_peer}")
        cmd.extend(["--initial_peers", existing_peer])
    else:
        logger.info("Creating new swarm")
        cmd.append("--new_swarm")
        
    logger.info(f"Starting server with command: {' '.join(cmd)} (git: {get_git_hash()})")
    return subprocess.Popen(cmd)

def detect_existing_swarm(discovery, model_name, timeout=10):
    """Check if there's an existing swarm on the local network."""
    logger.info('Checking for existing swarm...')
    discovery.start_discovery()
    
    # Wait briefly for peer discovery
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            peers = discovery.get_peers(model_name)
            if peers:
                logger.info(f'Found existing swarm peers: {peers}')
                # Ensure the peer address has the correct format
                peer_addr = peers[0]
                if not peer_addr.startswith('/ip4/'):
                    logger.warning(f"Invalid peer address format: {peer_addr}")
                    continue
                return peer_addr
        except Exception as e:
            logger.warning(f"Error during peer discovery: {e}")
        time.sleep(1)
    
    logger.info('No existing swarm found')
    return None

def main():
    # Try to acquire the process lock with model information for memory checks
    with PetalsProcessLock(model_name=MODEL_NAME).acquire_context(timeout=30) as lock:
        logger.info('Successfully acquired process lock')
        
        # Start service discovery
        discovery = SimplePeerDiscovery()
        
        # Check for existing swarm
        existing_peer = detect_existing_swarm(discovery, MODEL_NAME)
        
        # Start the server
        server_process = run_server(existing_peer)
        
        # Start advertising after a brief delay to ensure server is up
        def delayed_advertise():
            time.sleep(5)  # Wait for server to start
            try:
                discovery.start_advertising(
                    port=SERVER_PORT,
                    model_name=MODEL_NAME,
                    device='mps'
                )
                logger.info("Started advertising peer")
            except Exception as e:
                logger.error(f"Failed to start advertising: {e}")
        
        advertise_thread = threading.Thread(target=delayed_advertise)
        advertise_thread.start()
        
        # Handle shutdown gracefully
        def cleanup(signum, frame):
            print('\nShutting down...')
            discovery.stop()
            server_process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)
        
        # Wait for server process to finish
        server_process.wait()

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f'Error: {e}')
        sys.exit(1) 