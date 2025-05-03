from service_discovery import PetalsServiceDiscovery
from process_lock import PetalsProcessLock
import subprocess
import threading
import signal
import sys
import logging
import time
import torch
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-7b1-petals"

def get_device_config():
    """Get appropriate device configuration based on system."""
    if platform.system() == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            return "mps", "float32"  # Use float32 for better compatibility on MPS
        else:
            return "cpu", "float32"
    elif torch.cuda.is_available():
        return "cuda", "float16"
    else:
        return "cpu", "float32"

def run_server(existing_peer=None):
    """Start the server either as new swarm or joining existing one."""
    device, dtype = get_device_config()
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    cmd = [
        'python3', '-m', 'petals.cli.run_server',
        '--port', '31330',
        '--converted_model_name_or_path', MODEL_NAME,
        '--device', device,
        '--torch_dtype', dtype,
        '--max_batch_size', '2',  # Reduce batch size for better stability
        '--max_chunk_size_bytes', '1024'  # Reduce chunk size for better stability
    ]
    
    if existing_peer:
        # Join existing swarm
        cmd.extend(['--initial_peers', existing_peer])
        logger.info(f'Starting server to join existing peer: {existing_peer}')
    else:
        # Start new swarm
        cmd.append('--new_swarm')
        logger.info('Starting server as new swarm')
    
    try:
        process = subprocess.Popen(cmd)
        # Wait a bit to check if process starts successfully
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(f"Server process failed to start. Exit code: {process.returncode}")
        return process
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

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
                return peers[0]  # Return the first peer's address
        except Exception as e:
            logger.warning(f"Error during peer discovery: {e}")
        time.sleep(1)
    
    logger.info('No existing swarm found')
    return None

def run_server_secondary_local():
    process = subprocess.Popen([
        'python3', '-m', 'petals.cli.run_server',
        '--initial_peers', 'localhost:31330',
        '--port', '31331',
        '--converted_model_name_or_path', 'bigscience/bloom-7b1-petals',
        '--device', 'mps',
        '--torch_dtype', 'float16'
    ])
    return process

def main():
    try:
        # Try to acquire the process lock
        with PetalsProcessLock().acquire_context(timeout=30) as lock:
            logger.info('Successfully acquired process lock')
            
            # Start service discovery
            discovery = PetalsServiceDiscovery()
            
            # Check for existing swarm
            existing_peer = detect_existing_swarm(discovery, MODEL_NAME)
            
            # Start the server
            server_process = run_server(existing_peer)
            
            # Start advertising after a brief delay to ensure server is up
            def delayed_advertise():
                time.sleep(5)  # Wait for server to start
                try:
                    discovery.start_advertising(
                        port=31330,
                        model_name=MODEL_NAME,
                        device=get_device_config()[0]
                    )
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
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f'Fatal error: {e}')
        sys.exit(1)
