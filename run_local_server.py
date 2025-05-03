from service_discovery import PetalsServiceDiscovery
from process_lock import PetalsProcessLock
import subprocess
import threading
import signal
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_server():
    process = subprocess.Popen([
        'python3', '-m', 'petals.cli.run_server',
        '--new_swarm',
        '--port', '31330',
        '--converted_model_name_or_path', 'bigscience/bloom-7b1-petals',
        '--device', 'mps',
        '--torch_dtype', 'float16'
    ])
    return process

def main():
    # Try to acquire the process lock
    with PetalsProcessLock().acquire_context(timeout=30) as lock:
        logger.info('Successfully acquired process lock')
        
        # Start service discovery
        discovery = PetalsServiceDiscovery()
        discovery.start_discovery()
        
        # Start the server
        server_process = run_server()
        
        # Start advertising after a brief delay to ensure server is up
        def delayed_advertise():
            import time
            time.sleep(5)  # Wait for server to start
            discovery.start_advertising(
                port=31330,
                model_name='bigscience/bloom-7b1-petals',
                device='mps'
            )
        
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
