import os
import fcntl
import psutil
import time
import logging
import tempfile
import subprocess
from typing import Optional, Tuple, Dict
from contextlib import contextmanager
import threading
import atexit
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUMemoryMonitor:
    """Monitor GPU memory usage on Apple Silicon."""
    
    @staticmethod
    def get_gpu_memory() -> Dict[str, int]:
        """
        Get GPU memory usage using powermetrics.
        
        Returns:
            Dict with 'total', 'used', and 'free' memory in bytes
        """
        try:
            # Use powermetrics to get GPU memory info
            cmd = ['sudo', 'powermetrics', '--samplers', 'gpu_power', '-n', '1']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse the output
            gpu_mem = {}
            for line in result.stdout.split('\n'):
                if 'GPU memory' in line:
                    # Extract memory values (typically in GB)
                    parts = line.split()
                    used_gb = float(parts[2].strip('GB'))
                    total_gb = float(parts[5].strip('GB'))
                    
                    # Convert to bytes
                    gpu_mem['total'] = int(total_gb * 1024 * 1024 * 1024)
                    gpu_mem['used'] = int(used_gb * 1024 * 1024 * 1024)
                    gpu_mem['free'] = gpu_mem['total'] - gpu_mem['used']
                    break
            
            if not gpu_mem:
                raise ValueError("Could not find GPU memory information")
                
            return gpu_mem
            
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            # Return conservative estimates for M1/M2 chips
            return {
                'total': 16 * 1024 * 1024 * 1024,  # 16GB
                'used': 8 * 1024 * 1024 * 1024,    # Assume 50% used
                'free': 8 * 1024 * 1024 * 1024     # Assume 50% free
            }
    
    @staticmethod
    def estimate_model_memory_requirement(model_name: str) -> int:
        """
        Estimate memory requirement for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Estimated memory requirement in bytes
        """
        # Model size estimates in bytes (for float16)
        MODEL_SIZES = {
            'bigscience/bloom-7b1-petals': 7 * 1024 * 1024 * 1024,  # 7GB
            'meta-llama/Llama-2-7b-chat': 7 * 1024 * 1024 * 1024,   # 7GB
            'meta-llama/Llama-2-13b-chat': 13 * 1024 * 1024 * 1024, # 13GB
            'bigscience/bloom-petals': 176 * 1024 * 1024 * 1024,    # 176GB
        }
        
        # Get base model name without additional tags
        base_model = model_name.split('-petals')[0]
        
        # Return estimate or conservative default
        return MODEL_SIZES.get(base_model, 8 * 1024 * 1024 * 1024)  # Default 8GB

class PetalsProcessLock:
    """Process lock for Petals server to prevent multiple instances."""
    
    def __init__(self, model_name=None):
        self.lock_file = Path.home() / ".petals" / "server.lock"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.lock_handle = None
        self._register_cleanup()
        
    def _register_cleanup(self):
        """Register cleanup handlers."""
        atexit.register(self.release)
        signal.signal(signal.SIGTERM, lambda s, f: self.release())
        signal.signal(signal.SIGINT, lambda s, f: self.release())
        
    def _check_memory(self):
        """Check if system has enough memory."""
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_percent = mem.percent
            
            logger.info(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available ({used_percent:.1f}% used)")
            
            # For MPS devices, we can be more lenient with memory
            is_mps = False
            try:
                import torch
                is_mps = torch.backends.mps.is_available()
            except:
                pass
                
            if is_mps:
                # For MPS, we can use up to 90% of memory
                min_available = 4.0  # GB
                if available_gb < min_available:
                    logger.warning(f"Low memory available ({available_gb:.1f}GB) for MPS device")
                    return False
            else:
                # For other devices, require more free memory
                min_available = 8.0  # GB
                if available_gb < min_available:
                    logger.warning(f"Insufficient memory available ({available_gb:.1f}GB)")
                    return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")
            return True  # Be lenient if we can't check memory
            
    def acquire(self, timeout=30):
        """Acquire the process lock."""
        start_time = time.time()
        
        # First try to clean up any stale lock
        if self.lock_file.exists():
            try:
                # Check if the process that created the lock is still running
                with open(self.lock_file, 'r') as f:
                    pid = int(f.read().strip())
                if not psutil.pid_exists(pid):
                    logger.info(f"Found stale lock from process {pid}, removing")
                    self.lock_file.unlink()
            except:
                # If we can't read the lock file, it might be corrupted
                logger.warning("Found corrupted lock file, removing")
                self.lock_file.unlink()
        
        while time.time() - start_time < timeout:
            try:
                # Try to create the lock file
                self.lock_handle = open(self.lock_file, 'x')
                self.lock_handle.write(str(os.getpid()))
                self.lock_handle.flush()
                
                # Check memory after acquiring lock
                if not self._check_memory():
                    self.release()
                    raise RuntimeError("Insufficient system memory")
                    
                return True
                
            except FileExistsError:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Error acquiring lock: {e}", exc_info=e)
                self.release()
                raise RuntimeError(f"Failed to acquire Petals process lock: {e}") from e
                
        raise RuntimeError("Failed to acquire Petals process lock: timeout")
        
    def release(self):
        """Release the process lock."""
        try:
            if self.lock_handle:
                self.lock_handle.close()
                self.lock_handle = None
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        
    def acquire_context(self, timeout=30):
        """Get a context manager for the lock."""
        return self 