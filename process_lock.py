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
    """Process lock to prevent multiple Petals instances from running simultaneously."""
    
    def __init__(self, model_name=None, timeout=30):
        self.lock_file = "/tmp/petals.lock"
        self.timeout = timeout
        self.model_name = model_name
        self.lock = threading.Lock()
        
    def _check_memory(self):
        """Check if system has enough memory for the model."""
        try:
            # Get system memory info
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_percent = mem.percent
            
            # For MPS devices, we'll be more lenient with memory requirements
            is_mps = False
            try:
                is_mps = "mps" in str(subprocess.check_output(["python", "-c", "import torch; print(torch.device('mps'))"]))
            except:
                pass
                
            if is_mps:
                # For MPS devices, we'll use a percentage-based approach
                # Allow up to 90% memory usage since MPS can handle memory more efficiently
                if used_percent > 90:
                    logger.warning(f"Memory usage too high: {used_percent}%")
                    return False
                    
                # Also ensure we have at least 4GB available
                if available_gb < 4.0:
                    logger.warning(f"Insufficient available memory: {available_gb:.1f}GB")
                    return False
            else:
                # Standard GPU requirements
                required_gb = 9.6
                if available_gb < required_gb:
                    logger.warning(f"Insufficient memory. Required: {required_gb:.1f}GB, Available: {available_gb:.1f}GB")
                    return False
                
            logger.info(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available ({used_percent}% used)")
            return True
            
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            # Be more lenient if we can't check memory
            return True
            
    def acquire(self, timeout=None, check_resource_limits=True):
        """Try to acquire the process lock."""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if lock file exists
                if os.path.exists(self.lock_file):
                    # Check if process is still running
                    try:
                        with open(self.lock_file, 'r') as f:
                            pid = int(f.read().strip())
                        if psutil.pid_exists(pid):
                            time.sleep(1)
                            continue
                    except:
                        pass
                        
                # Check resource limits if requested
                if check_resource_limits and not self._check_memory():
                    return False
                    
                # Create lock file
                with open(self.lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                return True
                
            except Exception as e:
                logger.error(f"Error acquiring lock: {e}")
                time.sleep(1)
                
        return False
        
    def release(self):
        """Release the process lock."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            
    @contextmanager
    def acquire_context(self, timeout=None, check_resource_limits=True):
        """Context manager for acquiring and releasing the lock."""
        if self.acquire(timeout, check_resource_limits):
            try:
                yield self
            finally:
                self.release()
        else:
            raise RuntimeError("Failed to acquire Petals process lock") 