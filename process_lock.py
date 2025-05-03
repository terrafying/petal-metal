import os
import fcntl
import psutil
import time
import logging
import tempfile
import subprocess
from typing import Optional, Tuple, Dict
from contextlib import contextmanager

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
    """
    Process locking mechanism for Petals to prevent multiple instances from overloading the machine.
    Uses file-based locking which is reliable on macOS/Unix systems.
    """
    
    def __init__(self, lock_file: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the process lock.
        
        Args:
            lock_file: Optional path to lock file. If None, uses a default path in /tmp
            model_name: Name of the model for memory requirement estimation
        """
        if lock_file is None:
            lock_file = os.path.join(tempfile.gettempdir(), "petals_process.lock")
        self.lock_file = lock_file
        self.lock_fd = None
        self.model_name = model_name
        self.gpu_monitor = GPUMemoryMonitor()
        
    def _check_gpu_memory(self) -> Tuple[bool, str]:
        """
        Check if there's enough GPU memory available.
        
        Returns:
            Tuple of (bool, str) indicating if memory is sufficient and why/why not
        """
        if not self.model_name:
            return True, "No model specified for memory check"
            
        try:
            gpu_mem = self.gpu_monitor.get_gpu_memory()
            required = self.gpu_monitor.estimate_model_memory_requirement(self.model_name)
            
            # Calculate memory requirement with buffer
            required_with_buffer = int(required * 1.2)  # Add 20% buffer
            
            if gpu_mem['free'] < required_with_buffer:
                return False, (
                    f"Insufficient GPU memory. Required: {required_with_buffer / 1024**3:.1f}GB "
                    f"(including buffer), Available: {gpu_mem['free'] / 1024**3:.1f}GB"
                )
            
            # Check if we're trying to use too much of total memory
            max_safe_usage = int(gpu_mem['total'] * 0.9)  # Don't use more than 90%
            if (gpu_mem['used'] + required) > max_safe_usage:
                return False, (
                    f"Would exceed safe GPU memory usage. "
                    f"Total: {gpu_mem['total'] / 1024**3:.1f}GB, "
                    f"Currently used: {gpu_mem['used'] / 1024**3:.1f}GB, "
                    f"Required: {required / 1024**3:.1f}GB"
                )
            
            return True, "Sufficient GPU memory available"
            
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return True, "Could not check GPU memory, proceeding with caution"
                
    def acquire(self, timeout: int = 10, check_resource_limits: bool = True) -> bool:
        """
        Acquire the process lock.
        
        Args:
            timeout: Maximum time to wait for lock in seconds
            check_resource_limits: Whether to check system resources before acquiring
            
        Returns:
            bool: True if lock was acquired successfully
        """
        if check_resource_limits:
            # Check GPU memory first
            gpu_mem_ok, gpu_mem_msg = self._check_gpu_memory()
            if not gpu_mem_ok:
                logger.warning(f"GPU memory check failed: {gpu_mem_msg}")
                return False
            
            # Check other system resources
            if not self._check_resources():
                return False
            
        start_time = time.time()
        while True:
            try:
                # Create or open the lock file
                self.lock_fd = open(self.lock_file, 'w')
                
                # Try to acquire an exclusive lock
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Write current process ID to lock file
                self.lock_fd.write(str(os.getpid()))
                self.lock_fd.flush()
                
                logger.info("Successfully acquired Petals process lock")
                return True
                
            except (IOError, OSError) as e:
                if time.time() - start_time > timeout:
                    logger.warning(f"Failed to acquire lock after {timeout} seconds")
                    return False
                    
                # Check if the process holding the lock is still alive
                try:
                    with open(self.lock_file, 'r') as f:
                        pid = int(f.read().strip())
                        if not psutil.pid_exists(pid):
                            logger.info(f"Found stale lock from dead process {pid}, removing")
                            os.remove(self.lock_file)
                            continue
                except (IOError, ValueError):
                    pass
                
                time.sleep(1)
                continue
                
    def release(self):
        """Release the process lock."""
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                os.remove(self.lock_file)
                logger.info("Released Petals process lock")
            except (IOError, OSError) as e:
                logger.error(f"Error releasing lock: {e}")
            finally:
                self.lock_fd = None
                
    def _check_resources(self) -> bool:
        """
        Check if system has enough resources to run Petals.
        
        Returns:
            bool: True if system has sufficient resources
        """
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"CPU usage too high: {cpu_percent}%")
                return False
                
            # Check memory usage
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                logger.warning(f"Memory usage too high: {mem.percent}%")
                return False
                
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                logger.warning(f"Disk usage too high: {disk.percent}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False
            
    @contextmanager
    def acquire_context(self, timeout: int = 10, check_resource_limits: bool = True):
        """
        Context manager for acquiring and releasing the lock.
        
        Args:
            timeout: Maximum time to wait for lock
            check_resource_limits: Whether to check system resources
        """
        try:
            if not self.acquire(timeout, check_resource_limits):
                raise RuntimeError("Failed to acquire Petals process lock")
            yield
        finally:
            self.release()
            
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Failed to acquire Petals process lock")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release() 