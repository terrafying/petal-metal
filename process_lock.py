import os
import fcntl
import psutil
import time
import logging
import tempfile
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PetalsProcessLock:
    """
    Process locking mechanism for Petals to prevent multiple instances from overloading the machine.
    Uses file-based locking which is reliable on macOS/Unix systems.
    """
    
    def __init__(self, lock_file: Optional[str] = None):
        """
        Initialize the process lock.
        
        Args:
            lock_file: Optional path to lock file. If None, uses a default path in /tmp
        """
        if lock_file is None:
            lock_file = os.path.join(tempfile.gettempdir(), "petals_process.lock")
        self.lock_file = lock_file
        self.lock_fd = None
        
    def acquire(self, timeout: int = 10, check_resource_limits: bool = True) -> bool:
        """
        Acquire the process lock.
        
        Args:
            timeout: Maximum time to wait for lock in seconds
            check_resource_limits: Whether to check system resources before acquiring
            
        Returns:
            bool: True if lock was acquired successfully
        """
        if check_resource_limits and not self._check_resources():
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