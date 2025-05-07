import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

class TraceableLogger:
    """A logger that adds traceability to every log statement."""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] [%(processName)s] %(message)s\n'
            'File: %(pathname)s\n'
            'Function: %(funcName)s\n'
            'Stack Trace:\n%(stack_trace)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(file_handler)
    
    def _get_stack_trace(self) -> str:
        """Get the current stack trace, excluding logging-related frames."""
        stack = traceback.extract_stack()
        # Remove the last 3 frames which are from the logging module
        relevant_stack = stack[:-3]
        return ''.join(traceback.format_list(relevant_stack))
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method that adds stack trace to the log record."""
        extra = kwargs.get('extra', {})
        extra['stack_trace'] = self._get_stack_trace()
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log an exception with full traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)

def get_logger(name: str, log_dir: Optional[str] = None) -> TraceableLogger:
    """
    Get a traceable logger instance.
    
    Args:
        name: The name of the logger
        log_dir: Optional directory to store log files
    
    Returns:
        A TraceableLogger instance
    """
    return TraceableLogger(name, log_dir)

# Initialize default logging configuration
def initialize_logging(log_dir: Optional[str] = None):
    """
    Initialize the default logging configuration.
    
    Args:
        log_dir: Optional directory to store log files
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.expanduser("~"), ".petal-2-metal", "logs")
    
    # Create root logger
    root_logger = get_logger("root", log_dir)
    
    # Set up logging for common modules
    get_logger("pattern_manager", log_dir)
    get_logger("swarm_state", log_dir)
    get_logger("shape_visualizer", log_dir)
    get_logger("server", log_dir)
    get_logger("client", log_dir)
    
    return root_logger 