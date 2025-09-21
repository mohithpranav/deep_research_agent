"""
Logger utility for the Deep Researcher Agent
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup and return a logger with console and file handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    try:
        # Create logs directory
        project_root = Path(__file__).parent.parent.parent
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create file handler
        log_file = logs_dir / f"deep_researcher_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        # If file logging fails, just use console
        console_handler.setLevel(logging.WARNING)
        logger.warning(f"Could not setup file logging: {e}")
    
    return logger

# Create default logger for backward compatibility
logger = setup_logger("DeepResearcher")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance"""
    if name is None:
        return logger
    return setup_logger(name)