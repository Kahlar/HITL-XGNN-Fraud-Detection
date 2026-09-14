"""Structured logging configuration for HITL-XGNN."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "HITL-XGNN", level: Optional[int] = None) -> logging.Logger:
    """Returns a structured logger with standardized console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if level is not None:
        logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.INFO)
        
    return logger
