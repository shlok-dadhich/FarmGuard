"""Central logging."""
from __future__ import annotations

import logging
import sys

_configured = False


def get_logger(name: str = "agrovision", level: int = logging.INFO) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _configured = True
    return logger
