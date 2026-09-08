"""AgroVision core exceptions with actionable messages."""
from __future__ import annotations


class AgroVisionError(Exception):
    """Base error with WHAT/WHY/HOW structure."""

    def __init__(self, what: str, why: str = "", how: str = ""):
        msg = f"WHAT: {what}"
        if why:
            msg += f"\nWHY: {why}"
        if how:
            msg += f"\nHOW: {how}"
        super().__init__(msg)


class ConfigError(AgroVisionError):
    pass


class DataError(AgroVisionError):
    pass


class CheckpointError(AgroVisionError):
    pass


class ModelNotFoundError(AgroVisionError):
    pass


class UnsupportedArchitectureError(AgroVisionError):
    pass


class XAIError(AgroVisionError):
    pass


class ImageError(AgroVisionError):
    pass
