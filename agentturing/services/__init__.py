"""Backend service entrypoints for the FastAPI application."""

from functools import lru_cache

from agentturing.config import get_settings

from .agentic_backend import AgenticBackendUnavailable, AgenticMathBackend
from .base import BackendResponse


@lru_cache(maxsize=1)
def get_chat_backend():
    """Create and cache the active chat backend implementation."""
    return AgenticMathBackend(settings=get_settings())


__all__ = [
    "AgenticBackendUnavailable",
    "BackendResponse",
    "get_chat_backend",
]
