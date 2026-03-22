"""Shared response models for backend service implementations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResponse:
    """Structured response returned by the backend service layer."""

    answer: str
    backend: str
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
