from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResponse:
    answer: str
    backend: str
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
