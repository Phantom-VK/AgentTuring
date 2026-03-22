"""Environment-driven backend configuration."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()

def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class Settings:
    """Runtime settings loaded from environment variables."""

    cors_origins: tuple[str, ...]
    provider_api_key: str | None
    provider_base_url: str | None
    default_model: str
    triage_model: str
    solver_model: str
    research_model: str
    tavily_api_key: str | None
    agent_temperature: float
    tracing_disabled: bool

    @property
    def agentic_enabled(self) -> bool:
        """Return whether provider credentials are present for agent execution."""
        return bool(self.provider_api_key and self.provider_base_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache backend settings from the current environment."""
    cors_origins = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    )

    provider_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    provider_base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    return Settings(
        cors_origins=cors_origins or ("http://localhost:5173",),
        provider_api_key=provider_api_key,
        provider_base_url=provider_base_url,
        default_model=os.getenv("OPENAI_DEFAULT_MODEL", "deepseek-chat"),
        triage_model=os.getenv("AGENT_TRIAGE_MODEL", "deepseek-chat"),
        solver_model=os.getenv("AGENT_SOLVER_MODEL", "deepseek-reasoner"),
        research_model=os.getenv("AGENT_RESEARCH_MODEL", "deepseek-chat"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        agent_temperature=_get_float("AGENT_TEMPERATURE", 0.2),
        tracing_disabled=_get_bool("AGENT_TRACING_DISABLED", True),
    )
