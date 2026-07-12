"""Environment parsing for the standalone voice-agent worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, cast

from pydantic import AnyUrl, Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
PositiveInteger = Annotated[int, Field(gt=0)]
WebSearchResultLimit = Annotated[int, Field(ge=1, le=10)]
type AgentName = Literal["realtime-chat-voice"]


class _EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=None,
        extra="ignore",
    )

    livekit_url: AnyUrl = Field(validation_alias="LIVEKIT_URL")
    livekit_api_key: NonBlankString = Field(validation_alias="LIVEKIT_API_KEY")
    livekit_api_secret: NonBlankString = Field(validation_alias="LIVEKIT_API_SECRET")
    livekit_agent_name: AgentName = Field(
        default="realtime-chat-voice",
        validation_alias="LIVEKIT_AGENT_NAME",
    )
    api_url: AnyUrl = Field(validation_alias="API_URL")
    voice_agent_bridge_token: NonBlankString = Field(validation_alias="VOICE_AGENT_BRIDGE_TOKEN")
    tavily_api_key: NonBlankString = Field(validation_alias="TAVILY_API_KEY")
    voice_agent_http_timeout_ms: PositiveInteger = Field(
        default=5_000,
        validation_alias="VOICE_AGENT_HTTP_TIMEOUT_MS",
    )
    voice_agent_http_max_attempts: PositiveInteger = Field(
        default=4,
        validation_alias="VOICE_AGENT_HTTP_MAX_ATTEMPTS",
    )
    voice_agent_http_retry_base_ms: PositiveInteger = Field(
        default=250,
        validation_alias="VOICE_AGENT_HTTP_RETRY_BASE_MS",
    )
    voice_web_search_max_results: WebSearchResultLimit = Field(
        default=5,
        validation_alias="VOICE_WEB_SEARCH_MAX_RESULTS",
    )
    voice_web_search_timeout_ms: PositiveInteger = Field(
        default=6_000,
        validation_alias="VOICE_WEB_SEARCH_TIMEOUT_MS",
    )


@dataclass(frozen=True, slots=True)
class VoiceAgentConfig:
    agent_name: AgentName
    bridge_token: str
    api_url: str
    http_max_attempts: int
    http_retry_base_ms: int
    http_timeout_ms: int
    tavily_api_key: str
    web_search_max_results: int
    web_search_timeout_ms: int


def normalize_log_level(value: str | None) -> str:
    """Map the Node worker's ``fatal`` level to Python's ``critical`` level."""

    normalized = (value or "critical").strip().lower()
    return "critical" if normalized == "fatal" else normalized


def load_config(environment: Mapping[str, str] | None = None) -> VoiceAgentConfig:
    settings_from_environment = cast(
        Callable[[], _EnvironmentSettings],
        _EnvironmentSettings,
    )
    parsed = (
        settings_from_environment()
        if environment is None
        else _EnvironmentSettings.model_validate(dict(environment))
    )

    return VoiceAgentConfig(
        agent_name=parsed.livekit_agent_name,
        bridge_token=parsed.voice_agent_bridge_token,
        api_url=str(parsed.api_url).rstrip("/"),
        http_max_attempts=parsed.voice_agent_http_max_attempts,
        http_retry_base_ms=parsed.voice_agent_http_retry_base_ms,
        http_timeout_ms=parsed.voice_agent_http_timeout_ms,
        tavily_api_key=parsed.tavily_api_key,
        web_search_max_results=parsed.voice_web_search_max_results,
        web_search_timeout_ms=parsed.voice_web_search_timeout_ms,
    )


__all__ = ["VoiceAgentConfig", "load_config", "normalize_log_level"]
