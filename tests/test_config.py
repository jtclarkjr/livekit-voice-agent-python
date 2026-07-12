from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import (
    VoiceAgentConfig,
    load_config,
    normalize_log_level,
)
from agent.constants import AGENT_NAME

VALID_ENVIRONMENT = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "key",
    "LIVEKIT_API_SECRET": "secret",
    "API_URL": "http://localhost:4000/",
    "VOICE_AGENT_BRIDGE_TOKEN": "bridge-secret",
    "TAVILY_API_KEY": "tavily-secret",
}


def test_load_config_applies_defaults_and_normalizes_api_url() -> None:
    assert load_config(VALID_ENVIRONMENT) == VoiceAgentConfig(
        agent_name=AGENT_NAME,
        bridge_token="bridge-secret",
        api_url="http://localhost:4000",
        http_max_attempts=4,
        http_retry_base_ms=250,
        http_timeout_ms=5_000,
        tavily_api_key="tavily-secret",
        web_search_max_results=5,
        web_search_timeout_ms=6_000,
    )


def test_load_config_parses_numeric_overrides() -> None:
    config = load_config(
        {
            **VALID_ENVIRONMENT,
            "VOICE_AGENT_HTTP_TIMEOUT_MS": "7500",
            "VOICE_AGENT_HTTP_MAX_ATTEMPTS": "6",
            "VOICE_AGENT_HTTP_RETRY_BASE_MS": "100",
            "VOICE_WEB_SEARCH_MAX_RESULTS": "10",
            "VOICE_WEB_SEARCH_TIMEOUT_MS": "8000",
        }
    )

    assert config.http_timeout_ms == 7_500
    assert config.http_max_attempts == 6
    assert config.http_retry_base_ms == 100
    assert config.web_search_max_results == 10
    assert config.web_search_timeout_ms == 8_000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOICE_AGENT_HTTP_TIMEOUT_MS", "0"),
        ("VOICE_AGENT_HTTP_MAX_ATTEMPTS", "-1"),
        ("VOICE_AGENT_HTTP_RETRY_BASE_MS", "1.5"),
        ("VOICE_WEB_SEARCH_MAX_RESULTS", "0"),
        ("VOICE_WEB_SEARCH_MAX_RESULTS", "11"),
        ("VOICE_WEB_SEARCH_TIMEOUT_MS", "0"),
    ],
)
def test_load_config_rejects_invalid_numeric_settings(name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        load_config({**VALID_ENVIRONMENT, name: value})


@pytest.mark.parametrize(
    "name",
    [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "API_URL",
        "VOICE_AGENT_BRIDGE_TOKEN",
        "TAVILY_API_KEY",
    ],
)
def test_load_config_requires_every_runtime_credential(name: str) -> None:
    environment = dict(VALID_ENVIRONMENT)
    del environment[name]

    with pytest.raises(ValidationError):
        load_config(environment)


@pytest.mark.parametrize(
    "name",
    [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "VOICE_AGENT_BRIDGE_TOKEN",
        "TAVILY_API_KEY",
    ],
)
def test_load_config_rejects_blank_secrets(name: str) -> None:
    with pytest.raises(ValidationError):
        load_config({**VALID_ENVIRONMENT, name: "   "})


def test_load_config_rejects_dispatch_name_drift_and_invalid_urls() -> None:
    with pytest.raises(ValidationError):
        load_config({**VALID_ENVIRONMENT, "LIVEKIT_AGENT_NAME": "another-agent"})
    with pytest.raises(ValidationError):
        load_config({**VALID_ENVIRONMENT, "API_URL": "not a URL"})


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [(None, "critical"), ("fatal", "critical"), (" FATAL ", "critical"), ("info", "info")],
)
def test_normalize_log_level_accepts_the_node_fatal_alias(
    configured: str | None,
    normalized: str,
) -> None:
    assert normalize_log_level(configured) == normalized
