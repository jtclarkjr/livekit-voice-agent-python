from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent.contracts import VoiceContextResponse
from agent.voices import (
    CARTESIA_VOICE_IDS,
    DEFAULT_VOICE_KEY,
    resolve_cartesia_voice_id,
)

CONTEXT_RESPONSE = {
    "session": {"id": "voice_1", "status": "active"},
    "messages": [],
    "next_sequence": 0,
}


def test_supported_voice_mapping_is_immutable_and_exact() -> None:
    assert dict(CARTESIA_VOICE_IDS) == {
        "jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        "blake": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
        "robyn": "f31cc6a7-c1e8-4764-980c-60a361443dd1",
    }
    immutable_mapping: Any = CARTESIA_VOICE_IDS
    with pytest.raises(TypeError):
        immutable_mapping["jacqueline"] = "changed"


def test_selected_context_voice_resolves_to_provider_id() -> None:
    context = VoiceContextResponse.model_validate({**CONTEXT_RESPONSE, "voice_key": "blake"})

    assert resolve_cartesia_voice_id(context.voice_key) == ("a167e0f3-df7e-4d52-a9c3-f949145efdab")


def test_missing_voice_key_defaults_during_rollout() -> None:
    context = VoiceContextResponse.model_validate(CONTEXT_RESPONSE)

    assert context.voice_key == DEFAULT_VOICE_KEY
    assert resolve_cartesia_voice_id(context.voice_key) == ("9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")


@pytest.mark.parametrize(
    "voice_key",
    ["daniela", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"],
)
def test_unsupported_or_raw_provider_voice_ids_are_rejected(voice_key: str) -> None:
    with pytest.raises(ValidationError):
        VoiceContextResponse.model_validate({**CONTEXT_RESPONSE, "voice_key": voice_key})
