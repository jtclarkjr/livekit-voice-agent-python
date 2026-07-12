from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.contracts import (
    ContextMessage,
    DispatchMetadata,
    TurnsRequest,
    VoiceContextResponse,
    VoiceTurn,
)
from agent.voices import DEFAULT_VOICE_KEY


def test_dispatch_metadata_is_trimmed_nonblank_and_strict() -> None:
    metadata = DispatchMetadata.model_validate({"voice_session_id": " voice_1 "})

    assert metadata.voice_session_id == "voice_1"
    with pytest.raises(ValidationError):
        DispatchMetadata.model_validate({"voice_session_id": "voice_1", "unexpected": True})
    with pytest.raises(ValidationError):
        DispatchMetadata.model_validate({"voice_session_id": "   "})


@pytest.mark.parametrize(
    "created_at",
    ["2026-07-11", "2026-07-11T00:00:00", "not-a-timestamp"],
)
def test_context_messages_require_an_aware_iso_timestamp(created_at: str) -> None:
    with pytest.raises(ValidationError):
        ContextMessage.model_validate(
            {"role": "user", "content": "Hello", "created_at": created_at}
        )


def test_context_response_defaults_voice_and_allows_backend_extensions() -> None:
    context = VoiceContextResponse.model_validate(
        {
            "session": {
                "id": "voice_1",
                "status": "active",
                "participant_identity": "voice:user:1",
            },
            "messages": [],
            "next_sequence": 4,
            "future_backend_field": "accepted",
        }
    )

    assert context.voice_key == DEFAULT_VOICE_KEY
    assert context.session.model_extra == {"participant_identity": "voice:user:1"}
    assert context.model_extra == {"future_backend_field": "accepted"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice_key", "daniela"),
        ("next_sequence", -1),
        ("next_sequence", "1"),
    ],
)
def test_context_response_rejects_invalid_fields(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "session": {"id": "voice_1", "status": "pending"},
        "messages": [],
        "next_sequence": 0,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        VoiceContextResponse.model_validate(payload)


def test_context_response_rejects_terminal_sessions_and_excess_history() -> None:
    message = {
        "role": "user",
        "content": "Earlier",
        "created_at": "2026-07-11T00:00:00.000Z",
    }
    with pytest.raises(ValidationError):
        VoiceContextResponse.model_validate(
            {
                "session": {"id": "voice_1", "status": "ended"},
                "messages": [],
                "next_sequence": 0,
            }
        )
    with pytest.raises(ValidationError):
        VoiceContextResponse.model_validate(
            {
                "session": {"id": "voice_1", "status": "active"},
                "messages": [message] * 11,
                "next_sequence": 0,
            }
        )


def test_voice_turn_preserves_the_exact_idempotent_wire_shape() -> None:
    turn = VoiceTurn.model_validate(
        {
            "item_id": " item_1 ",
            "sequence": 2,
            "role": "assistant",
            "content": " concise answer ",
            "created_at": "2026-07-11T00:00:00.000+09:00",
            "interrupted": True,
        }
    )

    assert turn.model_dump(mode="json") == {
        "item_id": "item_1",
        "sequence": 2,
        "role": "assistant",
        "content": "concise answer",
        "created_at": "2026-07-11T00:00:00.000+09:00",
        "interrupted": True,
    }


def test_turn_request_and_turns_forbid_unknown_or_coerced_fields() -> None:
    valid_turn = {
        "item_id": "item_1",
        "sequence": 0,
        "role": "user",
        "content": "Hello",
        "created_at": "2026-07-11T00:00:00.000Z",
        "interrupted": False,
    }
    with pytest.raises(ValidationError):
        VoiceTurn.model_validate({**valid_turn, "sequence": "0"})
    with pytest.raises(ValidationError):
        VoiceTurn.model_validate({**valid_turn, "provider_artifact": "private"})
    with pytest.raises(ValidationError):
        TurnsRequest.model_validate({"turns": [valid_turn], "reason": "extra"})
