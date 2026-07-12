"""Validated wire contracts shared with the internal Elixir voice bridge."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from .voices import DEFAULT_VOICE_KEY, VoiceKey

type ChatRole = Literal["user", "assistant"]

NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

_ISO_DATETIME_WITH_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def _validate_aware_iso_datetime(value: str) -> str:
    if _ISO_DATETIME_WITH_OFFSET.fullmatch(value) is None:
        raise ValueError("timestamp must be an ISO 8601 datetime with an offset")

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


AwareIsoTimestamp = Annotated[str, AfterValidator(_validate_aware_iso_datetime)]


class _StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class DispatchMetadata(_StrictContract):
    voice_session_id: NonBlankString


class ContextMessage(_StrictContract):
    role: ChatRole
    content: str
    created_at: AwareIsoTimestamp


class VoiceSession(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    id: NonBlankString
    status: Literal["pending", "active"]


class VoiceContextResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    session: VoiceSession
    voice_key: VoiceKey = DEFAULT_VOICE_KEY
    messages: list[ContextMessage] = Field(max_length=10)
    next_sequence: int = Field(ge=0)


class VoiceTurn(_StrictContract):
    item_id: NonBlankString
    sequence: int = Field(ge=0)
    role: ChatRole
    content: NonBlankString
    created_at: AwareIsoTimestamp
    interrupted: bool


class TurnsRequest(_StrictContract):
    turns: list[VoiceTurn]


__all__ = [
    "ContextMessage",
    "DispatchMetadata",
    "TurnsRequest",
    "VoiceContextResponse",
    "VoiceSession",
    "VoiceTurn",
]
