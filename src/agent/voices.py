"""Stable product voice keys and their Cartesia provider IDs."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

type VoiceKey = Literal["jacqueline", "blake", "robyn"]

DEFAULT_VOICE_KEY: VoiceKey = "jacqueline"

CARTESIA_VOICE_IDS: MappingProxyType[VoiceKey, str] = MappingProxyType(
    {
        "jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        "blake": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
        "robyn": "f31cc6a7-c1e8-4764-980c-60a361443dd1",
    }
)


def resolve_cartesia_voice_id(voice_key: VoiceKey) -> str:
    return CARTESIA_VOICE_IDS[voice_key]


__all__ = [
    "CARTESIA_VOICE_IDS",
    "DEFAULT_VOICE_KEY",
    "VoiceKey",
    "resolve_cartesia_voice_id",
]
