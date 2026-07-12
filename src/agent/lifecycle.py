"""Once-only bridge settlement primitives for completed voice sessions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, TypedDict

from livekit.agents import llm

from .backend import VoiceBackendBridge
from .transcript import TranscriptLedger, TurnPersistence


class FinalizedSettlement(TypedDict):
    status: Literal["finalized"]
    turn_count: int


class FailedSettlement(TypedDict):
    status: Literal["failed"]
    reason: str
    turn_count: int


type SessionSettlement = FinalizedSettlement | FailedSettlement


def _describe_error(error: object) -> str:
    message = re.sub(r"\s+", " ", str(error)).strip()[:500]
    return message or "unknown error"


async def settle_voice_session(
    *,
    backend: VoiceBackendBridge,
    close_reason: str,
    history: Sequence[llm.ChatItem],
    ledger: TranscriptLedger,
    persistence: TurnPersistence,
    session_id: str,
    fatal_reason: str | None = None,
) -> SessionSettlement:
    await persistence.flush()
    turns = ledger.reconcile(history)

    if fatal_reason is not None:
        await backend.fail_session(session_id, turns, fatal_reason)
        return {
            "status": "failed",
            "reason": fatal_reason,
            "turn_count": len(turns),
        }

    try:
        await backend.finalize_session(session_id, turns, close_reason)
    except Exception as finalize_error:
        reason = f"finalization failed: {_describe_error(finalize_error)}"
        try:
            await backend.fail_session(session_id, turns, reason)
        except Exception as failure_error:
            raise ExceptionGroup(
                "Both voice-session finalization and failure reporting failed",
                [finalize_error, failure_error],
            ) from failure_error
        return {"status": "failed", "reason": reason, "turn_count": len(turns)}

    return {"status": "finalized", "turn_count": len(turns)}


def failure_reason(prefix: str, error: object) -> str:
    return f"{prefix}: {_describe_error(error)}"


__all__ = ["SessionSettlement", "failure_reason", "settle_voice_session"]
