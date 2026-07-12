from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from livekit.agents import llm

from agent.contracts import VoiceContextResponse, VoiceTurn
from agent.lifecycle import failure_reason, settle_voice_session
from agent.transcript import TranscriptLedger, TurnPersistence


class _Bridge:
    def __init__(self) -> None:
        self.finalize_error: Exception | None = None
        self.fail_error: Exception | None = None
        self.finalized: list[tuple[str, list[VoiceTurn], str]] = []
        self.failed: list[tuple[str, list[VoiceTurn], str]] = []
        self.events: list[str] = []

    async def get_context(self, session_id: str) -> VoiceContextResponse:
        del session_id
        raise NotImplementedError

    async def mark_started(
        self,
        session_id: str,
        identifiers: Mapping[str, str] | None = None,
        *,
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        del session_id, identifiers, job_id, room_id

    async def persist_turns(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
    ) -> None:
        del session_id, turns
        self.events.append("persist")

    async def finalize_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        end_reason: str,
    ) -> None:
        self.events.append("finalize")
        if self.finalize_error is not None:
            raise self.finalize_error
        self.finalized.append((session_id, list(turns), end_reason))

    async def fail_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        reason: str,
    ) -> None:
        self.events.append("fail")
        if self.fail_error is not None:
            raise self.fail_error
        self.failed.append((session_id, list(turns), reason))


def _history() -> list[llm.ChatItem]:
    return [
        llm.ChatMessage(id="user_1", role="user", content=["Hello"]),
        llm.ChatMessage(
            id="assistant_1",
            role="assistant",
            content=["Hi there"],
            interrupted=True,
        ),
    ]


def _state(backend: _Bridge) -> tuple[TranscriptLedger, TurnPersistence]:
    ledger = TranscriptLedger(set())
    persistence = TurnPersistence(
        backend=backend,
        ledger=ledger,
        session_id="voice_1",
    )
    return ledger, persistence


@pytest.mark.asyncio
async def test_finalizes_with_reconciled_complete_history() -> None:
    backend = _Bridge()
    ledger, persistence = _state(backend)

    result = await settle_voice_session(
        backend=backend,
        close_reason="participant_disconnected",
        history=_history(),
        ledger=ledger,
        persistence=persistence,
        session_id="voice_1",
    )

    assert result == {"status": "finalized", "turn_count": 2}
    session_id, turns, reason = backend.finalized[0]
    assert session_id == "voice_1"
    assert reason == "participant_disconnected"
    assert turns[1].interrupted is True
    assert turns[1].sequence == 1


@pytest.mark.asyncio
async def test_reports_partial_history_after_unrecoverable_model_error() -> None:
    backend = _Bridge()
    ledger, persistence = _state(backend)

    result = await settle_voice_session(
        backend=backend,
        close_reason="error",
        fatal_reason="unrecoverable LiveKit model error",
        history=_history(),
        ledger=ledger,
        persistence=persistence,
        session_id="voice_1",
    )

    assert result["status"] == "failed"
    assert backend.finalized == []
    assert backend.failed[0][0] == "voice_1"
    assert backend.failed[0][2] == "unrecoverable LiveKit model error"
    assert [turn.item_id for turn in backend.failed[0][1]] == [
        "user_1",
        "assistant_1",
    ]


@pytest.mark.asyncio
async def test_falls_back_to_failure_endpoint_when_finalization_fails() -> None:
    backend = _Bridge()
    backend.finalize_error = RuntimeError("bridge\n unavailable")
    ledger, persistence = _state(backend)

    result = await settle_voice_session(
        backend=backend,
        close_reason="job_shutdown",
        history=_history(),
        ledger=ledger,
        persistence=persistence,
        session_id="voice_1",
    )

    assert result == {
        "status": "failed",
        "reason": "finalization failed: bridge unavailable",
        "turn_count": 2,
    }
    assert backend.failed[0][2] == "finalization failed: bridge unavailable"


@pytest.mark.asyncio
async def test_raises_exception_group_when_both_settlement_writes_fail() -> None:
    backend = _Bridge()
    backend.finalize_error = RuntimeError("finalize unavailable")
    backend.fail_error = RuntimeError("fail unavailable")
    ledger, persistence = _state(backend)

    with pytest.raises(ExceptionGroup) as raised:
        await settle_voice_session(
            backend=backend,
            close_reason="job_shutdown",
            history=_history(),
            ledger=ledger,
            persistence=persistence,
            session_id="voice_1",
        )

    assert [str(error) for error in raised.value.exceptions] == [
        "finalize unavailable",
        "fail unavailable",
    ]


@pytest.mark.asyncio
async def test_flushes_live_turn_writes_before_finalization() -> None:
    backend = _Bridge()
    ledger, persistence = _state(backend)
    persistence.enqueue(llm.ChatMessage(id="user_1", role="user", content=["Hello"]))

    await settle_voice_session(
        backend=backend,
        close_reason="job_shutdown",
        history=_history(),
        ledger=ledger,
        persistence=persistence,
        session_id="voice_1",
    )

    assert backend.events == ["persist", "finalize"]


def test_failure_reason_is_single_line_bounded_and_nonempty() -> None:
    assert failure_reason("startup failed", RuntimeError("bad\n  bridge")) == (
        "startup failed: bad bridge"
    )
    assert failure_reason("startup failed", RuntimeError("")) == ("startup failed: unknown error")
    assert len(failure_reason("startup failed", RuntimeError("x" * 600))) == (
        len("startup failed: ") + 500
    )
