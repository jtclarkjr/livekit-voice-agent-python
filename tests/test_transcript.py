from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from livekit.agents import llm

from agent.contracts import VoiceContextResponse, VoiceTurn
from agent.transcript import (
    TranscriptLedger,
    TurnPersistence,
    build_initial_chat_context,
)


class _Bridge:
    def __init__(self) -> None:
        self.persisted: list[VoiceTurn] = []
        self.persist_error: Exception | None = None
        self.persist_calls = 0

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
        del session_id
        self.persist_calls += 1
        if self.persist_error is not None and self.persist_calls == 1:
            raise self.persist_error
        self.persisted.extend(turn.model_copy(deep=True) for turn in turns)

    async def finalize_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        end_reason: str,
    ) -> None:
        del session_id, turns, end_reason

    async def fail_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        reason: str,
    ) -> None:
        del session_id, turns, reason


def _message(
    *,
    item_id: str,
    role: llm.ChatRole,
    content: str,
    created_at: float = 1_783_728_000.0,
    interrupted: bool = False,
) -> llm.ChatMessage:
    return llm.ChatMessage(
        id=item_id,
        role=role,
        content=[content],
        created_at=created_at,
        interrupted=interrupted,
    )


def test_loads_prior_messages_chronologically_and_excludes_them() -> None:
    initial = build_initial_chat_context(
        [
            {
                "role": "assistant",
                "content": "Second",
                "created_at": "2026-07-11T00:00:02.000Z",
            },
            {
                "role": "user",
                "content": "First",
                "created_at": "2026-07-11T00:00:01.000Z",
            },
        ]
    )
    ledger = TranscriptLedger(initial.context_item_ids)

    assert [item.text_content for item in initial.chat_context.messages()] == [
        "First",
        "Second",
    ]
    assert ledger.reconcile(initial.chat_context.items) == []


def test_stable_sort_preserves_backend_order_for_equal_timestamps() -> None:
    timestamp = "2026-07-11T00:00:00.000Z"
    initial = build_initial_chat_context(
        [
            {"role": "user", "content": "First", "created_at": timestamp},
            {"role": "assistant", "content": "Second", "created_at": timestamp},
        ]
    )

    assert [item.text_content for item in initial.chat_context.messages()] == [
        "First",
        "Second",
    ]


def test_excludes_cloned_context_ids_captured_after_session_start() -> None:
    ledger = TranscriptLedger(frozenset({"voice_context_0"}))
    cloned = _message(item_id="cloned_by_session", role="user", content="Prior")

    ledger.exclude_context_items([cloned])

    assert ledger.reconcile([cloned]) == []


def test_preserves_sequence_and_truncates_interrupted_assistant_item() -> None:
    ledger = TranscriptLedger(set())
    original = _message(
        item_id="assistant_1",
        role="assistant",
        content="This answer continued too far",
    )
    interrupted = _message(
        item_id="assistant_1",
        role="assistant",
        content="This answer",
        interrupted=True,
    )

    original_turn = ledger.upsert(original)
    assert original_turn is not None
    assert original_turn.sequence == 0
    updated = ledger.upsert(interrupted)

    assert updated is not None
    assert updated.sequence == 0
    assert updated.content == "This answer"
    assert updated.interrupted is True
    assert len(ledger.snapshot()) == 1


def test_continues_from_backend_sequence_after_rejoin() -> None:
    turn = TranscriptLedger(set(), 7).upsert(
        _message(item_id="user_rejoined", role="user", content="Back")
    )

    assert turn is not None
    assert turn.sequence == 7


def test_ignores_function_calls_system_messages_and_blank_content() -> None:
    ledger = TranscriptLedger(set())
    function_call = llm.FunctionCall(
        call_id="search_call_1",
        name="searchWeb",
        arguments='{"query":"private"}',
    )
    system = _message(item_id="system_1", role="system", content="hidden")
    blank = _message(item_id="blank_1", role="user", content="   ")
    answer = _message(
        item_id="assistant_search_answer",
        role="assistant",
        content="Here is the concise answer.",
    )

    turns = ledger.reconcile([function_call, system, blank, answer])

    assert [turn.item_id for turn in turns] == ["assistant_search_answer"]


@pytest.mark.asyncio
async def test_serializes_live_writes_in_event_order() -> None:
    bridge = _Bridge()
    persistence = TurnPersistence(
        backend=bridge,
        ledger=TranscriptLedger(set()),
        session_id="voice_1",
    )

    persistence.enqueue(_message(item_id="user_1", role="user", content="Hello"))
    persistence.enqueue(_message(item_id="assistant_1", role="assistant", content="Hi"))
    await persistence.flush()

    assert [(turn.item_id, turn.sequence) for turn in bridge.persisted] == [
        ("user_1", 0),
        ("assistant_1", 1),
    ]


@pytest.mark.asyncio
async def test_live_write_failure_is_recorded_and_does_not_block_later_turns() -> None:
    bridge = _Bridge()
    bridge.persist_error = RuntimeError("bridge unavailable")
    persistence = TurnPersistence(
        backend=bridge,
        ledger=TranscriptLedger(set()),
        session_id="voice_1",
    )

    persistence.enqueue(_message(item_id="user_1", role="user", content="Hello"))
    persistence.enqueue(_message(item_id="assistant_1", role="assistant", content="Hi"))
    await persistence.flush()

    assert len(persistence.failures) == 1
    assert str(persistence.failures[0]) == "bridge unavailable"
    assert [turn.item_id for turn in bridge.persisted] == ["assistant_1"]


@pytest.mark.asyncio
async def test_event_snapshot_is_not_rewritten_by_later_interruption_update() -> None:
    release_first = asyncio.Event()

    class BlockingBridge(_Bridge):
        async def persist_turns(
            self,
            session_id: str,
            turns: Sequence[VoiceTurn],
        ) -> None:
            if self.persist_calls == 0:
                self.persist_calls += 1
                await release_first.wait()
                self.persisted.extend(turns)
                return
            await super().persist_turns(session_id, turns)

    bridge = BlockingBridge()
    persistence = TurnPersistence(
        backend=bridge,
        ledger=TranscriptLedger(set()),
        session_id="voice_1",
    )
    persistence.enqueue(
        _message(
            item_id="assistant_1",
            role="assistant",
            content="Long answer that continues",
        )
    )
    await asyncio.sleep(0)
    persistence.enqueue(
        _message(
            item_id="assistant_1",
            role="assistant",
            content="Long answer",
            interrupted=True,
        )
    )
    release_first.set()
    await persistence.flush()

    assert [turn.content for turn in bridge.persisted] == [
        "Long answer that continues",
        "Long answer",
    ]
    assert [turn.sequence for turn in bridge.persisted] == [0, 0]
