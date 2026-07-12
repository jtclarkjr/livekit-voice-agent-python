"""Durable transcript projection and ordered live-turn persistence."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from livekit.agents import llm

from .backend import VoiceBackendBridge
from .contracts import ChatRole, ContextMessage, VoiceTurn

type ContextMessageInput = ContextMessage | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class InitialChatContext:
    chat_context: llm.ChatContext
    context_item_ids: frozenset[str]


def _timestamp_seconds(value: str) -> float:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).timestamp()


def _iso_timestamp(timestamp: float) -> str:
    valid_timestamp = timestamp if math.isfinite(timestamp) else time.time()
    return (
        datetime.fromtimestamp(valid_timestamp, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def build_initial_chat_context(
    messages: Sequence[ContextMessageInput],
) -> InitialChatContext:
    validated = [
        message if isinstance(message, ContextMessage) else ContextMessage.model_validate(message)
        for message in messages
    ]
    chronological_messages = sorted(
        enumerate(validated),
        key=lambda entry: (_timestamp_seconds(entry[1].created_at), entry[0]),
    )

    chat_context = llm.ChatContext.empty()
    context_item_ids: set[str] = set()
    for index, (_, message) in enumerate(chronological_messages):
        item_id = f"voice_context_{index}"
        context_item_ids.add(item_id)
        chat_context.add_message(
            id=item_id,
            role=message.role,
            content=message.content,
            created_at=_timestamp_seconds(message.created_at),
            extra={"persisted_voice_context": True},
        )

    return InitialChatContext(
        chat_context=chat_context,
        context_item_ids=frozenset(context_item_ids),
    )


class TranscriptLedger:
    def __init__(
        self,
        context_item_ids: set[str] | frozenset[str],
        next_sequence: int = 0,
    ) -> None:
        self._context_item_ids = set(context_item_ids)
        self._turns_by_item_id: dict[str, VoiceTurn] = {}
        self._next_sequence = next_sequence

    def exclude_context_items(self, items: Sequence[llm.ChatItem]) -> None:
        for item in items:
            if isinstance(item, llm.ChatMessage) and item.role in {"user", "assistant"}:
                self._context_item_ids.add(item.id)

    def upsert(self, item: object) -> VoiceTurn | None:
        if not isinstance(item, llm.ChatMessage):
            return None
        if item.id in self._context_item_ids:
            return None
        if item.extra.get("persisted_voice_context") is True:
            return None
        role: ChatRole
        if item.role == "user":
            role = "user"
        elif item.role == "assistant":
            role = "assistant"
        else:
            return None

        content = (item.text_content or "").strip()
        if not content:
            return None

        previous = self._turns_by_item_id.get(item.id)
        turn = VoiceTurn(
            item_id=item.id,
            sequence=(previous.sequence if previous is not None else self._next_sequence),
            role=role,
            content=content,
            created_at=_iso_timestamp(item.created_at),
            interrupted=item.interrupted,
        )
        if previous is None:
            self._next_sequence += 1
        self._turns_by_item_id[item.id] = turn
        return turn.model_copy(deep=True)

    def reconcile(self, items: Sequence[llm.ChatItem]) -> list[VoiceTurn]:
        for item in items:
            self.upsert(item)
        return self.snapshot()

    def snapshot(self) -> list[VoiceTurn]:
        return [
            turn.model_copy(deep=True)
            for turn in sorted(
                self._turns_by_item_id.values(),
                key=lambda candidate: candidate.sequence,
            )
        ]


class TurnPersistence:
    def __init__(
        self,
        *,
        backend: VoiceBackendBridge,
        ledger: TranscriptLedger,
        session_id: str,
    ) -> None:
        self._backend = backend
        self._ledger = ledger
        self._session_id = session_id
        self._errors: list[Exception] = []
        self._queue: asyncio.Queue[VoiceTurn] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def enqueue(self, item: object) -> None:
        turn = self._ledger.upsert(item)
        if turn is None:
            return

        self._queue.put_nowait(turn.model_copy(deep=True))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            try:
                turn = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                await self._backend.persist_turns(self._session_id, [turn])
            except Exception as error:
                self._errors.append(error)
            finally:
                self._queue.task_done()

    async def flush(self) -> None:
        await self._queue.join()
        if self._worker is not None:
            await self._worker

    @property
    def failures(self) -> tuple[Exception, ...]:
        return tuple(self._errors)


__all__ = [
    "InitialChatContext",
    "TranscriptLedger",
    "TurnPersistence",
    "build_initial_chat_context",
]
