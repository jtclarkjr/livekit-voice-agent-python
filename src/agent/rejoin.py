"""Participant reconnect grace handling for transient room disconnects."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

type RejoinShutdownReason = Literal[
    "backend_session_ended",
    "participant_disconnected",
]
type SessionIsOpen = Callable[[], Awaitable[bool]]
type Shutdown = Callable[[RejoinShutdownReason], None]


class ParticipantRejoinGuard:
    def __init__(
        self,
        *,
        session_is_open: SessionIsOpen,
        shutdown: Shutdown,
        participant_identity: str | None = None,
        grace_seconds: float | None = None,
        grace_ms: int | None = None,
    ) -> None:
        if (grace_seconds is None) == (grace_ms is None):
            raise ValueError("provide exactly one of grace_seconds or grace_ms")
        if grace_seconds is not None:
            resolved_grace = grace_seconds
        elif grace_ms is not None:
            resolved_grace = grace_ms / 1_000
        else:  # pragma: no cover - guarded by the exact-one validation above.
            raise AssertionError("unreachable")
        self._grace_seconds = max(0.0, resolved_grace)
        self._session_is_open = session_is_open
        self._shutdown = shutdown
        self._participant_identity = participant_identity
        self._connected = False
        self._disposed = False
        self._generation = 0
        self._shutdown_requested = False
        self._status_tasks: set[asyncio.Task[None]] = set()
        self._timer_task: asyncio.Task[None] | None = None

    def participant_connected(self, identity: str) -> None:
        if self._disposed or self._shutdown_requested:
            return
        if self._participant_identity is None:
            self._participant_identity = identity
        if identity != self._participant_identity:
            return

        self._connected = True
        self._generation += 1
        self._cancel_timer()

    def participant_disconnected(self, identity: str) -> None:
        if (
            self._disposed
            or self._shutdown_requested
            or self._participant_identity is None
            or identity != self._participant_identity
        ):
            return

        self._connected = False
        self._generation += 1
        generation = self._generation
        task = asyncio.get_running_loop().create_task(self._resolve_disconnect(generation))
        self._status_tasks.add(task)
        task.add_done_callback(self._status_done)

    def dispose(self) -> None:
        self._disposed = True
        self._generation += 1
        self._cancel_timer()

    async def _resolve_disconnect(self, generation: int) -> None:
        session_open = True
        try:
            session_open = await self._session_is_open()
        except Exception:
            # A bridge outage must not strand the worker; retain the bounded grace.
            pass

        if not self._is_current_disconnect(generation):
            return
        if not session_open:
            self._request_shutdown("backend_session_ended")
            return

        self._cancel_timer()
        timer_task = asyncio.get_running_loop().create_task(self._wait_for_grace(generation))
        self._timer_task = timer_task
        timer_task.add_done_callback(self._timer_done)

    async def _wait_for_grace(self, generation: int) -> None:
        await asyncio.sleep(self._grace_seconds)
        if self._is_current_disconnect(generation):
            self._request_shutdown("participant_disconnected")

    def _is_current_disconnect(self, generation: int) -> bool:
        return (
            not self._disposed
            and not self._shutdown_requested
            and not self._connected
            and generation == self._generation
        )

    def _request_shutdown(self, reason: RejoinShutdownReason) -> None:
        if self._disposed or self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._cancel_timer()
        self._shutdown(reason)

    def _cancel_timer(self) -> None:
        task = self._timer_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        self._timer_task = None

    def _status_done(self, task: asyncio.Task[None]) -> None:
        self._status_tasks.discard(task)
        self._report_task_exception(task)

    def _timer_done(self, task: asyncio.Task[None]) -> None:
        if self._timer_task is task:
            self._timer_task = None
        self._report_task_exception(task)

    @staticmethod
    def _report_task_exception(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            task.get_loop().call_exception_handler(
                {
                    "message": "Participant rejoin guard task failed",
                    "exception": exception,
                    "task": task,
                }
            )


__all__ = ["ParticipantRejoinGuard", "RejoinShutdownReason"]
