from __future__ import annotations

import asyncio

import pytest

from agent.rejoin import ParticipantRejoinGuard


@pytest.mark.asyncio
async def test_reconnect_during_status_check_cancels_shutdown() -> None:
    status: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    shutdown_reasons: list[str] = []

    async def session_is_open() -> bool:
        return await status

    guard = ParticipantRejoinGuard(
        grace_seconds=0.01,
        participant_identity="voice:user:session",
        session_is_open=session_is_open,
        shutdown=shutdown_reasons.append,
    )
    guard.participant_connected("voice:user:session")
    guard.participant_disconnected("voice:user:session")
    await asyncio.sleep(0)
    guard.participant_connected("voice:user:session")
    status.set_result(True)
    await asyncio.sleep(0.02)

    assert shutdown_reasons == []
    guard.dispose()


@pytest.mark.asyncio
async def test_explicit_backend_end_shuts_down_immediately() -> None:
    shutdown = asyncio.Event()
    shutdown_reasons: list[str] = []

    async def session_is_open() -> bool:
        return False

    def request_shutdown(reason: str) -> None:
        shutdown_reasons.append(reason)
        shutdown.set()

    guard = ParticipantRejoinGuard(
        grace_seconds=10,
        participant_identity="voice:user:session",
        session_is_open=session_is_open,
        shutdown=request_shutdown,
    )
    guard.participant_connected("voice:user:session")
    guard.participant_disconnected("voice:user:session")
    await asyncio.wait_for(shutdown.wait(), timeout=0.2)

    assert shutdown_reasons == ["backend_session_ended"]
    guard.dispose()


@pytest.mark.asyncio
async def test_open_session_gets_bounded_grace_before_shutdown() -> None:
    shutdown = asyncio.Event()
    shutdown_reasons: list[str] = []

    async def session_is_open() -> bool:
        return True

    def request_shutdown(reason: str) -> None:
        shutdown_reasons.append(reason)
        shutdown.set()

    guard = ParticipantRejoinGuard(
        grace_ms=20,
        participant_identity="voice:user:session",
        session_is_open=session_is_open,
        shutdown=request_shutdown,
    )
    guard.participant_connected("voice:user:session")
    guard.participant_disconnected("voice:user:session")
    await asyncio.sleep(0.005)
    assert shutdown_reasons == []

    await asyncio.wait_for(shutdown.wait(), timeout=0.2)
    assert shutdown_reasons == ["participant_disconnected"]
    guard.dispose()


@pytest.mark.asyncio
async def test_bridge_outage_falls_back_to_bounded_grace() -> None:
    shutdown = asyncio.Event()
    shutdown_reasons: list[str] = []

    async def session_is_open() -> bool:
        raise RuntimeError("bridge unavailable")

    def request_shutdown(reason: str) -> None:
        shutdown_reasons.append(reason)
        shutdown.set()

    guard = ParticipantRejoinGuard(
        grace_ms=1,
        participant_identity="voice:user:session",
        session_is_open=session_is_open,
        shutdown=request_shutdown,
    )
    guard.participant_disconnected("voice:user:session")
    await asyncio.wait_for(shutdown.wait(), timeout=0.2)

    assert shutdown_reasons == ["participant_disconnected"]
    guard.dispose()


@pytest.mark.asyncio
async def test_wrong_participant_and_disposed_guard_are_ignored() -> None:
    calls = 0
    shutdown_reasons: list[str] = []

    async def session_is_open() -> bool:
        nonlocal calls
        calls += 1
        return False

    guard = ParticipantRejoinGuard(
        grace_ms=1,
        participant_identity="expected",
        session_is_open=session_is_open,
        shutdown=shutdown_reasons.append,
    )
    guard.participant_disconnected("other")
    guard.dispose()
    guard.participant_disconnected("expected")
    await asyncio.sleep(0.01)

    assert calls == 0
    assert shutdown_reasons == []


def test_requires_exactly_one_grace_unit() -> None:
    async def session_is_open() -> bool:
        return True

    with pytest.raises(ValueError):
        ParticipantRejoinGuard(
            session_is_open=session_is_open,
            shutdown=lambda _reason: None,
        )
    with pytest.raises(ValueError):
        ParticipantRejoinGuard(
            grace_seconds=1,
            grace_ms=1_000,
            session_is_open=session_is_open,
            shutdown=lambda _reason: None,
        )
