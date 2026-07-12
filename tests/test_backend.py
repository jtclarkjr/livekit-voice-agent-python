from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest
from pydantic import ValidationError

from agent.backend import VoiceBackendClient, VoiceBackendError
from agent.contracts import VoiceTurn

TURN = VoiceTurn(
    item_id="item_1",
    sequence=0,
    role="user",
    content="Hello",
    created_at="2026-07-11T00:00:00.000Z",
    interrupted=False,
)


def _client(
    http_client: httpx.AsyncClient,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> VoiceBackendClient:
    if sleep is None:
        return VoiceBackendClient(
            http_client=http_client,
            base_url="http://localhost:4000",
            bridge_token="bridge-secret",
            max_attempts=3,
            retry_base_ms=1,
            timeout_ms=1_000,
        )
    return VoiceBackendClient(
        http_client=http_client,
        base_url="http://localhost:4000",
        bridge_token="bridge-secret",
        max_attempts=3,
        retry_base_ms=1,
        timeout_ms=1_000,
        sleep=sleep,
    )


@pytest.mark.asyncio
async def test_retries_retryable_responses_and_validates_context() -> None:
    attempts = 0
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        requests.append(request)
        if attempts == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            json={
                "session": {"id": "voice_1", "status": "active"},
                "next_sequence": 4,
                "messages": [
                    {
                        "role": "user",
                        "content": "Earlier message",
                        "created_at": "2026-07-11T00:00:00.000Z",
                    }
                ],
            },
        )

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        context = await _client(http, sleep=sleep).get_context("voice_1")

        assert not http.is_closed

    assert context.session.id == "voice_1"
    assert context.next_sequence == 4
    assert context.voice_key == "jacqueline"
    assert attempts == 2
    assert sleeps == [0.001]
    assert str(requests[-1].url) == (
        "http://localhost:4000/internal/voice/sessions/voice_1/context"
    )
    assert requests[-1].headers["authorization"] == "Bearer bridge-secret"


@pytest.mark.asyncio
async def test_sends_exact_started_and_turn_envelopes_with_encoded_session_id() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = _client(http)
        await client.mark_started(
            "voice/with space",
            job_id="AJ_job_1",
            room_id="RM_room_1",
        )
        await client.persist_turns("voice_1", [TURN])

    assert requests[0].url.raw_path.endswith(
        b"/internal/voice/sessions/voice%2Fwith%20space/started"
    )
    assert json.loads(requests[0].content) == {
        "job_id": "AJ_job_1",
        "room_id": "RM_room_1",
    }
    assert json.loads(requests[1].content) == {"turns": [TURN.model_dump(mode="json")]}


@pytest.mark.asyncio
async def test_finalize_and_fail_use_the_exact_bridge_envelopes() -> None:
    bodies: list[object] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = _client(http)
        await client.finalize_session("voice_1", [TURN], "participant_disconnected")
        await client.fail_session("voice_1", [TURN], "model failed")

    turns = [TURN.model_dump(mode="json")]
    assert bodies == [
        {"turns": turns, "end_reason": "participant_disconnected"},
        {"turns": turns, "reason": "model failed"},
    ]


@pytest.mark.asyncio
async def test_does_not_retry_authorization_failures() -> None:
    attempts = 0
    sleeps: list[float] = []

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(403, text="forbidden")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(VoiceBackendError) as raised:
            await _client(http, sleep=sleep).mark_started("voice_1")

    assert raised.value.status == 403
    assert raised.value.retryable is False
    assert attempts == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_network_errors_retry_with_exponential_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("slow bridge", request=request)
        return httpx.Response(200, json={"ok": True})

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await _client(http, sleep=sleep).mark_started("voice_1")

    assert attempts == 3
    assert sleeps == [0.001, 0.002]


@pytest.mark.asyncio
async def test_invalid_context_is_not_retried() -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={
                "session": {"id": "voice_1", "status": "ended"},
                "next_sequence": 0,
                "messages": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ValidationError):
            await _client(http).get_context("voice_1")

    assert attempts == 1
