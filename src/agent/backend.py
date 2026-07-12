"""Authenticated async client for the Elixir internal voice bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, TypedDict
from urllib.parse import quote

import httpx

from .contracts import TurnsRequest, VoiceContextResponse, VoiceTurn

type Sleep = Callable[[float], Awaitable[None]]


class StartIdentifiers(TypedDict, total=False):
    job_id: str
    room_id: str


class VoiceBackendBridge(Protocol):
    async def fail_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        reason: str,
    ) -> None: ...

    async def finalize_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        end_reason: str,
    ) -> None: ...

    async def get_context(self, session_id: str) -> VoiceContextResponse: ...

    async def mark_started(
        self,
        session_id: str,
        identifiers: Mapping[str, str] | None = None,
        *,
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None: ...

    async def persist_turns(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
    ) -> None: ...


class VoiceBackendError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        if cause is not None:
            self.__cause__ = cause


def _is_retryable_status(status: int) -> bool:
    return status in {408, 425, 429} or status >= 500


def _request_path(session_id: str, action: str | None = None) -> str:
    encoded_session_id = quote(session_id, safe="!~*'()-._")
    suffix = "" if action is None else f"/{action}"
    return f"internal/voice/sessions/{encoded_session_id}{suffix}"


class VoiceBackendClient(VoiceBackendBridge):
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str,
        bridge_token: str,
        max_attempts: int,
        retry_base_ms: int,
        timeout_ms: int,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http_client = http_client
        self._base_url = f"{base_url.rstrip('/')}/"
        self._bridge_token = bridge_token
        self._max_attempts = max_attempts
        self._retry_base_ms = retry_base_ms
        self._timeout_seconds = timeout_ms / 1_000
        self._sleep = sleep

    async def get_context(self, session_id: str) -> VoiceContextResponse:
        response = await self._request(
            _request_path(session_id, "context"),
            method="GET",
        )
        return VoiceContextResponse.model_validate(response.json())

    async def mark_started(
        self,
        session_id: str,
        identifiers: Mapping[str, str] | None = None,
        *,
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        body = dict(identifiers or {})
        if job_id is not None:
            body["job_id"] = job_id
        if room_id is not None:
            body["room_id"] = room_id
        await self._post(_request_path(session_id, "started"), body)

    async def persist_turns(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
    ) -> None:
        body = TurnsRequest(turns=list(turns)).model_dump(mode="json")
        await self._post(_request_path(session_id, "turns"), body)

    async def finalize_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        end_reason: str,
    ) -> None:
        body = TurnsRequest(turns=list(turns)).model_dump(mode="json")
        body["end_reason"] = end_reason
        await self._post(_request_path(session_id, "finalize"), body)

    async def fail_session(
        self,
        session_id: str,
        turns: Sequence[VoiceTurn],
        reason: str,
    ) -> None:
        body = TurnsRequest(turns=list(turns)).model_dump(mode="json")
        body["reason"] = reason
        await self._post(_request_path(session_id, "fail"), body)

    async def _post(self, path: str, body: object) -> None:
        response = await self._request(path, method="POST", body=body)
        await response.aread()

    async def _request(
        self,
        path: str,
        *,
        method: str,
        body: object | None = None,
    ) -> httpx.Response:
        last_error: VoiceBackendError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._request_once(path, method=method, body=body)
            except VoiceBackendError as error:
                last_error = error
                if not error.retryable or attempt == self._max_attempts:
                    raise
                delay_seconds = self._retry_base_ms * (2 ** (attempt - 1)) / 1_000
                await self._sleep(delay_seconds)

        detail = str(last_error) if last_error is not None else "unknown error"
        raise VoiceBackendError(
            f"Bridge request failed: {detail}",
            retryable=False,
            cause=last_error,
        )

    async def _request_once(
        self,
        path: str,
        *,
        method: str,
        body: object | None,
    ) -> httpx.Response:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self._bridge_token}",
        }

        try:
            response = await self._http_client.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=body if body is not None else None,
                timeout=self._timeout_seconds,
            )
        except Exception as error:
            raise VoiceBackendError(
                f"Bridge request failed: {error}",
                retryable=True,
                cause=error,
            ) from error

        if response.is_success:
            return response

        response_text = response.text[:500]
        suffix = f": {response_text}" if response_text else ""
        raise VoiceBackendError(
            f"Bridge request failed with HTTP {response.status_code}{suffix}",
            retryable=_is_retryable_status(response.status_code),
            status=response.status_code,
        )


__all__ = [
    "StartIdentifiers",
    "VoiceBackendBridge",
    "VoiceBackendClient",
    "VoiceBackendError",
]
