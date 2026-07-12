"""Runtime wiring tests for the LiveKit job entrypoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc
from livekit.agents import Agent
from livekit.plugins import ai_coustics

from agent import worker
from agent.config import VoiceAgentConfig
from agent.constants import (
    STT_LANGUAGE,
    STT_MODEL,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
)


@pytest.mark.asyncio
async def test_model_and_audio_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")

    session = worker.create_agent_session("jacqueline")
    session_state = cast(Any, session)
    try:
        assert session_state._opts.max_tool_steps == 1
        assert session_state._opts.turn_handling["preemptive_generation"]["enabled"]
        assert session_state._turn_detection.__class__.__name__ == "TurnDetector"
        assert session_state._stt._opts.model == STT_MODEL
        assert session_state._stt._opts.language == STT_LANGUAGE
        assert session_state._tts._opts.model == TTS_MODEL
        assert session_state._tts._opts.sample_rate == TTS_SAMPLE_RATE

        options = worker.create_room_options("participant-1")
        assert options.participant_identity == "participant-1"
        assert options.close_on_disconnect is False
        audio_output = cast(Any, options.audio_output)
        audio_input = cast(Any, options.audio_input)
        assert audio_output.sample_rate == 48_000
        assert audio_output.num_channels == 1
        enhancer = audio_input.noise_cancellation
        assert enhancer._model is ai_coustics.EnhancerModel.QUAIL_VF_S
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_initial_agent_state_is_replayed_after_room_connect() -> None:
    attributes: list[dict[str, str]] = []

    class LocalParticipant:
        async def set_attributes(self, value: dict[str, str]) -> None:
            attributes.append(value)

    await worker.sync_room_agent_state(
        cast(Any, SimpleNamespace(agent_state="listening")),
        cast(Any, SimpleNamespace(local_participant=LocalParticipant())),
    )

    assert attributes == [{"lk.agent.state": "listening"}]


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:'asyncio.iscoroutinefunction' is deprecated.*:DeprecationWarning"
)
async def test_room_runtime_prepares_before_connection_with_explicit_participant() -> None:
    session = worker.AgentSession(turn_handling={"turn_detection": None})
    runtime: worker.RoomRuntime | None = None
    try:
        await session.start(
            Agent(
                instructions="test agent",
                stt=None,
                vad=None,
                llm=None,
                tts=None,
            ),
            record=False,
        )
        runtime = await worker.prepare_room_runtime(
            session,
            rtc.Room(),
            worker.create_room_options("participant-1"),
        )

        assert cast(Any, runtime.adapter)._participant_identity == "participant-1"
    finally:
        if runtime is not None:
            await runtime.aclose()
        await session.aclose()


def test_dispatch_and_log_level_compatibility() -> None:
    assert (
        worker.parse_dispatch_metadata('{"voice_session_id":"voice-session-1"}').voice_session_id
        == "voice-session-1"
    )
    with pytest.raises(ValueError):
        worker.parse_dispatch_metadata('{"voice_session_id":"voice-session-1","unexpected":true}')

    environment = {"LOG_LEVEL": "fatal"}
    assert worker.configure_livekit_log_level(environment) == "critical"
    assert environment["LIVEKIT_LOG_LEVEL"] == "critical"

    warning_environment = {"LOG_LEVEL": "warning"}
    assert worker.configure_livekit_log_level(warning_environment) == "warn"
    assert warning_environment["LIVEKIT_LOG_LEVEL"] == "warn"

    trace_environment = {"LOG_LEVEL": "trace"}
    assert worker.configure_livekit_log_level(trace_environment) == "trace"
    assert trace_environment["LIVEKIT_LOG_LEVEL"] == "trace"


class _FakeRoom:
    def __init__(self) -> None:
        self.sid = "room-1"
        self.remote_participants: dict[str, Any] = {}
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, callback: Any) -> None:
        self.handlers[event] = callback

    def off(self, event: str, callback: Any) -> None:
        if self.handlers.get(event) is callback:
            self.handlers.pop(event)


class _FakeContext:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.room = _FakeRoom()
        self.job = SimpleNamespace(
            id="job-1",
            metadata='{"voice_session_id":"voice-session-1"}',
            participant=SimpleNamespace(identity="participant-1"),
            room=SimpleNamespace(sid="room-1"),
        )
        self.shutdown_callbacks: list[Any] = []
        self.recording_options: dict[str, bool] | None = None

    def init_recording(self, options: dict[str, bool]) -> None:
        self.recording_options = options

    def add_shutdown_callback(self, callback: Any) -> None:
        self.shutdown_callbacks.append(callback)

    async def connect(self) -> None:
        self.events.append("connect")

    def shutdown(self, *, reason: str) -> None:
        self.events.append(f"shutdown:{reason}")

    def make_session_report(self, _session: Any) -> Any:
        return SimpleNamespace(chat_history=SimpleNamespace(items=[]))


class _FakeSession:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.handlers: dict[str, Any] = {}
        self.history = SimpleNamespace(items=[SimpleNamespace(id="cloned-context")])
        self.start_kwargs: dict[str, Any] = {}

    def on(self, event: str, callback: Any) -> None:
        self.handlers[event] = callback
        if event == "conversation_item_added":
            self.events.append("subscribe")

    async def start(self, **kwargs: Any) -> None:
        self.events.append("session.start")
        self.start_kwargs = kwargs


class _FakeLedger:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def exclude_context_items(self, _items: Any) -> None:
        self.events.append("exclude-context")


class _FakePersistence:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.failures: tuple[Exception, ...] = ()

    def enqueue(self, _item: Any) -> None:
        self.events.append("enqueue")

    async def flush(self) -> None:
        self.events.append("flush")


class _FakeRoomAdapter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def aclose(self) -> None:
        self.events.append("room-io.close")


class _FakeBackendError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class _FakeBackend:
    def __init__(self, events: list[str], *, duplicate: bool = False) -> None:
        self.events = events
        self.duplicate = duplicate

    async def get_context(self, _session_id: str) -> Any:
        self.events.append("get-context")
        return SimpleNamespace(
            session=SimpleNamespace(id="voice-session-1"),
            messages=[],
            next_sequence=0,
            voice_key="jacqueline",
        )

    async def mark_started(self, _session_id: str, **_identifiers: Any) -> None:
        self.events.append("mark-started")
        if self.duplicate:
            raise _FakeBackendError(409)

    async def fail_session(self, *_args: Any) -> None:
        self.events.append("fail-session")


def _runtime_config() -> VoiceAgentConfig:
    return VoiceAgentConfig(
        agent_name="realtime-chat-voice",
        bridge_token="bridge-token",
        api_url="https://elixir.example",
        http_max_attempts=4,
        http_retry_base_ms=250,
        http_timeout_ms=5_000,
        tavily_api_key="tavily-key",
        web_search_max_results=5,
        web_search_timeout_ms=6_000,
    )


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str],
    duplicate: bool = False,
) -> tuple[_FakeContext, _FakeSession, _FakePersistence]:
    context = _FakeContext(events)
    session = _FakeSession(events)
    ledger = _FakeLedger(events)
    persistence = _FakePersistence(events)
    backend = _FakeBackend(events, duplicate=duplicate)

    monkeypatch.setattr(worker, "VoiceBackendError", _FakeBackendError)
    monkeypatch.setattr(worker, "VoiceBackendClient", lambda **_kwargs: backend)
    monkeypatch.setattr(
        worker,
        "build_initial_chat_context",
        lambda _messages: SimpleNamespace(
            chat_context="initial-chat-context",
            context_item_ids=frozenset(),
        ),
    )
    monkeypatch.setattr(worker, "TranscriptLedger", lambda *_args: ledger)
    monkeypatch.setattr(worker, "TurnPersistence", lambda **_kwargs: persistence)
    monkeypatch.setattr(worker, "TavilyWebSearchClient", lambda **_kwargs: object())
    monkeypatch.setattr(worker, "create_web_search_tool", lambda _client: "search-tool")
    monkeypatch.setattr(worker, "create_agent_session", lambda _voice: session)
    monkeypatch.setattr(worker, "create_voice_agent", lambda *_args: "voice-agent")
    monkeypatch.setattr(worker, "create_room_options", lambda _identity: "room-options")

    async def fake_prepare_room_runtime(*_args: Any) -> _FakeRoomAdapter:
        events.append("room-runtime.prepare")
        events.append("room-io.start")
        return _FakeRoomAdapter(events)

    async def fake_sync_room_agent_state(*_args: Any) -> None:
        events.append("sync-agent-state")

    monkeypatch.setattr(worker, "prepare_room_runtime", fake_prepare_room_runtime)
    monkeypatch.setattr(worker, "sync_room_agent_state", fake_sync_room_agent_state)
    return context, session, persistence


@pytest.mark.asyncio
async def test_startup_order_recording_privacy_and_no_greeting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context, session, _persistence = _install_runtime_fakes(
        monkeypatch,
        events=events,
    )

    await worker.run_voice_worker(
        cast(Any, context),
        config=_runtime_config(),
        http_client=cast(Any, object()),
    )

    assert events == [
        "get-context",
        "session.start",
        "exclude-context",
        "subscribe",
        "room-runtime.prepare",
        "room-io.start",
        "connect",
        "sync-agent-state",
        "mark-started",
    ]
    assert session.start_kwargs["record"] is False
    assert "room" not in session.start_kwargs
    assert "room_options" not in session.start_kwargs
    assert context.recording_options == {
        "audio": False,
        "transcript": False,
        "traces": False,
        "logs": False,
    }
    assert "generate_reply" not in session.handlers


@pytest.mark.asyncio
async def test_duplicate_job_suppresses_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context, _session, _persistence = _install_runtime_fakes(
        monkeypatch,
        events=events,
        duplicate=True,
    )
    settlements = 0

    async def fake_settle(**_kwargs: Any) -> dict[str, str]:
        nonlocal settlements
        settlements += 1
        return {"status": "finalized"}

    monkeypatch.setattr(worker, "settle_voice_session", fake_settle)

    await worker.run_voice_worker(
        cast(Any, context),
        config=_runtime_config(),
        http_client=cast(Any, object()),
    )
    assert context.shutdown_callbacks
    await context.shutdown_callbacks[0]("duplicate_agent_job")

    assert "shutdown:duplicate_agent_job" in events
    assert "flush" in events
    assert settlements == 0


@pytest.mark.asyncio
async def test_recording_is_disabled_before_metadata_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context = _FakeContext(events)
    context.job.metadata = "not-json"
    client_created = False

    def unexpected_client(**_kwargs: Any) -> Any:
        nonlocal client_created
        client_created = True
        return object()

    monkeypatch.setattr(worker.httpx, "AsyncClient", unexpected_client)

    with pytest.raises(ValueError, match="valid JSON"):
        await worker.run_voice_worker(cast(Any, context), config=_runtime_config())

    assert context.recording_options == {
        "audio": False,
        "transcript": False,
        "traces": False,
        "logs": False,
    }
    assert client_created is False


@pytest.mark.asyncio
async def test_startup_reporting_failure_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context = _FakeContext(events)

    class FailingBackend:
        async def get_context(self, _session_id: str) -> Any:
            raise RuntimeError("context fetch failed")

        async def fail_session(self, *_args: Any) -> None:
            raise RuntimeError("failure reporting failed")

    monkeypatch.setattr(worker, "VoiceBackendClient", lambda **_kwargs: FailingBackend())

    with pytest.raises(RuntimeError, match="context fetch failed"):
        await worker.run_voice_worker(
            cast(Any, context),
            config=_runtime_config(),
            http_client=cast(Any, object()),
        )


@pytest.mark.asyncio
async def test_shutdown_reason_is_authoritative_and_settlement_is_once_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context, session, _persistence = _install_runtime_fakes(
        monkeypatch,
        events=events,
    )
    close_reasons: list[str] = []

    async def fake_settle(**kwargs: Any) -> dict[str, str]:
        close_reasons.append(kwargs["close_reason"])
        return {"status": "finalized"}

    monkeypatch.setattr(worker, "settle_voice_session", fake_settle)
    await worker.run_voice_worker(
        cast(Any, context),
        config=_runtime_config(),
        http_client=cast(Any, object()),
    )

    session.handlers["close"](
        SimpleNamespace(
            error=None,
            reason=SimpleNamespace(value="user_initiated"),
        )
    )
    await context.shutdown_callbacks[0]("room_disconnected")
    await context.shutdown_callbacks[0]("room_disconnected")

    assert close_reasons == ["room_disconnected"]
