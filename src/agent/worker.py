"""LiveKit worker lifecycle and runtime wiring."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx
from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentServer,
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    RecordingOptions,
    TurnHandlingOptions,
    inference,
    room_io,
)
from livekit.agents.voice.remote_session import RoomSessionTransport, SessionHost
from livekit.plugins import ai_coustics

from .backend import VoiceBackendClient, VoiceBackendError
from .config import VoiceAgentConfig, load_config, normalize_log_level
from .constants import (
    AGENT_NAME,
    LLM_MODEL,
    STT_LANGUAGE,
    STT_MODEL,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
)
from .contracts import DispatchMetadata
from .lifecycle import failure_reason, settle_voice_session
from .rejoin import ParticipantRejoinGuard
from .transcript import (
    TranscriptLedger,
    TurnPersistence,
    build_initial_chat_context,
)
from .voice_agent import create_voice_agent
from .voices import VoiceKey, resolve_cartesia_voice_id
from .web_search import TavilyWebSearchClient, create_web_search_tool

if TYPE_CHECKING:
    from livekit.agents.llm import ChatItem

logger = logging.getLogger("realtime-chat-voice")

PARTICIPANT_REJOIN_GRACE_SECONDS = 10.0
LIVEKIT_AGENT_STATE_ATTRIBUTE = "lk.agent.state"

_LOG_LEVEL_ALIASES = {"fatal": "critical", "warning": "warn"}
_LIVEKIT_LOG_LEVELS = {
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "critical",
}


def configure_livekit_log_level(
    environment: MutableMapping[str, str] = os.environ,
) -> str:
    """Set LiveKit's log level while preserving the legacy LOG_LEVEL contract."""

    configured = environment.get("LOG_LEVEL", "fatal").strip().lower()
    normalized = _LOG_LEVEL_ALIASES.get(configured, normalize_log_level(configured))
    if normalized not in _LIVEKIT_LOG_LEVELS:
        allowed = ", ".join(sorted(_LIVEKIT_LOG_LEVELS | set(_LOG_LEVEL_ALIASES)))
        raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
    environment.setdefault("LIVEKIT_LOG_LEVEL", normalized)
    return normalized


def parse_dispatch_metadata(raw_metadata: str) -> DispatchMetadata:
    """Parse and strictly validate explicit LiveKit dispatch metadata."""

    try:
        return DispatchMetadata.model_validate_json(raw_metadata)
    except ValueError as error:
        raise ValueError("LiveKit dispatch metadata must be valid JSON") from error


def create_agent_session(voice_key: VoiceKey) -> AgentSession:
    """Create the fixed STT/TTS/turn-handling pipeline for a selected voice."""

    return AgentSession(
        max_tool_steps=1,
        stt=inference.STT(model=STT_MODEL, language=STT_LANGUAGE),
        tts=inference.TTS(
            model=TTS_MODEL,
            voice=resolve_cartesia_voice_id(voice_key),
            sample_rate=TTS_SAMPLE_RATE,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
    )


def create_room_options(participant_identity: str | None) -> room_io.RoomOptions:
    """Create room I/O options with explicit privacy and reconnect behavior."""

    common: dict[str, Any] = {
        "audio_input": room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_S,
            ),
        ),
        "audio_output": room_io.AudioOutputOptions(
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=1,
        ),
        "close_on_disconnect": False,
    }
    if participant_identity is not None:
        common["participant_identity"] = participant_identity
    return room_io.RoomOptions(**common)


def _participant_identity(ctx: JobContext) -> str | None:
    participant = ctx.job.participant
    if participant is None:
        return None
    return participant.identity or None


async def _room_id(ctx: JobContext) -> str | None:
    job_room = ctx.job.room
    if job_room is not None and job_room.sid:
        return job_room.sid
    return await ctx.room.sid or None


def _is_non_agent(participant: rtc.RemoteParticipant) -> bool:
    return participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


def _session_history(
    ctx: JobContext,
    session: AgentSession,
) -> list[ChatItem]:
    try:
        return list(ctx.make_session_report(session).chat_history.items)
    except Exception:
        return list(session.history.items)


@dataclass(slots=True)
class RoomRuntime:
    """Room media plus the client-facing ``lk.agent.session`` transport."""

    adapter: room_io.RoomIO
    session_host: SessionHost

    async def aclose(self) -> None:
        try:
            await self.session_host.aclose()
        finally:
            await self.adapter.aclose()


async def prepare_room_runtime(
    session: AgentSession,
    room: rtc.Room,
    options: room_io.RoomOptions,
) -> RoomRuntime:
    """Register text/session transports before the room connects."""

    # RoomIO's legacy transcription output reads the local participant while it
    # starts. Hide the explicit link until all pre-connect handlers exist, then
    # restore it before the room can emit participant events.
    linked_identity = (
        options.participant_identity if isinstance(options.participant_identity, str) else None
    )
    adapter_options = copy.copy(options)
    adapter_options.participant_identity = NOT_GIVEN
    adapter = room_io.RoomIO(agent_session=session, room=room, options=adapter_options)
    session_host = SessionHost(RoomSessionTransport(room))
    try:
        await adapter.start()
        if linked_identity is not None:
            cast(Any, adapter)._participant_identity = linked_identity
        text_input_options = options.get_text_input_options()
        if text_input_options is not None:
            adapter.register_text_input(text_input_options.text_input_cb)
        session_host.register_session(session)
        await session_host.start()
    except Exception:
        try:
            await session_host.aclose()
        except Exception:
            logger.exception("failed to clean up partial LiveKit session transport")
        try:
            await adapter.aclose()
        except Exception:
            logger.exception("failed to clean up partial LiveKit room I/O")
        raise
    return RoomRuntime(adapter=adapter, session_host=session_host)


async def sync_room_agent_state(session: AgentSession, room: rtc.Room) -> None:
    """Best-effort replay of the state transition emitted before room attachment."""

    try:
        await room.local_participant.set_attributes(
            {LIVEKIT_AGENT_STATE_ATTRIBUTE: session.agent_state}
        )
    except Exception:
        logger.warning("failed to synchronize initial LiveKit agent state", exc_info=True)


@dataclass(slots=True)
class _RuntimeState:
    close_reason: str = "job_shutdown"
    fatal_reason: str | None = None
    settlement_task: asyncio.Task[Any] | None = None
    shutdown_requested: bool = False
    skip_settlement: bool = False


async def run_voice_worker(
    ctx: JobContext,
    *,
    config: VoiceAgentConfig | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Run one dispatched voice job."""

    ctx.init_recording(RecordingOptions(audio=False, transcript=False, traces=False, logs=False))
    dispatch = parse_dispatch_metadata(ctx.job.metadata)
    session_id = dispatch.voice_session_id
    resolved_config = config or load_config()
    owns_http_client = http_client is None
    client = http_client or httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    backend = VoiceBackendClient(
        base_url=resolved_config.api_url,
        bridge_token=resolved_config.bridge_token,
        http_client=client,
        max_attempts=resolved_config.http_max_attempts,
        retry_base_ms=resolved_config.http_retry_base_ms,
        timeout_ms=resolved_config.http_timeout_ms,
    )

    state = _RuntimeState()
    shutdown_registered = False

    try:
        context = await backend.get_context(session_id)
        if context.session.id != session_id:
            raise ValueError("Voice context response did not match the dispatched session")

        initial = build_initial_chat_context(context.messages)
        ledger = TranscriptLedger(initial.context_item_ids, context.next_sequence)
        session = create_agent_session(context.voice_key)
        persistence = TurnPersistence(backend=backend, ledger=ledger, session_id=session_id)
        web_search_client = TavilyWebSearchClient(
            api_key=resolved_config.tavily_api_key,
            http_client=client,
            max_results=resolved_config.web_search_max_results,
            timeout_ms=resolved_config.web_search_timeout_ms,
        )
        web_search_tool = create_web_search_tool(web_search_client)
        room_runtime: RoomRuntime | None = None

        def request_shutdown(reason: str) -> None:
            if state.shutdown_requested:
                return
            state.shutdown_requested = True
            state.close_reason = reason
            ctx.shutdown(reason=reason)

        async def session_is_open() -> bool:
            try:
                await backend.get_context(session_id)
                return True
            except VoiceBackendError as error:
                if error.status in {404, 409}:
                    return False
                raise

        participant_identity = _participant_identity(ctx)
        rejoin_guard = ParticipantRejoinGuard(
            grace_seconds=PARTICIPANT_REJOIN_GRACE_SECONDS,
            participant_identity=participant_identity,
            session_is_open=session_is_open,
            shutdown=request_shutdown,
        )

        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            if _is_non_agent(participant):
                rejoin_guard.participant_connected(participant.identity)

        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            if _is_non_agent(participant):
                rejoin_guard.participant_disconnected(participant.identity)

        def on_session_error(event: ErrorEvent) -> None:
            if event.error.recoverable:
                return
            state.fatal_reason = failure_reason(
                "unrecoverable LiveKit model error",
                event.error,
            )
            request_shutdown("unrecoverable_model_error")

        def on_session_close(event: CloseEvent) -> None:
            if event.error is not None:
                state.fatal_reason = failure_reason(
                    "LiveKit session closed with an error",
                    event.error,
                )
            request_shutdown(event.reason.value)

        async def on_shutdown(shutdown_reason: str) -> None:
            state.shutdown_requested = True
            state.close_reason = shutdown_reason or state.close_reason
            ctx.room.off("participant_connected", on_participant_connected)
            ctx.room.off("participant_disconnected", on_participant_disconnected)
            rejoin_guard.dispose()
            try:
                if room_runtime is not None:
                    try:
                        await room_runtime.aclose()
                    except Exception:
                        logger.exception(
                            "failed to close LiveKit room I/O",
                            extra={"session_id": session_id},
                        )
                if state.skip_settlement:
                    logger.warning(
                        "duplicate voice agent job exited without session settlement",
                        extra={"session_id": session_id},
                    )
                    await persistence.flush()
                    return
                if state.settlement_task is None:
                    state.settlement_task = asyncio.create_task(
                        settle_voice_session(
                            backend=backend,
                            close_reason=state.close_reason,
                            history=_session_history(ctx, session),
                            ledger=ledger,
                            persistence=persistence,
                            session_id=session_id,
                            fatal_reason=state.fatal_reason,
                        ),
                    )
                result = await state.settlement_task
                logger.info(
                    "voice session bridge settlement completed",
                    extra={
                        "result": result,
                        "session_id": session_id,
                        "live_turn_persistence_failures": len(persistence.failures),
                    },
                )
            finally:
                if owns_http_client:
                    await client.aclose()

        ctx.room.on("participant_connected", on_participant_connected)
        ctx.room.on("participant_disconnected", on_participant_disconnected)
        session.on("error", on_session_error)
        session.on("close", on_session_close)
        ctx.add_shutdown_callback(on_shutdown)
        shutdown_registered = True

        await session.start(
            agent=create_voice_agent(initial.chat_context, web_search_tool),
            record=False,
        )

        ledger.exclude_context_items(session.history.items)

        def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
            persistence.enqueue(event.item)

        session.on("conversation_item_added", on_conversation_item_added)

        room_runtime = await prepare_room_runtime(
            session,
            ctx.room,
            create_room_options(participant_identity),
        )
        await ctx.connect()
        await sync_room_agent_state(session, ctx.room)
        for participant in ctx.room.remote_participants.values():
            on_participant_connected(participant)

        try:
            await backend.mark_started(
                session_id,
                job_id=ctx.job.id,
                room_id=await _room_id(ctx),
            )
        except VoiceBackendError as error:
            if error.status == 409:
                state.skip_settlement = True
                request_shutdown("duplicate_agent_job")
                return
            raise

        logger.info(
            "voice agent is listening",
            extra={
                "session_id": session_id,
                "llm_model": LLM_MODEL,
                "context_messages": len(context.messages),
                "voice_key": context.voice_key,
                "tts_model": TTS_MODEL,
                "tts_sample_rate": TTS_SAMPLE_RATE,
            },
        )
        # Deliberately do not generate a reply. The user always starts the call.
    except Exception as error:
        reason = failure_reason("voice agent startup failed", error)
        state.fatal_reason = reason
        logger.exception(
            "voice agent entrypoint failed",
            extra={"session_id": session_id},
        )
        if not shutdown_registered:
            try:
                await backend.fail_session(session_id, [], reason)
            except Exception:
                logger.exception(
                    "failed to report voice startup failure",
                    extra={"session_id": session_id},
                )
            finally:
                if owns_http_client:
                    await client.aclose()
        raise


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def voice_entrypoint(ctx: JobContext) -> None:
    """LiveKit AgentServer entrypoint."""

    configure_livekit_log_level()
    await run_voice_worker(ctx)
