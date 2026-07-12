"""LiveKit agent that enforces the trusted per-turn search policy."""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from livekit.agents import (
    Agent,
    FlushSentinel,
    ModelSettings,
    NotGivenOr,
    inference,
    llm,
)

from .constants import LLM_MODEL
from .prompt_loader import load_prompt
from .recency import RecencyDecision
from .voice_context import prepare_voice_turn

WEB_SEARCH_TOOL_NAME = "searchWeb"

type LlmNodeOutput = (
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str | FlushSentinel]]
    | Coroutine[Any, Any, str]
    | Coroutine[Any, Any, llm.ChatChunk]
    | Coroutine[Any, Any, None]
)


def resolve_voice_tool_choice(
    *,
    decision: RecencyDecision,
    has_current_turn_search: bool,
    requested_tool_choice: NotGivenOr[llm.ToolChoice],
    tools: Sequence[llm.Tool],
) -> NotGivenOr[llm.ToolChoice]:
    """Resolve SDK, opt-out, and required-search signals in priority order."""

    if requested_tool_choice == "none" or has_current_turn_search:
        return "none"
    if decision.mode == "forbidden":
        return "none"
    if decision.mode == "required" and any(tool.id == WEB_SEARCH_TOOL_NAME for tool in tools):
        return cast(
            llm.ToolChoice,
            {"type": "function", "function": {"name": WEB_SEARCH_TOOL_NAME}},
        )
    return requested_tool_choice


class VoiceFastAgent(Agent):
    """Inference agent with bounded history and deterministic search enforcement."""

    def __init__(
        self,
        chat_context: llm.ChatContext,
        web_search_tool: llm.Tool,
        *,
        clock: Callable[[], datetime] | None = None,
        language_model: llm.LLM | None = None,
    ) -> None:
        super().__init__(
            chat_ctx=chat_context,
            id="default_agent",
            instructions=load_prompt("voice_instructions.md"),
            llm=language_model
            or inference.LLM(
                model=LLM_MODEL,
                extra_kwargs={"parallel_tool_calls": False},
            ),
            tools=[web_search_tool],
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> LlmNodeOutput:
        """Rewrite inference context and propagate the effective tool choice."""

        prepared = prepare_voice_turn(chat_ctx, self._clock())
        model_settings.tool_choice = resolve_voice_tool_choice(
            decision=prepared.decision,
            has_current_turn_search=prepared.has_current_turn_search,
            requested_tool_choice=model_settings.tool_choice,
            tools=tools,
        )
        return super().llm_node(prepared.chat_context, tools, model_settings)


def create_voice_agent(
    chat_context: llm.ChatContext,
    web_search_tool: llm.Tool,
) -> VoiceFastAgent:
    """Create the worker's single policy-enforcing voice agent."""

    return VoiceFastAgent(chat_context, web_search_tool)
