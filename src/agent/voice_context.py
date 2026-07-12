"""Build a bounded, policy-annotated inference context for each voice turn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeGuard, cast

from livekit.agents import llm

from .prompt_loader import load_prompt, render_prompt
from .recency import RecencyDecision, RecencyHistoryMessage, classify_recency

MAX_INFERENCE_CONVERSATION_MESSAGES = 10
WEB_SEARCH_TOOL_NAME = "searchWeb"


@dataclass(frozen=True, slots=True)
class PreparedVoiceTurn:
    """The inference-only context and trusted policy derived for one turn."""

    chat_context: llm.ChatContext
    decision: RecencyDecision
    has_current_turn_search: bool


def _is_conversation_message(item: llm.ChatItem) -> TypeGuard[llm.ChatMessage]:
    return isinstance(item, llm.ChatMessage) and item.role in {"user", "assistant"}


def _is_instruction_message(item: llm.ChatItem) -> TypeGuard[llm.ChatMessage]:
    return isinstance(item, llm.ChatMessage) and item.role in {"system", "developer"}


def _is_tool_artifact(
    item: llm.ChatItem,
) -> TypeGuard[llm.FunctionCall | llm.FunctionCallOutput]:
    return isinstance(item, (llm.FunctionCall, llm.FunctionCallOutput))


def _escape_untrusted_json(value: str) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def _search_instruction(decision: RecencyDecision, has_current_turn_search: bool) -> str:
    if has_current_turn_search:
        return load_prompt("search_already_completed.md")
    if decision.mode == "forbidden":
        return load_prompt("search_forbidden.md")
    if decision.mode == "required":
        instruction = load_prompt("search_required.md")
        if decision.query_hint is not None:
            instruction = f"{instruction}\n{load_prompt('search_required_hint.md')}"
        return instruction
    return load_prompt("search_auto.md")


def _query_hint_message(
    decision: RecencyDecision, latest_user: llm.ChatMessage | None
) -> llm.ChatMessage | None:
    if decision.query_hint is None or latest_user is None:
        return None
    return llm.ChatMessage(
        id="voice_search_query_hint",
        role="assistant",
        content=[
            render_prompt(
                "search_query_hint.md",
                query_hint_json=_escape_untrusted_json(decision.query_hint),
            )
        ],
        created_at=max(0.0, latest_user.created_at - 0.000001),
        extra={"untrustedVoiceQueryHint": True},
    )


def _utc_timestamp(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _runtime_note(
    now: datetime,
    decision: RecencyDecision,
    has_current_turn_search: bool,
) -> llm.ChatMessage:
    aware_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return llm.ChatMessage(
        id="voice_runtime_policy",
        role="system",
        content=[
            render_prompt(
                "runtime_policy.md",
                current_utc=_utc_timestamp(aware_now),
                search_instruction=_search_instruction(
                    decision,
                    has_current_turn_search,
                ),
            )
        ],
        created_at=aware_now.timestamp(),
        extra={"trustedVoiceRuntimePolicy": True},
    )


def prepare_voice_turn(
    canonical_context: llm.ChatContext,
    now: datetime | None = None,
) -> PreparedVoiceTurn:
    """Create an inference context without mutating canonical session history."""

    current_time = now or datetime.now(UTC)
    items = canonical_context.items
    latest_user_index = -1
    latest_user: llm.ChatMessage | None = None
    for index in range(len(items) - 1, -1, -1):
        candidate = items[index]
        if isinstance(candidate, llm.ChatMessage) and candidate.role == "user":
            latest_user_index = index
            latest_user = candidate
            break
    previous_messages = [
        RecencyHistoryMessage(
            role=cast(Literal["user", "assistant"], item.role),
            content=item.text_content or "",
        )
        for item in items[:latest_user_index]
        if _is_conversation_message(item)
    ]
    decision = classify_recency(
        latest_user.text_content or "" if latest_user is not None else "",
        previous_messages,
    )

    conversation_indices = [
        index for index, item in enumerate(items) if _is_conversation_message(item)
    ][-MAX_INFERENCE_CONVERSATION_MESSAGES:]
    retained_conversation_indices = set(conversation_indices)
    has_current_turn_search = latest_user_index >= 0 and any(
        index > latest_user_index and _is_tool_artifact(item) and item.name == WEB_SEARCH_TOOL_NAME
        for index, item in enumerate(items)
    )

    instruction_messages = [item for item in items if _is_instruction_message(item)]
    retained_non_instructions = [
        item
        for index, item in enumerate(items)
        if not _is_instruction_message(item)
        and (
            index in retained_conversation_indices
            or (latest_user_index >= 0 and index > latest_user_index and _is_tool_artifact(item))
        )
    ]
    query_hint = None if has_current_turn_search else _query_hint_message(decision, latest_user)

    if query_hint is not None and latest_user is not None:
        if (
            sum(_is_conversation_message(item) for item in retained_non_instructions)
            >= MAX_INFERENCE_CONVERSATION_MESSAGES
        ):
            oldest_index = next(
                (
                    index
                    for index, item in enumerate(retained_non_instructions)
                    if _is_conversation_message(item)
                ),
                -1,
            )
            if oldest_index >= 0:
                retained_non_instructions.pop(oldest_index)
        latest_user_prompt_index = next(
            (index for index, item in enumerate(retained_non_instructions) if item is latest_user),
            -1,
        )
        if latest_user_prompt_index >= 0:
            retained_non_instructions.insert(latest_user_prompt_index, query_hint)

    return PreparedVoiceTurn(
        chat_context=llm.ChatContext(
            [
                *instruction_messages,
                _runtime_note(current_time, decision, has_current_turn_search),
                *retained_non_instructions,
            ]
        ),
        decision=decision,
        has_current_turn_search=has_current_turn_search,
    )
