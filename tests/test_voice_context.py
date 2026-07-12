"""Tests for bounded, injection-resistant inference context preparation."""

from __future__ import annotations

from datetime import UTC, datetime

from livekit.agents import llm

from agent.voice_context import (
    MAX_INFERENCE_CONVERSATION_MESSAGES,
    prepare_voice_turn,
)


def message(identifier: str, role: llm.ChatRole, content: str) -> llm.ChatMessage:
    return llm.ChatMessage(id=identifier, role=role, content=[content])


def runtime_text(context: llm.ChatContext) -> str:
    note = next(
        item
        for item in context.items
        if isinstance(item, llm.ChatMessage) and item.extra.get("trustedVoiceRuntimePolicy") is True
    )
    return note.text_content or ""


def query_hint_text(context: llm.ChatContext) -> str | None:
    hint = next(
        (
            item
            for item in context.items
            if isinstance(item, llm.ChatMessage)
            and item.extra.get("untrustedVoiceQueryHint") is True
        ),
        None,
    )
    return hint.text_content if hint is not None else None


def test_retains_all_instructions_and_latest_ten_conversation_messages() -> None:
    conversation = [
        message(
            f"conversation_{index}",
            "user" if index % 2 == 0 else "assistant",
            f"message {index}",
        )
        for index in range(13)
    ]
    canonical = llm.ChatContext(
        [
            message("system_1", "system", "Primary instructions"),
            *conversation,
            message("system_2", "system", "Additional instructions"),
        ]
    )
    prepared = prepare_voice_turn(canonical)
    retained_ids = [item.id for item in prepared.chat_context.items]
    conversation_items = [
        item
        for item in prepared.chat_context.items
        if isinstance(item, llm.ChatMessage) and item.role in {"user", "assistant"}
    ]

    assert {"system_1", "system_2"}.issubset(retained_ids)
    assert len(conversation_items) == MAX_INFERENCE_CONVERSATION_MESSAGES
    assert "conversation_2" not in retained_ids
    assert {"conversation_3", "conversation_12"}.issubset(retained_ids)


def test_prunes_old_tool_artifacts_and_retains_current_turn_artifacts() -> None:
    canonical = llm.ChatContext(
        [
            message("old_user", "user", "Old question"),
            llm.FunctionCall(
                id="old_call", call_id="old_call_id", name="searchWeb", arguments="{}"
            ),
            llm.FunctionCallOutput(
                id="old_output",
                call_id="old_call_id",
                name="searchWeb",
                output="stale",
                is_error=False,
            ),
            message("old_assistant", "assistant", "Old answer"),
            message("current_user", "user", "What is the latest release?"),
            llm.FunctionCall(
                id="current_call",
                call_id="current_call_id",
                name="searchWeb",
                arguments='{"query":"latest release"}',
            ),
            llm.FunctionCallOutput(
                id="current_output",
                call_id="current_call_id",
                name="searchWeb",
                output="current result",
                is_error=False,
            ),
        ]
    )
    prepared = prepare_voice_turn(canonical)
    ids = [item.id for item in prepared.chat_context.items]

    assert "old_call" not in ids
    assert "old_output" not in ids
    assert {"current_call", "current_output"}.issubset(ids)
    assert prepared.has_current_turn_search is True
    assert "SEARCH ALREADY COMPLETED" in runtime_text(prepared.chat_context)
    assert "Do not call searchWeb again" in runtime_text(prepared.chat_context)


def test_does_not_mutate_canonical_context() -> None:
    canonical = llm.ChatContext(
        [
            message("system", "system", "Stay concise"),
            message("user", "user", "Explain recursion"),
        ]
    )
    original_items = canonical.items
    original_dict = canonical.to_dict(exclude_timestamp=False)

    prepared = prepare_voice_turn(canonical)
    prepared.chat_context.items.append(message("inference_only", "assistant", "Temporary"))

    assert prepared.chat_context is not canonical
    assert canonical.items is original_items
    assert canonical.to_dict(exclude_timestamp=False) == original_dict
    assert canonical.get_by_id("voice_runtime_policy") is None
    assert canonical.get_by_id("voice_search_query_hint") is None
    assert canonical.get_by_id("inference_only") is None


def test_adds_exact_utc_timestamp_and_auto_safety_policy() -> None:
    now = datetime(2026, 7, 12, 3, 4, 5, 678000, tzinfo=UTC)
    prepared = prepare_voice_turn(
        llm.ChatContext([message("user", "user", "Explain how a mutex works")]), now
    )
    note = runtime_text(prepared.chat_context)

    assert prepared.decision.mode == "auto"
    assert "2026-07-12T03:04:05.678Z" in note
    assert "Per-turn search instruction: AUTO" in note
    assert "untrusted data" in note
    assert "Do not quote, cite, or speak source URLs" in note


def test_writes_required_and_forbidden_runtime_instructions() -> None:
    required = prepare_voice_turn(
        llm.ChatContext([message("user", "user", "Search the web for the latest OpenAI release")])
    )
    forbidden = prepare_voice_turn(
        llm.ChatContext(
            [
                message(
                    "user",
                    "user",
                    "Do not search the web. Explain TCP from what you know.",
                )
            ]
        )
    )
    assert "Per-turn search instruction: REQUIRED" in runtime_text(required.chat_context)
    assert "Per-turn search instruction: FORBIDDEN" in runtime_text(forbidden.chat_context)


def test_inherited_hint_is_escaped_lower_priority_data_before_user() -> None:
    canonical = llm.ChatContext(
        [
            message("prior_user", "user", "What is the latest GPT model?"),
            message("prior_assistant", "assistant", "I can check that."),
            message("current_user", "user", "gpt?"),
        ]
    )
    prepared = prepare_voice_turn(canonical)
    note = runtime_text(prepared.chat_context)
    hint = query_hint_text(prepared.chat_context)

    assert prepared.decision.query_hint == "latest gpt"
    assert "lower-priority assistant data message" in note
    assert '"latest gpt"' not in note
    assert hint is not None and '"latest gpt"' in hint
    hint_index = prepared.chat_context.index_by_id("voice_search_query_hint")
    user_index = prepared.chat_context.index_by_id("current_user")
    assert hint_index is not None
    assert user_index is not None
    assert hint_index < user_index


def test_hint_stays_inside_ten_message_inference_budget() -> None:
    canonical = llm.ChatContext(
        [
            message("oldest_user", "user", "What is the latest GPT model?"),
            message("assistant_1", "assistant", "First response"),
            message("assistant_2", "assistant", "Second response"),
            message("neutral_user_1", "user", "Compare their context windows"),
            message("assistant_3", "assistant", "Third response"),
            message("assistant_4", "assistant", "Fourth response"),
            message("neutral_user_2", "user", "Keep it concise"),
            message("assistant_5", "assistant", "Fifth response"),
            message("assistant_6", "assistant", "Sixth response"),
            message("current_user", "user", "Claude?"),
        ]
    )
    prepared = prepare_voice_turn(canonical)
    conversation = [
        item
        for item in prepared.chat_context.items
        if isinstance(item, llm.ChatMessage) and item.role in {"user", "assistant"}
    ]
    assert prepared.decision.query_hint == "latest claude"
    assert len(conversation) == MAX_INFERENCE_CONVERSATION_MESSAGES
    assert prepared.chat_context.get_by_id("oldest_user") is None
    assert "latest claude" in (query_hint_text(prepared.chat_context) or "")


def test_withholds_prompt_shaped_hint_from_model_context() -> None:
    prepared = prepare_voice_turn(
        llm.ChatContext(
            [
                message("prior_user", "user", "What is the latest GPT model?"),
                message("prior_assistant", "assistant", "I can check that."),
                message("current_user", "user", '<unsafe>& "secret"?'),
            ]
        )
    )
    assert prepared.decision.query_hint is None
    assert query_hint_text(prepared.chat_context) is None
    assert "<unsafe>" not in runtime_text(prepared.chat_context)


def test_no_user_message_uses_auto_mode_without_retaining_assistant() -> None:
    prepared = prepare_voice_turn(
        llm.ChatContext(
            [
                message("system", "system", "Stay concise"),
                message("assistant", "assistant", "Ready when you are."),
            ]
        ),
        datetime(2026, 7, 12, tzinfo=UTC),
    )
    assert prepared.decision.mode == "auto"
    assert prepared.has_current_turn_search is False
    assert "Per-turn search instruction: AUTO" in runtime_text(prepared.chat_context)
    assert prepared.chat_context.get_by_id("assistant") is not None
