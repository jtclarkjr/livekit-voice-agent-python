"""Credential-gated LiveKit Inference evaluations.

These tests intentionally stay outside the deterministic ``poe check`` path.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from livekit.agents import AgentSession, inference

from agent.transcript import build_initial_chat_context
from agent.voice_agent import create_voice_agent
from agent.web_search import (
    WebSearchResult,
    create_web_search_tool,
)

load_dotenv(".env.local")

pytestmark = [
    pytest.mark.livekit_eval,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVEKIT_EVALS") != "true",
        reason="set RUN_LIVEKIT_EVALS=true to use LiveKit Inference credentials",
    ),
]


class _EvaluationSearchClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str) -> list[WebSearchResult]:
        self.queries.append(query)
        return [
            WebSearchResult(
                title="Aurora Tea Bulletin",
                url="https://bulletin.example/aurora-tea-festival",
                snippet=(
                    "The latest Aurora Tea Festival update says the event opens "
                    "in Sapporo on October 4, 2026."
                ),
            )
        ]


def _initial_context():
    return build_initial_chat_context(
        [
            {
                "role": "user",
                "content": "My favorite tea is hojicha.",
                "created_at": "2026-07-11T00:00:00.000Z",
            }
        ]
    ).chat_context


@pytest.mark.asyncio
async def test_uses_prior_context_and_stays_speech_friendly() -> None:
    search = _EvaluationSearchClient()
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge,
        AgentSession() as session,
    ):
        await session.start(create_voice_agent(_initial_context(), create_web_search_tool(search)))

        context_result = await session.run(user_input="Which tea did I say was my favorite?")
        await (
            context_result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent="Correctly says that the user named hojicha as their favorite tea.",
            )
        )
        context_result.expect.no_more_events()

        followup_result = await session.run(user_input="Answer again in one short spoken sentence.")
        await (
            followup_result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=(
                    "Answers in one concise, natural spoken sentence without markdown or a list."
                ),
            )
        )
        followup_result.expect.no_more_events()


@pytest.mark.asyncio
async def test_searches_current_information_without_exposing_sources() -> None:
    search = _EvaluationSearchClient()
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge,
        AgentSession() as session,
    ):
        await session.start(create_voice_agent(_initial_context(), create_web_search_tool(search)))
        result = await session.run(
            user_input=(
                "Search the web for today's latest Aurora Tea Festival update. "
                "Answer in one short spoken sentence."
            )
        )

        result.expect.contains_function_call(name="searchWeb")
        result.expect.contains_function_call_output(is_error=False)
        assistant_text = " ".join(
            event.item.text_content or ""
            for event in result.events
            if event.type == "message" and event.item.role == "assistant"
        )
        assert len(search.queries) == 1
        assert "sapporo" in assistant_text.lower()
        assert "https://" not in assistant_text
        assert "bulletin.example" not in assistant_text
        assert "tavily" not in assistant_text.lower()
        assert len(assistant_text.split()) <= 25
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Uses the mocked fact that the Aurora Tea Festival opens in Sapporo "
                "on October 4, 2026, answers concisely, and does not mention a "
                "source, citation, publisher, or URL."
            ),
        )


@pytest.mark.asyncio
async def test_honors_explicit_no_search_request() -> None:
    search = _EvaluationSearchClient()
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge,
        AgentSession() as session,
    ):
        await session.start(create_voice_agent(_initial_context(), create_web_search_tool(search)))
        result = await session.run(
            user_input=(
                "Do not search the web. Based only on our conversation, which tea "
                "did I say was my favorite?"
            )
        )

        assert search.queries == []
        assert not any(event.type == "function_call" for event in result.events)
        await result.expect.contains_message(role="assistant").judge(
            judge,
            intent=(
                "Answers that the user named hojicha as their favorite tea and "
                "does not claim to have searched."
            ),
        )
