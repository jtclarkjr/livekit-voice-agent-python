"""Tests for bounded provider requests and sanitized model-facing results."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import httpx
import pytest
from livekit.agents import llm

from agent.web_search import (
    TavilyWebSearchClient,
    WebSearchResult,
    create_web_search_tool,
)


class FakeSpeechHandle:
    def __init__(self) -> None:
        self.interrupted = False

    async def wait_if_not_interrupted(self, futures: list[asyncio.Future[Any]]) -> None:
        await asyncio.gather(*futures)


class FakeRunContext:
    def __init__(self, speech_handle: FakeSpeechHandle | None = None) -> None:
        self.speech_handle = speech_handle or FakeSpeechHandle()


class StaticSearchClient:
    def __init__(self, results: list[WebSearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    async def search(self, query: str) -> list[WebSearchResult]:
        self.queries.append(query)
        return self.results


def make_client(
    handler: httpx.AsyncBaseTransport | httpx.MockTransport,
    *,
    api_key: str = "tavily-secret",
    max_results: int = 5,
    timeout_ms: float = 6_000,
) -> tuple[TavilyWebSearchClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=handler)
    return (
        TavilyWebSearchClient(
            api_key=api_key,
            http_client=http_client,
            max_results=max_results,
            timeout_ms=timeout_ms,
        ),
        http_client,
    )


@pytest.mark.asyncio
async def test_sends_bounded_tavily_request_with_bearer_token() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": []})

    client, http_client = make_client(httpx.MockTransport(handler), max_results=4)
    async with http_client:
        assert await client.search("  latest hojicha news  ") == []

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.tavily.com/search"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer tavily-secret"
    assert json.loads(request.content) == {
        "query": "latest hojicha news",
        "search_depth": "basic",
        "max_results": 4,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "auto_parameters": False,
    }
    assert "api_key" not in json.loads(request.content)


@pytest.mark.asyncio
async def test_normalizes_filters_deduplicates_truncates_and_preserves_order() -> None:
    payload = {
        "results": [
            {
                "title": "  First   result ",
                "url": "https://example.com/first#section",
                "content": f"  Most   relevant\n{'x' * 600}  ",
            },
            {
                "title": "Duplicate",
                "url": "https://example.com/first",
                "content": "Duplicate content",
            },
            {
                "title": "Unsafe URL",
                "url": "file:///tmp/result",
                "content": "Must be ignored",
            },
            {
                "title": "Missing snippet",
                "url": "https://example.com/missing",
                "content": "   ",
            },
            {
                "title": "Second result",
                "url": "http://example.com/second",
                "content": "Second result content",
            },
            {
                "title": "Past limit",
                "url": "https://example.com/third",
                "content": "Must not be returned",
            },
        ]
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    client, http_client = make_client(transport, max_results=2)
    async with http_client:
        results = await client.search("query")

    assert len(results) == 2
    assert results[0].title == "First result"
    assert results[0].url == "https://example.com/first"
    assert len(results[0].snippet) == 500
    assert results[0].snippet.startswith("Most relevant ")
    assert results[1] == WebSearchResult(
        title="Second result",
        url="http://example.com/second",
        snippet="Second result content",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"detail": "secret response body"}),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"answer": "missing results"}),
    ],
)
async def test_provider_failures_are_sanitized(response: httpx.Response) -> None:
    transport = httpx.MockTransport(lambda _request: response)
    client, http_client = make_client(transport, api_key="key-that-must-not-leak")
    async with http_client:
        with pytest.raises(llm.ToolError) as caught:
            await client.search("query-that-must-not-leak")
    assert "key-that-must-not-leak" not in str(caught.value)
    assert "query-that-must-not-leak" not in str(caught.value)
    assert "secret response body" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", " ", "x" * 501])
async def test_rejects_invalid_query_before_request(query: str) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": []})

    client, http_client = make_client(httpx.MockTransport(handler))
    async with http_client:
        with pytest.raises(llm.ToolError):
            await client.search(query)
    assert calls == 0


@pytest.mark.asyncio
async def test_timeout_becomes_sanitized_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider detail", request=request)

    client, http_client = make_client(httpx.MockTransport(handler), timeout_ms=50)
    async with http_client:
        with pytest.raises(llm.ToolError) as caught:
            await client.search("query")
    assert str(caught.value) == "Web search timed out. Please try again."


@pytest.mark.asyncio
async def test_unexpected_transport_failure_is_sanitized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("provider secret must not leak")

    client, http_client = make_client(httpx.MockTransport(handler))
    async with http_client:
        with pytest.raises(llm.ToolError) as caught:
            await client.search("query must not leak")
    assert str(caught.value) == "Web search is temporarily unavailable. Please try again."


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": "   "},
        {"max_results": 0},
        {"max_results": 11},
        {"timeout_ms": 0},
    ],
)
def test_validates_client_configuration_eagerly(kwargs: dict[str, object]) -> None:
    http_client = httpx.AsyncClient()
    defaults: dict[str, object] = {
        "api_key": "secret",
        "max_results": 5,
        "timeout_ms": 6_000,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        TavilyWebSearchClient(http_client=http_client, **defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_returns_only_sanitized_snippets() -> None:
    client = StaticSearchClient(
        [
            WebSearchResult(
                title="Hidden source",
                url="https://example.com/result",
                snippet=(
                    "Current fact [Publisher](https://example.com/source) [1]. More at "
                    "https://example.com/raw, www.example.org/page, mirror.example.com/path, "
                    "nhk.or.jp/news, example.jp, or example.de. Email editor@example.com. "
                    "[Publisher, 2026] (Publisher, 2026) [source](/news/item) "
                    "[Citation: Archive] [^note] (Source: Example)"
                ),
            )
        ]
    )
    tool = create_web_search_tool(client)
    result = await tool(cast(Any, FakeRunContext()), " current fact ")

    assert result is not None
    assert client.queries == ["current fact"]
    assert result["results"][0]["snippet"].startswith("Current fact Publisher.")
    serialized = json.dumps(result)
    for forbidden in (
        "http://",
        "https://",
        "example.com",
        "example.org",
        "nhk.or.jp",
        "editor@",
        "[1]",
        "Publisher, 2026",
        "Citation:",
        "[^",
        "/news/item",
    ):
        assert forbidden not in serialized
    assert "Hidden source" not in serialized


@pytest.mark.asyncio
async def test_tool_allows_only_one_attempt_per_speech_turn() -> None:
    client = StaticSearchClient([])
    tool = create_web_search_tool(client)
    context = FakeRunContext()

    assert await tool(cast(Any, context), "first query") == {"results": []}
    with pytest.raises(llm.ToolError, match="one attempt per turn"):
        await tool(cast(Any, context), "second query")
    assert await tool(cast(Any, FakeRunContext()), "next turn") == {"results": []}
    assert client.queries == ["first query", "next turn"]


@pytest.mark.asyncio
async def test_tool_sanitizes_unexpected_client_failure() -> None:
    class FailingClient:
        async def search(self, query: str) -> list[WebSearchResult]:
            raise RuntimeError(f"secret leaked through {query}")

    tool = create_web_search_tool(FailingClient())
    with pytest.raises(llm.ToolError) as caught:
        await tool(cast(Any, FakeRunContext()), "current fact")
    assert str(caught.value) == "Web search is temporarily unavailable. Please try again."


@pytest.mark.asyncio
async def test_interruption_cancels_in_flight_search() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingClient:
        async def search(self, query: str) -> list[WebSearchResult]:
            del query
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return []

    class InterruptibleHandle(FakeSpeechHandle):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def wait_if_not_interrupted(self, futures: list[asyncio.Future[Any]]) -> None:
            del futures
            await self.release.wait()

    handle = InterruptibleHandle()
    tool = create_web_search_tool(BlockingClient())
    execution = asyncio.create_task(tool(cast(Any, FakeRunContext(handle)), "current fact"))
    await started.wait()
    handle.interrupted = True
    handle.release.set()

    assert await asyncio.wait_for(execution, timeout=1) is None
    await asyncio.wait_for(cancelled.wait(), timeout=1)


def test_tool_metadata_matches_livekit_contract() -> None:
    tool = create_web_search_tool(StaticSearchClient([]))
    assert tool.id == "searchWeb"
    assert tool.info.name == "searchWeb"
    assert tool.info.on_duplicate == "reject"
