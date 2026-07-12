"""Bounded Tavily search and the model-facing sanitized search tool."""

from __future__ import annotations

import asyncio
import math
import re
import weakref
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, cast

import httpx
from livekit.agents import RunContext, function_tool, llm
from pydantic import Field

from .prompt_loader import load_prompt

TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
MAX_QUERY_LENGTH = 500
MAX_SNIPPET_LENGTH = 500
PROVIDER_ERROR_MESSAGE = "Web search is temporarily unavailable. Please try again."
INVALID_RESPONSE_MESSAGE = "Web search returned an invalid response. Please try again."
TIMEOUT_ERROR_MESSAGE = "Web search timed out. Please try again."
QUERY_ERROR_MESSAGE = "Web search queries must contain between 1 and 500 characters."
DUPLICATE_SEARCH_MESSAGE = "Web search is limited to one attempt per turn."

SearchQuery = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description=load_prompt("search_query_description.md"),
    ),
]


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """A validated provider result; source metadata never reaches the model."""

    title: str
    url: str
    snippet: str


class WebSearchClient(Protocol):
    """Minimal asynchronous search interface used by the function tool."""

    async def search(self, query: str) -> Sequence[WebSearchResult]: ...


type WebSearchToolResult = dict[str, list[dict[str, str]]] | None
type WebSearchTool = llm.FunctionTool[..., Coroutine[Any, Any, WebSearchToolResult]]


def _normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _normalize_http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL, ValueError:
        return None
    if url.scheme not in {"http", "https"} or not url.host:
        return None
    return str(url.copy_with(fragment=None))


def _normalize_results(payload: object, max_results: int) -> list[WebSearchResult]:
    if not isinstance(payload, dict):
        raise llm.ToolError(INVALID_RESPONSE_MESSAGE)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise llm.ToolError(INVALID_RESPONSE_MESSAGE)

    results: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for candidate in cast(list[object], raw_results):
        if not isinstance(candidate, dict):
            continue
        title = _normalize_text(candidate.get("title"))
        url = _normalize_http_url(candidate.get("url"))
        content = _normalize_text(candidate.get("content"))
        if title is None or url is None or content is None or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(WebSearchResult(title=title, url=url, snippet=content[:MAX_SNIPPET_LENGTH]))
        if len(results) == max_results:
            break
    return results


def _parse_query(query: str) -> str:
    normalized = query.strip()
    if normalized == "" or len(normalized) > MAX_QUERY_LENGTH:
        raise llm.ToolError(QUERY_ERROR_MESSAGE)
    return normalized


def _url_replacement(match: re.Match[str]) -> str:
    return "".join(re.findall(r"[),.;:!?]+$", match.group(0)))


def _sanitize_snippet(value: object) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None

    sanitized = re.sub(
        r"\[([^\]]+)\]\((?:(?:https?|ftp)://|www\.|//|\.{0,2}/|#|mailto:)[^)]+\)",
        r"\1",
        normalized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"<(?:(?:https?|ftp)://|www\.|//|mailto:)[^>]+>",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"\b(?:(?:https?|ftp)://|www\.)[^\s<>\"']+",
        _url_replacement,
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[^\W\d_]{2,63}\b", "", sanitized, flags=re.UNICODE)
    sanitized = re.sub(
        r"\b(?:[\w](?:[\w-]{0,62})\.)+(?:[^\W\d_]{2,63}|xn--[a-z0-9-]+)(?::\d+)?(?:/[^\s<>\"']*)?",
        "",
        sanitized,
        flags=re.IGNORECASE | re.UNICODE,
    )
    sanitized = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s<>\"']*)?", "", sanitized)
    sanitized = re.sub(r"\[(?=[^\]]*\b(?:19|20)\d{2}\b)[^\]]+\]", "", sanitized)
    sanitized = re.sub(
        r"\[(?:source|sources|citation|reference|ref):?[^\]]*\]",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\[\^[^\]]+\]", "", sanitized)
    sanitized = re.sub(r"\[(?:\d+(?:\s*[-,]\s*\d+)*)\]", "", sanitized)
    sanitized = re.sub(
        r"\([^\W\d_][\w .&'-]{0,80},\s*(?:19|20)\d{2}[a-z]?\)",
        "",
        sanitized,
        flags=re.IGNORECASE | re.UNICODE,
    )
    sanitized = re.sub(
        r"\((?:source|sources|citation):[^)]*\)",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = sanitized.replace("<>", "")
    sanitized = re.sub(r"\s+([,.;:!?])", r"\1", sanitized)
    return _normalize_text(sanitized)


class TavilyWebSearchClient:
    """Tavily adapter using a job-scoped ``httpx.AsyncClient``."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        api_key: str,
        max_results: int,
        timeout_ms: float,
    ) -> None:
        normalized_key = api_key.strip()
        if normalized_key == "":
            raise ValueError("Tavily API key must not be empty")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 10
        ):
            raise ValueError("Tavily max results must be an integer from 1 to 10")
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("Tavily timeout must be greater than zero")
        self._http_client = http_client
        self._api_key = normalized_key
        self._max_results = max_results
        self._timeout_seconds = timeout_ms / 1_000

    async def search(self, query: str) -> list[WebSearchResult]:
        """Execute one bounded provider request and validate its public metadata."""

        normalized_query = _parse_query(query)
        try:
            response = await self._http_client.post(
                TAVILY_SEARCH_ENDPOINT,
                headers={
                    "accept": "application/json",
                    "authorization": f"Bearer {self._api_key}",
                    "content-type": "application/json",
                },
                json={
                    "query": normalized_query,
                    "search_depth": "basic",
                    "max_results": self._max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                    "auto_parameters": False,
                },
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise llm.ToolError(TIMEOUT_ERROR_MESSAGE) from error
        except Exception as error:
            raise llm.ToolError(PROVIDER_ERROR_MESSAGE) from error

        if not response.is_success:
            raise llm.ToolError(PROVIDER_ERROR_MESSAGE)
        try:
            payload: object = response.json()
        except ValueError as error:
            raise llm.ToolError(INVALID_RESPONSE_MESSAGE) from error
        return _normalize_results(payload, self._max_results)


def create_web_search_tool(client: WebSearchClient) -> WebSearchTool:
    """Create the cancellable, once-per-speech-turn ``searchWeb`` tool."""

    searched_turns: weakref.WeakSet[object] = weakref.WeakSet()

    @function_tool(
        name="searchWeb",
        description=load_prompt("search_tool_description.md"),
        on_duplicate="reject",
    )
    async def search_web(
        context: RunContext[Any], query: SearchQuery
    ) -> dict[str, list[dict[str, str]]] | None:
        normalized_query = _parse_query(query)
        speech_handle = context.speech_handle
        if speech_handle in searched_turns:
            raise llm.ToolError(DUPLICATE_SEARCH_MESSAGE)
        searched_turns.add(speech_handle)

        request = asyncio.ensure_future(client.search(normalized_query))
        try:
            await speech_handle.wait_if_not_interrupted([request])
            if speech_handle.interrupted:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                return None
            results = request.result()
            snippets: list[dict[str, str]] = []
            for result in results:
                snippet = _sanitize_snippet(result.snippet)
                if snippet is not None:
                    snippets.append({"snippet": snippet[:MAX_SNIPPET_LENGTH]})
            return {"results": snippets}
        except asyncio.CancelledError:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            raise
        except llm.ToolError:
            raise
        except Exception as error:
            raise llm.ToolError(PROVIDER_ERROR_MESSAGE) from error

    return cast(WebSearchTool, search_web)
