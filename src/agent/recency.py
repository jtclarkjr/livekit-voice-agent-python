"""Deterministic per-turn policy for current-information web search."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

SearchMode = Literal["required", "forbidden", "auto"]


@dataclass(frozen=True, slots=True)
class RecencyHistoryMessage:
    """The small history projection used by the recency classifier."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class RecencyDecision:
    """A trusted runtime decision about search for one user turn."""

    mode: SearchMode
    reason: str
    query_hint: str | None = None


MAX_HISTORY_MESSAGES = 4

_FORCE_WEB_SEARCH = (
    "search web",
    "search the web",
    "search the internet",
    "search online",
    "look it up online",
    "browse the web",
    "browse the internet",
    "browse online",
    "check online",
    "use internet",
    "use the internet",
    "look it up",
    "verify online",
)

_DISABLE_WEB_SEARCH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:no|without)\s+(?:(?:web|online|internet)\s+)?(?:search(?:ing)?|browsing)(?:\s+(?:the\s+)?(?:web|internet|online))?\b",
        r"\bwithout\s+(?:(?:doing|performing|running|using)\s+)?(?:(?:a|any|the)\s+)?(?:(?:web|online|internet)\s+)?(?:search(?:ing)?|browsing)\b",
        r"\bwithout\s+(?:searching|browsing|using|checking)\s+(?:the\s+)?(?:web|internet|online)\b",
        r"\b(?:do not|don't|dont|never)\s+(?:(?:do|perform|run|use)\s+)?(?:(?:a|any|the)\s+)?(?:(?:web|online|internet)\s+)?(?:search|browse|browsing)\b",
        r"\b(?:do not|don't|dont|never)\s+look\s+(?:it|this|that)\s+up\b",
        r"\b(?:do not|don't|dont|never)\s+(?:check|verify)\s+online\b",
        r"\b(?:do not|don't|dont|never)\s+go\s+online\b",
        r"\b(?:do not|don't|dont|never)\s+use(?:\s+the)?\s+(?:internet|web)\b",
        r"\b(?:no|without)(?:\s+the)?\s+internet\b",
        r"\b(?:avoid|skip)\s+(?:web\s+)?(?:search|searching|browsing)\b",
        r"\b(?:avoid|skip)\s+(?:the\s+)?(?:internet|web)\b",
        r"\b(?:stay|keep\s+it)\s+offline\b",
        r"\boffline\s+only\b",
    )
)

_RECENCY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\blatest\b",
        r"\brecent\b",
        r"\bcurrent\b",
        r"\bup[- ]to[- ]date\b",
        r"\bright now\b",
        r"\btoday\b",
        r"\byesterday\b",
        r"\bthis week\b",
        r"\bthis month\b",
        r"\bthis year\b",
        r"\bbreaking news\b",
        r"\bjust announced\b",
        r"\bprice of\b",
        r"\bstock price\b",
        r"\bweather\b",
        r"\bscore\b",
        r"\belection\b",
        r"\bwho is (the )?(president|ceo|prime minister)\b",
    )
)

_COMPANY_FACT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhere is\b",
        r"\blocated\b",
        r"\baddress\b",
        r"\b(headquarters|headquartered|hq)\b",
        r"\b(ceo|founder|funding|valuation)\b",
        r"\bavailable in\b",
        r"\bcontact\b",
    )
)

_VERIFICATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bverify\b",
        r"\bverification\b",
        r"\bconfirm\b",
        r"\bconfirmed\b",
        r"\baccurate\b",
        r"\baccuracy\b",
        r"\bcorrect\b",
        r"\bfact[- ]?check\b",
        r"\bcross[- ]?check\b",
    )
)

_MODEL_TERM_PATTERNS = (
    re.compile(
        r"\b(gpt|openai|claude|gemini|llama|mistral|o1|o3|o4|opus|sonnet|haiku)\b",
        re.IGNORECASE,
    ),
)

_MODEL_RELEASE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(version|release|released|announced|launch|latest model|new model)\b",
        r"\b(new|newest|latest|current|available)\b.{0,40}\bmodels?\b",
        r"\bmodels?\b.{0,40}\b(new|newest|latest|current|available)\b",
        r"\bmodel (lineup|catalog|catalogue|availability)\b",
        r"\b\d+(\.\d+){1,2}\b",
    )
)

_FOLLOW_UP_PREFIX = re.compile(r"^\s*(and|also|too|what about|how about|about)\b", re.IGNORECASE)
_NON_TOPIC_FRAGMENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:i|i'm|im|we|we're|you|you're|he|she|it|it's|they|they're|this|that|these|those|there|here)\b",
        r"^(?:see|talk|catch)\s+you\b",
        r"^(?:tell|explain|show|give|please|stop|cancel|done|finished|heading|leaving|bye|goodbye|later|nevermind|never mind)\b",
        r"^(?:all done|that's all|thats all|great answer|nice answer|cool answer)\b",
    )
)

_ACKNOWLEDGEMENT_MESSAGES = frozenset(
    {
        "thanks",
        "thanks for that",
        "thanks for the help",
        "thanks for your help",
        "thanks again",
        "thanks so much",
        "thank you",
        "thank you for that",
        "thank you for the help",
        "thank you for your help",
        "thank you again",
        "thank you very much",
        "much appreciated",
        "appreciate it",
        "ok",
        "okay",
        "cool",
        "great",
        "perfect",
        "excellent",
        "awesome",
        "nice",
        "got it",
        "sounds good",
        "sounds great",
        "all good",
        "understood",
    }
)

_UNSAFE_HINT_PATTERN = re.compile(
    r"\b(ignore|override|reveal|system|developer|instruction|prompt|secret|role|tool|output)\b",
    re.IGNORECASE,
)


def _normalize_message(value: str) -> str:
    return " ".join(value.translate(str.maketrans("‘’ʼ", "'''")).strip().split()).lower()


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _has_recency_signal(text: str) -> bool:
    return _matches_any(text, _RECENCY_PATTERNS)


def _has_verification_signal(text: str) -> bool:
    return _matches_any(text, _VERIFICATION_PATTERNS)


def _has_company_fact_signal(text: str) -> bool:
    return _matches_any(text, _COMPANY_FACT_PATTERNS)


def _has_model_release_signal(text: str) -> bool:
    return _matches_any(text, _MODEL_TERM_PATTERNS) and _matches_any(text, _MODEL_RELEASE_PATTERNS)


def _is_safe_topic_fragment(value: str) -> bool:
    return value != "" and all(character.isalnum() or character in "_.+ -" for character in value)


def _is_follow_up_message(message: str) -> bool:
    normalized = _normalize_message(message)
    words = normalized.split() if normalized else []
    terse_topic_fragment = (
        len(words) <= 4
        and _is_safe_topic_fragment(normalized)
        and not _matches_any(normalized, _NON_TOPIC_FRAGMENT_PATTERNS)
    )
    terse_question = (
        len(words) <= 4
        and normalized.endswith("?")
        and len(normalized[:-1]) <= 40
        and _is_safe_topic_fragment(normalized[:-1].strip())
    )
    return (
        normalized != ""
        and len(normalized) <= 80
        and (
            terse_question
            or terse_topic_fragment
            or _FOLLOW_UP_PREFIX.search(normalized) is not None
        )
    )


def _is_acknowledgement_message(message: str) -> bool:
    normalized = re.sub(r"[.!?]+$", "", _normalize_message(message)).strip()
    return normalized in _ACKNOWLEDGEMENT_MESSAGES


def _recent_user_history_text(
    previous_messages: Sequence[RecencyHistoryMessage],
) -> str:
    user_messages = [
        normalized
        for message in previous_messages
        if message.role == "user" and (normalized := _normalize_message(message.content)) != ""
    ]
    return " ".join(user_messages[-MAX_HISTORY_MESSAGES:])


def _history_requires_search(history: str) -> bool:
    return (
        _has_recency_signal(history)
        or _has_verification_signal(history)
        or _has_company_fact_signal(history)
        or _has_model_release_signal(history)
    )


def _extract_follow_up_subject(message: str) -> str:
    subject = re.sub(
        r"^\s*(what about|how about|about|and|also)\s+(?:the\s+)?",
        "",
        message,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"\?+$", "", subject).strip()


def _build_follow_up_query_hint(message: str) -> str | None:
    normalized = _normalize_message(message)
    subject = _extract_follow_up_subject(normalized)
    query_hint = (
        normalized
        if subject == ""
        else subject
        if _has_recency_signal(normalized)
        else f"latest {subject}"
    )

    if (
        len(query_hint) > 80
        or not _is_safe_topic_fragment(query_hint)
        or _UNSAFE_HINT_PATTERN.search(query_hint) is not None
    ):
        return None
    return query_hint


def classify_recency(
    message: str,
    previous_messages: Sequence[RecencyHistoryMessage] = (),
) -> RecencyDecision:
    """Classify a user turn, giving explicit search opt-out highest priority."""

    normalized = _normalize_message(message)

    if _matches_any(normalized, _DISABLE_WEB_SEARCH_PATTERNS):
        return RecencyDecision(mode="forbidden", reason="explicit_disable")
    if any(phrase in normalized for phrase in _FORCE_WEB_SEARCH):
        return RecencyDecision(mode="required", reason="explicit_search")
    if _has_verification_signal(normalized):
        return RecencyDecision(mode="required", reason="verification")
    if "http://" in normalized or "https://" in normalized:
        return RecencyDecision(mode="required", reason="url")
    if _has_company_fact_signal(normalized):
        return RecencyDecision(mode="required", reason="company_fact")
    if _has_model_release_signal(normalized):
        return RecencyDecision(mode="required", reason="model_release")
    if _has_recency_signal(normalized):
        return RecencyDecision(mode="required", reason="recency")
    if _is_acknowledgement_message(normalized):
        return RecencyDecision(mode="auto", reason="acknowledgement")

    history = _recent_user_history_text(previous_messages)
    if _is_follow_up_message(normalized) and history != "" and _history_requires_search(history):
        return RecencyDecision(
            mode="required",
            reason="inherited_recency",
            query_hint=_build_follow_up_query_hint(normalized),
        )

    return RecencyDecision(mode="auto", reason="no_signal")
