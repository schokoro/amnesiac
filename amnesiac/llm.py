"""Provider calls, retry handling, and concurrency limiter resolution."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import openai

from amnesiac.exceptions import ConfigurationError
from amnesiac.types import Usage, _add_usage

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    json.JSONDecodeError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class _EmptyContentError(Exception):
    """Signal an empty provider response within the retry loop."""


async def _call_with_retry(
    *,
    client: Any,
    model: str,
    messages: Sequence[dict[str, str]],
    temperature: float,
    usage: Usage,
    max_attempts: int,
    retry_delays: Sequence[float],
    axis_name: str,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> str | None:
    """Call the provider, retry transient failures, and accumulate returned usage."""
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                "OpenRouter request attempt %s/%s for axis %s",
                attempt,
                max_attempts,
                axis_name,
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            provider_usage = getattr(response, "usage", None)
            if provider_usage is not None:
                _add_usage(
                    usage,
                    Usage(
                        prompt_tokens=provider_usage.prompt_tokens or 0,
                        completion_tokens=provider_usage.completion_tokens or 0,
                        total_tokens=provider_usage.total_tokens or 0,
                        calls=1,
                    ),
                )
            content = response.choices[0].message.content
            if content is None:
                raise _EmptyContentError("provider returned empty content")
            return content
        except _RETRYABLE_ERRORS + (_EmptyContentError,) as exc:
            last_error = exc
            if attempt >= max_attempts:
                if isinstance(exc, _EmptyContentError):
                    return None
                logger.exception(
                    "OpenRouter request failed after %s attempts for axis %s",
                    max_attempts,
                    axis_name,
                )
                raise

            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            logger.warning(
                "OpenRouter request failed for axis %s on attempt %s/%s: %r; retrying in %ss",
                axis_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await sleep(delay)

    raise last_error


def _resolve_limiter(
    *,
    limiter: AbstractAsyncContextManager[Any] | None,
    concurrency: int,
    concurrency_is_explicit: bool,
) -> AbstractAsyncContextManager[Any]:
    """Return the supplied limiter or create the default semaphore."""
    if limiter is not None and concurrency_is_explicit:
        raise ConfigurationError("limiter and concurrency cannot both be set")
    if limiter is not None:
        return limiter
    return asyncio.Semaphore(concurrency)
