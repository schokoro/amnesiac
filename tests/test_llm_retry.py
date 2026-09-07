"""Tests for provider retry and cancellation behavior."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
import openai
import pytest

from amnesiac.llm import _call_with_retry
from amnesiac.types import Usage


async def call_provider(
    client: Any,
    usage: Usage,
    sleep: Callable[[float], Awaitable[Any]],
    *,
    max_attempts: int,
    retry_delays: Sequence[float],
) -> str | None:
    """Call the internal provider helper with shared test arguments."""
    return await _call_with_retry(
        client=client,
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.3,
        usage=usage,
        max_attempts=max_attempts,
        retry_delays=retry_delays,
        axis_name="test-axis",
        sleep=sleep,
    )


async def test_backoff_repeats_last_configured_delay(
    fake_client_factory: Callable[[Sequence[object]], Any],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    errors = [httpx.ReadTimeout(f"failure {attempt}") for attempt in range(5)]
    client = fake_client_factory(errors)
    delays, sleep = recording_sleep

    with pytest.raises(httpx.ReadTimeout) as exc_info:
        await call_provider(
            client,
            Usage(),
            sleep,
            max_attempts=5,
            retry_delays=(1, 2, 3),
        )

    assert exc_info.value is errors[-1]
    assert delays == [1, 2, 3, 3]
    assert len(client.recorded) == 5


async def test_empty_content_retries_and_then_succeeds(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory(
        [response_factory(None), response_factory(None), response_factory("text")]
    )
    delays, sleep = recording_sleep

    result = await call_provider(
        client,
        Usage(),
        sleep,
        max_attempts=3,
        retry_delays=(2, 4),
    )

    assert result == "text"
    assert len(client.recorded) == 3
    assert delays == [2, 4]


async def test_empty_content_exhausts_and_returns_none(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([response_factory(None) for _ in range(3)])
    delays, sleep = recording_sleep

    result = await call_provider(
        client,
        Usage(),
        sleep,
        max_attempts=3,
        retry_delays=(3, 6),
    )

    assert result is None
    assert len(client.recorded) == 3
    assert delays == [3, 6]


async def test_empty_content_backoff_repeats_last_configured_delay(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([response_factory(None) for _ in range(5)])
    delays, sleep = recording_sleep

    result = await call_provider(
        client,
        Usage(),
        sleep,
        max_attempts=5,
        retry_delays=(1, 2, 3),
    )

    assert result is None
    assert len(client.recorded) == 5
    assert delays == [1, 2, 3, 3]


async def test_usage_counts_every_billed_empty_content_attempt(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory(
        [
            response_factory(
                None,
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
                include_usage=True,
            ),
            response_factory(
                None,
                prompt_tokens=7,
                completion_tokens=11,
                total_tokens=18,
                include_usage=True,
            ),
            response_factory(
                "text",
                prompt_tokens=13,
                completion_tokens=17,
                total_tokens=30,
                include_usage=True,
            ),
        ]
    )
    delays, sleep = recording_sleep
    usage = Usage()

    result = await call_provider(
        client,
        usage,
        sleep,
        max_attempts=3,
        retry_delays=(1,),
    )

    assert result == "text"
    assert usage == Usage(prompt_tokens=22, completion_tokens=31, total_tokens=53, calls=3)
    assert len(client.recorded) == 3
    assert delays == [1, 1]


RETRYABLE_ERRORS = [
    json.JSONDecodeError("msg", "doc", 0),
    httpx.ReadTimeout("msg"),
    httpx.RemoteProtocolError("msg"),
    openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid")),
    openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),
    openai.RateLimitError(
        "msg",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        body=None,
    ),
    openai.InternalServerError(
        "msg",
        response=httpx.Response(
            500,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        body=None,
    ),
]


@pytest.mark.parametrize("error", RETRYABLE_ERRORS, ids=lambda error: type(error).__name__)
async def test_each_retryable_error_succeeds_on_second_attempt(
    error: BaseException,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([error, response_factory("expected content")])
    delays, sleep = recording_sleep

    result = await call_provider(
        client,
        Usage(),
        sleep,
        max_attempts=2,
        retry_delays=(7,),
    )

    assert result == "expected content"
    assert len(client.recorded) == 2
    assert delays == [7]


NON_RETRYABLE_ERRORS = [
    ValueError("msg"),
    openai.APIStatusError(
        "msg",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        body=None,
    ),
    openai.PermissionDeniedError(
        "msg",
        response=httpx.Response(
            403,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        body=None,
    ),
]


@pytest.mark.parametrize("error", NON_RETRYABLE_ERRORS, ids=lambda error: type(error).__name__)
async def test_non_retryable_errors_escape_first_attempt(
    error: Exception,
    fake_client_factory: Callable[[Sequence[object]], Any],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([error])
    delays, sleep = recording_sleep

    with pytest.raises(type(error)) as exc_info:
        await call_provider(
            client,
            Usage(),
            sleep,
            max_attempts=5,
            retry_delays=(1, 2, 3),
        )

    assert exc_info.value is error
    assert len(client.recorded) == 1
    assert delays == []


async def test_cancellation_propagates_without_sleep_or_usage_damage(
    fake_client_factory: Callable[[Sequence[object]], Any],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([asyncio.CancelledError()])
    delays, sleep = recording_sleep
    usage = Usage()

    with pytest.raises(asyncio.CancelledError):
        await call_provider(
            client,
            usage,
            sleep,
            max_attempts=5,
            retry_delays=(1, 2, 3),
        )

    assert len(client.recorded) == 1
    assert delays == []
    assert usage == Usage()
