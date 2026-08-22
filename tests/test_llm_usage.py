"""Tests for provider usage accumulation."""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from amnesiac.llm import _call_with_retry
from amnesiac.types import Usage


async def call_provider(
    client: Any,
    usage: Usage,
    sleep: Callable[[float], Awaitable[Any]],
) -> str | None:
    """Call the provider helper with shared usage-test arguments."""
    return await _call_with_retry(
        client=client,
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.3,
        usage=usage,
        max_attempts=2,
        retry_delays=(4,),
        axis_name="test-axis",
        sleep=sleep,
    )


async def test_usage_survives_a_failed_attempt(
    fake_client_factory: Callable[[Sequence[object]], Any],
    failing_choices_response_factory: Callable[..., object],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    first_response = failing_choices_response_factory(
        json.JSONDecodeError("msg", "doc", 0),
        prompt_tokens=3,
        completion_tokens=5,
        total_tokens=8,
    )
    second_response = response_factory(
        "second attempt",
        prompt_tokens=7,
        completion_tokens=11,
        total_tokens=18,
        include_usage=True,
    )
    client = fake_client_factory([first_response, second_response])
    delays, sleep = recording_sleep
    usage = Usage()

    result = await call_provider(client, usage, sleep)

    assert result == "second attempt"
    assert usage == Usage(prompt_tokens=10, completion_tokens=16, total_tokens=26, calls=2)
    assert delays == [4]


async def test_usage_accumulates_across_separate_calls(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory(
        [
            response_factory(
                "first",
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
                include_usage=True,
            ),
            response_factory(
                "second",
                prompt_tokens=7,
                completion_tokens=11,
                total_tokens=18,
                include_usage=True,
            ),
        ]
    )
    delays, sleep = recording_sleep
    usage = Usage()

    first_result = await call_provider(client, usage, sleep)
    second_result = await call_provider(client, usage, sleep)

    assert (first_result, second_result) == ("first", "second")
    assert usage == Usage(prompt_tokens=9, completion_tokens=14, total_tokens=23, calls=2)
    assert delays == []


async def test_missing_usage_block_leaves_accumulator_unchanged(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory([response_factory("content without usage")])
    delays, sleep = recording_sleep
    usage = Usage()

    result = await call_provider(client, usage, sleep)

    assert result == "content without usage"
    assert usage == Usage()
    assert delays == []
