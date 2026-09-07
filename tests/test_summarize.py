"""Tests for two-stage summarization orchestration."""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import openai
import pytest

import amnesiac.summarize.summarizer as summarizer_module
from amnesiac.exceptions import (
    ConfigurationError,
    MetaSummaryError,
    PromptRenderError,
    SummarizeError,
    TooManyAxisFailures,
)
from amnesiac.summarize import (
    AxisSummariesResult,
    MetaResult,
    PromptPack,
    SummarizeConfig,
    SummarizeResult,
    Usage,
    summarize,
    summarize_axes,
    summarize_meta,
)
from amnesiac.types import Doc


def make_prompts() -> PromptPack:
    return PromptPack(
        name="test",
        axis_system="axis system {axis}",
        axis_user="axis user {axis}: {docs}",
        meta_system="meta system",
        meta_user="meta user: {axis_blocks}",
        doc_template="{day_number}|{channel}|{text}",
        axis_block_template="BLOCK[{axis}]={summary}",
        axis_block_separator="\n---\n",
    )


def make_doc(text: str = "news") -> Doc:
    return Doc(text=text, channel="channel", day_number=1)


class CountingLimiter(AbstractAsyncContextManager[None]):
    def __init__(self) -> None:
        self.enters = 0
        self.exits = 0

    async def __aenter__(self) -> None:
        self.enters += 1

    async def __aexit__(self, *args: object) -> None:
        self.exits += 1


class RecordingLimiter(AbstractAsyncContextManager[None]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self) -> None:
        self.events.append("enter")

    async def __aexit__(self, *args: object) -> None:
        self.events.append("exit")


async def test_empty_axes_raise_configuration_error(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    client = fake_client_factory([])

    with pytest.raises(ConfigurationError):
        await summarize(client=client, model="model", axes={}, prompts=make_prompts())

    assert client.recorded == []


async def test_unbound_axis_prompt_raises_before_any_provider_call(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    prompts = make_prompts().model_copy(
        update={"axis_system": "axis system {axis} for {horizon_days} days"}
    )
    client = fake_client_factory([])

    with pytest.raises(PromptRenderError) as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={"first": [make_doc()], "second": [make_doc()]},
            prompts=prompts,
        )

    assert "axis_system" in str(exc_info.value)
    assert "horizon_days" in str(exc_info.value)
    assert client.recorded == []


async def test_empty_document_list_still_produces_axis_call(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory("axis"), response_factory("meta")])

    await summarize(client=client, model="model", axes={"empty": []}, prompts=make_prompts())

    assert len(client.recorded) == 2
    assert client.recorded[0]["messages"][1]["content"] == "axis user empty: "


async def test_explicit_concurrency_conflicts_with_supplied_limiter(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    client = fake_client_factory([])

    with pytest.raises(ConfigurationError):
        await summarize(
            client=client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
            limiter=CountingLimiter(),
            config=SummarizeConfig(concurrency=5),
        )

    assert client.recorded == []


async def test_supplied_limiter_is_legal_with_default_config(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    limiter = CountingLimiter()
    client = fake_client_factory([response_factory("axis"), response_factory("meta")])

    await summarize(
        client=client,
        model="model",
        axes={"axis": [make_doc()]},
        prompts=make_prompts(),
        limiter=limiter,
        config=SummarizeConfig(),
    )

    assert (limiter.enters, limiter.exits) == (2, 2)


async def test_missing_limiter_builds_semaphore_from_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    resolved: list[AbstractAsyncContextManager[Any]] = []
    original = summarizer_module._resolve_limiter

    def recording_resolve(**kwargs: Any) -> AbstractAsyncContextManager[Any]:
        result = original(**kwargs)
        resolved.append(result)
        return result

    monkeypatch.setattr(summarizer_module, "_resolve_limiter", recording_resolve)
    client = fake_client_factory([response_factory("axis"), response_factory("meta")])

    await summarize(
        client=client,
        model="model",
        axes={"axis": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(concurrency=3),
    )

    assert len(resolved) == 1
    assert isinstance(resolved[0], asyncio.Semaphore)
    assert resolved[0]._value == 3


async def test_every_call_uses_limiter_and_meta_is_last_and_single(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    limiter = CountingLimiter()
    client = fake_client_factory(
        [response_factory("first"), response_factory("second"), response_factory("meta")]
    )

    await summarize(
        client=client,
        model="model",
        axes={"first": [make_doc()], "second": [make_doc()]},
        prompts=make_prompts(),
        limiter=limiter,
    )

    assert (limiter.enters, limiter.exits) == (3, 3)
    systems = [call["messages"][0]["content"] for call in client.recorded]
    assert systems == ["axis system first", "axis system second", "meta system"]


async def test_meta_blocks_keep_axis_order_when_an_axis_fails(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    placeholder = "MISSING"
    client = fake_client_factory(
        [
            response_factory("summary-b"),
            ValueError("broken"),
            response_factory("summary-c"),
            response_factory("meta"),
        ]
    )

    result = await summarize(
        client=client,
        model="model",
        axes={"b": [make_doc()], "a": [make_doc()], "c": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(failed_axis_placeholder=placeholder),
    )

    meta_user = client.recorded[-1]["messages"][1]["content"]
    assert meta_user.index("BLOCK[b]=summary-b") < meta_user.index("BLOCK[a]=MISSING")
    assert meta_user.index("BLOCK[a]=MISSING") < meta_user.index("BLOCK[c]=summary-c")
    assert list(result.axis_summaries) == ["b", "a", "c"]
    assert set(result.axis_summaries) == {"a", "b", "c"}


async def test_failed_axis_degrades_into_result_and_actual_meta_prompt(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    error = ValueError("provider failure")
    placeholder = "FAILED AXIS"
    client = fake_client_factory([error, response_factory("ok"), response_factory("meta")])

    result = await summarize(
        client=client,
        model="model",
        axes={"bad": [make_doc()], "good": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(max_failed_axes=2, failed_axis_placeholder=placeholder),
    )

    assert result.failed_axes == ["bad"]
    assert result.axis_errors == {"bad": repr(error)}
    assert result.axis_summaries["bad"] == placeholder
    assert placeholder in client.recorded[-1]["messages"][1]["content"]


async def test_too_many_failures_carries_all_errors_and_skips_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    errors = [ValueError("one"), RuntimeError("two"), LookupError("three")]
    client = fake_client_factory(errors)

    with pytest.raises(TooManyAxisFailures) as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={name: [make_doc()] for name in ("one", "two", "three")},
            prompts=make_prompts(),
            config=SummarizeConfig(max_failed_axes=2),
        )

    assert exc_info.value.failures == {
        "one": repr(errors[0]),
        "two": repr(errors[1]),
        "three": repr(errors[2]),
    }
    assert len(client.recorded) == 3


async def test_empty_axis_retries_then_degrades_and_reaches_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory(
        [
            response_factory(
                None,
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                include_usage=True,
            ),
            response_factory(
                None,
                prompt_tokens=4,
                completion_tokens=5,
                total_tokens=9,
                include_usage=True,
            ),
            response_factory(
                None,
                prompt_tokens=6,
                completion_tokens=7,
                total_tokens=13,
                include_usage=True,
            ),
            response_factory(
                "meta",
                prompt_tokens=8,
                completion_tokens=9,
                total_tokens=17,
                include_usage=True,
            ),
        ]
    )

    result = await summarize(
        client=client,
        model="model",
        axes={"empty": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(retry_delays=(0.0,), failed_axis_placeholder="FAILED AXIS"),
    )

    assert result.failed_axes == ["empty"]
    assert "SummarizeError" in result.axis_errors["empty"]
    assert result.axis_summaries["empty"] == "FAILED AXIS"
    assert "FAILED AXIS" in client.recorded[-1]["messages"][1]["content"]
    assert len(client.recorded) == 4
    assert result.usage == Usage(
        prompt_tokens=19,
        completion_tokens=23,
        total_tokens=42,
        calls=4,
    )


async def test_empty_axes_respect_max_failed_axes_and_skip_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory(None) for _ in range(6)])

    with pytest.raises(TooManyAxisFailures) as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={name: [make_doc()] for name in ("one", "two", "three")},
            prompts=make_prompts(),
            config=SummarizeConfig(
                max_attempts=2,
                retry_delays=(0.0,),
                max_failed_axes=2,
            ),
        )

    assert len(client.recorded) == 6
    assert set(exc_info.value.failures) == {"one", "two", "three"}


async def test_empty_meta_content_raises_summarize_error(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory(
        [
            response_factory("axis"),
            response_factory(None),
            response_factory(None),
            response_factory(None),
        ]
    )

    with pytest.raises(MetaSummaryError, match="meta"):
        await summarize(
            client=client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
            config=SummarizeConfig(retry_delays=(0.0,)),
        )

    assert len(client.recorded) == 4


async def test_meta_summary_error_carries_full_first_stage_result(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    placeholder = "DISTINCTIVE FAILED AXIS PLACEHOLDER"
    scripted_responses = [
        ("good summary", 1, 2, 3),
        (None, 4, 5, 9),
        (None, 6, 7, 13),
        (None, 8, 9, 17),
        (None, 10, 11, 21),
        (None, 12, 13, 25),
        (None, 14, 15, 29),
    ]
    client = fake_client_factory(
        [
            response_factory(
                content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                include_usage=True,
            )
            for content, prompt_tokens, completion_tokens, total_tokens in scripted_responses
        ]
    )

    with pytest.raises(MetaSummaryError, match="meta") as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={"good": [make_doc()], "bad": [make_doc()]},
            prompts=make_prompts(),
            config=SummarizeConfig(
                retry_delays=(0.0,),
                failed_axis_placeholder=placeholder,
            ),
        )

    assert exc_info.value.axis_summaries == {
        "good": "good summary",
        "bad": placeholder,
    }
    assert exc_info.value.failed_axes == ["bad"]
    assert exc_info.value.axis_errors == {
        "bad": repr(SummarizeError("Model returned empty content for axis 'bad'"))
    }
    assert exc_info.value.usage == Usage(
        prompt_tokens=55,
        completion_tokens=62,
        total_tokens=117,
        calls=7,
    )


async def test_too_many_axis_failures_carries_succeeded_summaries_and_usage(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    scripted_responses = [
        (None, 1, 11, 12),
        (None, 2, 12, 14),
        ("good summary", 3, 13, 16),
        (None, 4, 14, 18),
        (None, 5, 15, 20),
        (None, 6, 16, 22),
        (None, 7, 17, 24),
    ]
    client = fake_client_factory(
        [
            response_factory(
                content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                include_usage=True,
            )
            for content, prompt_tokens, completion_tokens, total_tokens in scripted_responses
        ]
    )

    with pytest.raises(TooManyAxisFailures) as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={
                "bad-one": [make_doc()],
                "bad-two": [make_doc()],
                "good": [make_doc()],
            },
            prompts=make_prompts(),
            config=SummarizeConfig(
                retry_delays=(0.0,),
                max_failed_axes=1,
                failed_axis_placeholder="MUST NOT APPEAR",
            ),
        )

    assert exc_info.value.axis_summaries == {"good": "good summary"}
    assert set(exc_info.value.axis_summaries) == {"good"}
    assert "bad-one" not in exc_info.value.axis_summaries
    assert "bad-two" not in exc_info.value.axis_summaries
    assert exc_info.value.failures == {
        "bad-one": repr(SummarizeError("Model returned empty content for axis 'bad-one'")),
        "bad-two": repr(SummarizeError("Model returned empty content for axis 'bad-two'")),
    }
    assert exc_info.value.usage == Usage(
        prompt_tokens=28,
        completion_tokens=98,
        total_tokens=126,
        calls=7,
    )
    assert len(client.recorded) == 7
    assert all(call["messages"][0]["content"] != "meta system" for call in client.recorded)


async def test_usage_includes_meta_and_usage_from_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    failing_choices_response_factory: Callable[..., object],
) -> None:
    error = httpx.ReadTimeout("retry")
    outcomes = [
        failing_choices_response_factory(
            error, prompt_tokens=1, completion_tokens=2, total_tokens=3
        ),
        response_factory(
            "axis", prompt_tokens=4, completion_tokens=5, total_tokens=9, include_usage=True
        ),
        response_factory(
            "meta", prompt_tokens=6, completion_tokens=7, total_tokens=13, include_usage=True
        ),
    ]
    client = fake_client_factory(outcomes)
    original = summarizer_module._call_with_retry

    async def no_sleep_call(**kwargs: Any) -> str | None:
        async def no_sleep(_: float) -> None:
            return None

        return await original(**kwargs, sleep=no_sleep)

    monkeypatch.setattr(summarizer_module, "_call_with_retry", no_sleep_call)

    result = await summarize(
        client=client,
        model="model",
        axes={"axis": [make_doc()]},
        prompts=make_prompts(),
    )

    assert result.usage == Usage(
        prompt_tokens=11,
        completion_tokens=14,
        total_tokens=25,
        calls=3,
    )


async def test_summarize_axes_returns_only_axis_stage_and_uses_limiter_before_calls(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    events: list[str] = []
    limiter = RecordingLimiter(events)
    client = fake_client_factory(
        [
            response_factory("summary-b"),
            response_factory("summary-a"),
            response_factory("summary-c"),
        ]
    )
    original_create = client.chat.completions.create

    async def recording_create(**kwargs: Any) -> object:
        events.append("call")
        return await original_create(**kwargs)

    client.chat.completions.create = recording_create

    result = await summarize_axes(
        client=client,
        model="model",
        axes={"b": [make_doc()], "a": [make_doc()], "c": [make_doc()]},
        prompts=make_prompts(),
        limiter=limiter,
    )

    assert isinstance(result, AxisSummariesResult)
    assert result.axis_summaries == {
        "b": "summary-b",
        "a": "summary-a",
        "c": "summary-c",
    }
    assert result.failed_axes == []
    assert result.axis_errors == {}
    assert result.usage == Usage()
    assert set(AxisSummariesResult.model_fields) == {
        "axis_summaries",
        "failed_axes",
        "axis_errors",
        "usage",
    }
    assert not hasattr(result, "meta")
    assert len(client.recorded) == 3
    assert all(call["messages"][0]["content"] != "meta system" for call in client.recorded)
    assert events == ["enter", "call", "exit"] * 3


async def test_summarize_axes_rejects_empty_axes_and_limiter_concurrency_conflict(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    empty_client = fake_client_factory([])
    with pytest.raises(ConfigurationError):
        await summarize_axes(
            client=empty_client,
            model="model",
            axes={},
            prompts=make_prompts(),
        )

    conflict_client = fake_client_factory([])
    with pytest.raises(ConfigurationError):
        await summarize_axes(
            client=conflict_client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
            limiter=CountingLimiter(),
            config=SummarizeConfig(concurrency=2),
        )

    assert empty_client.recorded == []
    assert conflict_client.recorded == []


async def test_summarize_axes_raises_on_threshold_and_skips_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    client = fake_client_factory([ValueError("one"), ValueError("two"), ValueError("three")])

    with pytest.raises(TooManyAxisFailures):
        await summarize_axes(
            client=client,
            model="model",
            axes={name: [make_doc()] for name in ("one", "two", "three")},
            prompts=make_prompts(),
            config=SummarizeConfig(max_failed_axes=2),
        )

    assert len(client.recorded) == 3
    assert all(call["messages"][0]["content"] != "meta system" for call in client.recorded)


async def test_summarize_axes_builds_semaphore_from_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    resolved: list[AbstractAsyncContextManager[Any]] = []
    original = summarizer_module._resolve_limiter

    def recording_resolve(**kwargs: Any) -> AbstractAsyncContextManager[Any]:
        result = original(**kwargs)
        resolved.append(result)
        return result

    monkeypatch.setattr(summarizer_module, "_resolve_limiter", recording_resolve)
    client = fake_client_factory([response_factory("axis")])

    await summarize_axes(
        client=client,
        model="model",
        axes={"axis": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(concurrency=4),
    )

    assert len(resolved) == 1
    assert isinstance(resolved[0], asyncio.Semaphore)
    assert resolved[0]._value == 4


async def test_summarize_meta_calls_once_with_only_supplied_summaries_and_limiter(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    events: list[str] = []
    limiter = RecordingLimiter(events)
    client = fake_client_factory([response_factory("meta result")])
    original_create = client.chat.completions.create

    async def recording_create(**kwargs: Any) -> object:
        events.append("call")
        return await original_create(**kwargs)

    client.chat.completions.create = recording_create

    result = await summarize_meta(
        client=client,
        model="model",
        axis_summaries={"first": "ready one", "second": "ready two"},
        prompts=make_prompts(),
        limiter=limiter,
    )

    assert result == MetaResult(meta="meta result", usage=Usage())
    assert len(client.recorded) == 1
    assert client.recorded[0]["messages"] == [
        {"role": "system", "content": "meta system"},
        {
            "role": "user",
            "content": "meta user: BLOCK[first]=ready one\n---\nBLOCK[second]=ready two",
        },
    ]
    assert events == ["enter", "call", "exit"]


async def test_summarize_meta_builds_semaphore_from_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    resolved: list[AbstractAsyncContextManager[Any]] = []
    original = summarizer_module._resolve_limiter

    def recording_resolve(**kwargs: Any) -> AbstractAsyncContextManager[Any]:
        result = original(**kwargs)
        resolved.append(result)
        return result

    monkeypatch.setattr(summarizer_module, "_resolve_limiter", recording_resolve)
    client = fake_client_factory([response_factory("meta")])

    result = await summarize_meta(
        client=client,
        model="model",
        axis_summaries={"axis": "summary"},
        prompts=make_prompts(),
        config=SummarizeConfig(concurrency=7),
    )

    assert result == MetaResult(meta="meta", usage=Usage())
    assert len(resolved) == 1
    assert isinstance(resolved[0], asyncio.Semaphore)
    assert resolved[0]._value == 7


async def test_summarize_meta_rejects_empty_input_and_limiter_concurrency_conflict(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    empty_client = fake_client_factory([])
    with pytest.raises(ConfigurationError):
        await summarize_meta(
            client=empty_client,
            model="model",
            axis_summaries={},
            prompts=make_prompts(),
        )

    conflict_client = fake_client_factory([])
    with pytest.raises(ConfigurationError):
        await summarize_meta(
            client=conflict_client,
            model="model",
            axis_summaries={"axis": "summary"},
            prompts=make_prompts(),
            limiter=CountingLimiter(),
            config=SummarizeConfig(concurrency=2),
        )

    assert empty_client.recorded == []
    assert conflict_client.recorded == []


async def test_summarize_meta_preserves_input_mapping_block_order(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory("meta")])

    await summarize_meta(
        client=client,
        model="model",
        axis_summaries={"б": "summary-b", "а": "summary-a", "в": "summary-v"},
        prompts=make_prompts(),
    )

    assert client.recorded[0]["messages"][1]["content"] == (
        "meta user: BLOCK[б]=summary-b\n---\nBLOCK[а]=summary-a\n---\nBLOCK[в]=summary-v"
    )


async def test_summarize_equals_explicit_stage_composition_byte_for_byte(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    error = ValueError("axis failed")
    outcomes = [
        error,
        response_factory(
            "good summary",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            include_usage=True,
        ),
        response_factory(
            "meta result",
            prompt_tokens=4,
            completion_tokens=5,
            total_tokens=9,
            include_usage=True,
        ),
    ]
    composed_client = fake_client_factory(outcomes)
    staged_client = fake_client_factory(outcomes)
    axes = {"bad": [make_doc("bad news")], "good": [make_doc("good news")]}
    config = SummarizeConfig(max_failed_axes=1, failed_axis_placeholder="MISSING")

    composed = await summarize(
        client=composed_client,
        model="model",
        axes=axes,
        prompts=make_prompts(),
        config=config,
    )
    axis_stage = await summarize_axes(
        client=staged_client,
        model="model",
        axes=axes,
        prompts=make_prompts(),
        config=config,
    )
    meta_stage = await summarize_meta(
        client=staged_client,
        model="model",
        axis_summaries=axis_stage.axis_summaries,
        prompts=make_prompts(),
        config=config,
    )

    assert composed.meta == meta_stage.meta
    assert composed.axis_summaries == axis_stage.axis_summaries
    assert composed.failed_axes == axis_stage.failed_axes
    assert composed.axis_errors == axis_stage.axis_errors
    assert composed.usage == axis_stage.usage + meta_stage.usage
    assert len(composed_client.recorded) == len(staged_client.recorded)
    assert composed_client.recorded == staged_client.recorded


async def test_summarize_with_explicit_concurrency_and_no_limiter_completes(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory("axis"), response_factory("meta")])

    result = await summarize(
        client=client,
        model="model",
        axes={"axis": [make_doc()]},
        prompts=make_prompts(),
        config=SummarizeConfig(concurrency=2),
    )

    assert result.meta == "meta"
    assert len(client.recorded) == 2


async def test_meta_recovery_costs_one_call_and_reuses_identical_prompt(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    failed_client = fake_client_factory(
        [
            response_factory("summary-b"),
            response_factory("summary-a"),
            response_factory(None),
            response_factory(None),
            response_factory(None),
        ]
    )

    with pytest.raises(MetaSummaryError) as exc_info:
        await summarize(
            client=failed_client,
            model="model",
            axes={"б": [make_doc()], "а": [make_doc()]},
            prompts=make_prompts(),
            config=SummarizeConfig(retry_delays=(0.0,)),
        )

    recovery_client = fake_client_factory([response_factory("recovered meta")])
    recovered = await summarize_meta(
        client=recovery_client,
        model="model",
        axis_summaries=exc_info.value.axis_summaries,
        prompts=make_prompts(),
    )

    assert len(recovery_client.recorded) == 1
    assert recovered == MetaResult(meta="recovered meta", usage=Usage())
    assert recovery_client.recorded[0]["messages"] == failed_client.recorded[-1]["messages"]


async def test_summarize_meta_empty_exhaustion_carries_input_and_exact_usage(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    axis_summaries = {"б": "summary-b", "а": "summary-a", "в": "summary-v"}
    client = fake_client_factory(
        [
            response_factory(
                None,
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                include_usage=True,
            ),
            response_factory(
                None,
                prompt_tokens=4,
                completion_tokens=5,
                total_tokens=9,
                include_usage=True,
            ),
            response_factory(
                None,
                prompt_tokens=6,
                completion_tokens=7,
                total_tokens=13,
                include_usage=True,
            ),
        ]
    )

    with pytest.raises(MetaSummaryError) as exc_info:
        await summarize_meta(
            client=client,
            model="model",
            axis_summaries=axis_summaries,
            prompts=make_prompts(),
            config=SummarizeConfig(retry_delays=(0.0,)),
        )

    assert exc_info.value.axis_summaries == axis_summaries
    assert list(exc_info.value.axis_summaries) == ["б", "а", "в"]
    assert exc_info.value.failed_axes == []
    assert exc_info.value.axis_errors == {}
    assert exc_info.value.usage == Usage(
        prompt_tokens=11,
        completion_tokens=14,
        total_tokens=25,
        calls=3,
    )


def test_summarize_result_is_mutable_and_ignores_extra_fields() -> None:
    result = SummarizeResult(
        meta="meta",
        axis_summaries={},
        failed_axes=[],
        axis_errors={},
        usage=Usage(),
        ignored="extra",
    )

    result.meta = "changed"

    assert result.meta == "changed"
    assert result.model_config == {}


def test_stage_result_models_use_default_model_config() -> None:
    assert AxisSummariesResult.model_config == {}
    assert MetaResult.model_config == {}


async def test_axis_cancellation_escapes_and_skips_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    client = fake_client_factory([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await summarize(
            client=client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
        )

    assert len(client.recorded) == 1


async def test_meta_provider_exception_escapes_unchanged(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    error = openai.APIStatusError(
        "provider failure",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        body=None,
    )
    client = fake_client_factory([response_factory("axis"), error])

    with pytest.raises(openai.APIStatusError) as exc_info:
        await summarize(
            client=client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
        )

    assert exc_info.value is error


def test_summarize_subpackage_exports_exact_public_api() -> None:
    import amnesiac.summarize as summarize_package

    assert summarize_package.__all__ == [
        "AxisSummariesResult",
        "MetaResult",
        "MetaSummaryError",
        "PromptPack",
        "PromptRenderError",
        "SummarizeConfig",
        "SummarizeError",
        "SummarizeResult",
        "TooManyAxisFailures",
        "Usage",
        "summarize",
        "summarize_axes",
        "summarize_meta",
    ]
