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
    PromptRenderError,
    SummarizeError,
    TooManyAxisFailures,
)
from amnesiac.summarize import (
    PromptPack,
    SummarizeConfig,
    SummarizeResult,
    Usage,
    summarize,
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


async def test_empty_axes_raise_configuration_error(
    fake_client_factory: Callable[[Sequence[object]], Any],
) -> None:
    client = fake_client_factory([])

    with pytest.raises(ConfigurationError):
        await summarize(client=client, model="model", axes={}, prompts=make_prompts())

    assert client.create.calls == []


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
    assert client.create.calls == []


async def test_empty_document_list_still_produces_axis_call(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory("axis"), response_factory("meta")])

    await summarize(client=client, model="model", axes={"empty": []}, prompts=make_prompts())

    assert len(client.create.calls) == 2
    assert client.create.calls[0]["messages"][1]["content"] == "axis user empty: "


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

    assert client.create.calls == []


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
    systems = [call["messages"][0]["content"] for call in client.create.calls]
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

    meta_user = client.create.calls[-1]["messages"][1]["content"]
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
    assert placeholder in client.create.calls[-1]["messages"][1]["content"]


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
    assert len(client.create.calls) == 3


async def test_empty_axis_content_is_one_failed_call_and_reaches_meta(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory(None), response_factory("meta")])

    result = await summarize(
        client=client,
        model="model",
        axes={"empty": [make_doc()]},
        prompts=make_prompts(),
    )

    assert result.failed_axes == ["empty"]
    assert "SummarizeError" in result.axis_errors["empty"]
    assert result.axis_summaries["empty"] in client.create.calls[-1]["messages"][1]["content"]
    assert len(client.create.calls) == 2


async def test_empty_meta_content_raises_summarize_error(
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
) -> None:
    client = fake_client_factory([response_factory("axis"), response_factory(None)])

    with pytest.raises(SummarizeError, match="meta"):
        await summarize(
            client=client,
            model="model",
            axes={"axis": [make_doc()]},
            prompts=make_prompts(),
        )


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

    assert len(client.create.calls) == 1


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
        "PromptPack",
        "PromptRenderError",
        "SummarizeConfig",
        "SummarizeError",
        "SummarizeResult",
        "TooManyAxisFailures",
        "Usage",
        "summarize",
    ]
