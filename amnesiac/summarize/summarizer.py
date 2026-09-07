"""Orchestration for two-stage news summarization."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from amnesiac.exceptions import (
    ConfigurationError,
    MetaSummaryError,
    SummarizeError,
    TooManyAxisFailures,
)
from amnesiac.llm import _call_with_retry, _resolve_limiter
from amnesiac.types import Doc, Usage

from .config import SummarizeConfig
from .prompts import PromptPack

logger = logging.getLogger(__name__)


class SummarizeResult(BaseModel):
    """The meta-summary, per-axis results, failures, and aggregate provider usage."""

    meta: str
    axis_summaries: dict[str, str]
    failed_axes: list[str]
    axis_errors: dict[str, str]
    usage: Usage


class AxisSummariesResult(BaseModel):
    """Per-axis summaries, failures, and aggregate provider usage."""

    axis_summaries: dict[str, str]
    failed_axes: list[str]
    axis_errors: dict[str, str]
    usage: Usage


class MetaResult(BaseModel):
    """The meta-summary and aggregate provider usage."""

    meta: str
    usage: Usage


def _resolve_resources(
    *,
    limiter: AbstractAsyncContextManager[Any] | None,
    config: SummarizeConfig | None,
) -> tuple[SummarizeConfig, AbstractAsyncContextManager[Any]]:
    """Resolve configuration and limiter at a public entry-point boundary."""
    concurrency_is_explicit = config is not None and "concurrency" in config.model_fields_set
    resolved_config = config if config is not None else SummarizeConfig()
    resolved_limiter = _resolve_limiter(
        limiter=limiter,
        concurrency=resolved_config.concurrency,
        concurrency_is_explicit=concurrency_is_explicit,
    )
    return resolved_config, resolved_limiter


async def summarize(
    *,
    client: AsyncOpenAI,
    model: str,
    axes: Mapping[str, Sequence[Doc]],
    prompts: PromptPack,
    limiter: AbstractAsyncContextManager[Any] | None = None,
    config: SummarizeConfig | None = None,
) -> SummarizeResult:
    """Summarize documents per axis, then combine the summaries into one narrative."""
    if not axes:
        raise ConfigurationError("axes cannot be empty")

    resolved_config, resolved_limiter = _resolve_resources(
        limiter=limiter,
        config=config,
    )
    axes_result = await _summarize_axes_resolved(
        client=client,
        model=model,
        axes=axes,
        prompts=prompts,
        limiter=resolved_limiter,
        config=resolved_config,
    )
    meta_result = await _summarize_meta_resolved(
        client=client,
        model=model,
        axis_summaries=axes_result.axis_summaries,
        prompts=prompts,
        limiter=resolved_limiter,
        config=resolved_config,
        usage=axes_result.usage,
        failed_axes=axes_result.failed_axes,
        axis_errors=axes_result.axis_errors,
    )

    return SummarizeResult(
        meta=meta_result.meta,
        axis_summaries=axes_result.axis_summaries,
        failed_axes=axes_result.failed_axes,
        axis_errors=axes_result.axis_errors,
        usage=meta_result.usage,
    )


async def summarize_axes(
    *,
    client: AsyncOpenAI,
    model: str,
    axes: Mapping[str, Sequence[Doc]],
    prompts: PromptPack,
    limiter: AbstractAsyncContextManager[Any] | None = None,
    config: SummarizeConfig | None = None,
) -> AxisSummariesResult:
    """Summarize documents per axis without making a meta-summary call."""
    if not axes:
        raise ConfigurationError("axes cannot be empty")

    resolved_config, resolved_limiter = _resolve_resources(limiter=limiter, config=config)
    return await _summarize_axes_resolved(
        client=client,
        model=model,
        axes=axes,
        prompts=prompts,
        limiter=resolved_limiter,
        config=resolved_config,
    )


async def _summarize_axes_resolved(
    *,
    client: AsyncOpenAI,
    model: str,
    axes: Mapping[str, Sequence[Doc]],
    prompts: PromptPack,
    limiter: AbstractAsyncContextManager[Any],
    config: SummarizeConfig,
) -> AxisSummariesResult:
    """Run the axis stage with already-resolved resources."""
    usage = Usage()

    axis_names = list(axes.keys())
    axis_messages = []
    for name in axis_names:
        rendered_docs = prompts._render_docs(axes[name])
        axis_messages.append(
            [
                {"role": "system", "content": prompts._render_axis_system(axis=name)},
                {
                    "role": "user",
                    "content": prompts._render_axis_user(axis=name, docs=rendered_docs),
                },
            ]
        )

    async def bounded(name: str, messages: list[dict[str, str]]) -> str:
        async with limiter:
            summary = await _call_with_retry(
                client=client,
                model=model,
                messages=messages,
                temperature=config.temperature,
                usage=usage,
                max_attempts=config.max_attempts,
                retry_delays=config.retry_delays,
                axis_name=name,
            )
        if summary is None:
            raise SummarizeError(f"Model returned empty content for axis {name!r}")
        return summary

    results = await asyncio.gather(
        *(bounded(name, axis_messages[index]) for index, name in enumerate(axis_names)),
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result

    failed_axes: list[str] = []
    axis_errors: dict[str, str] = {}
    axis_summaries: dict[str, str] = {}
    for index, name in enumerate(axis_names):
        result = results[index]
        if isinstance(result, Exception):
            failed_axes.append(name)
            axis_errors[name] = repr(result)
            axis_summaries[name] = config.failed_axis_placeholder
        else:
            axis_summaries[name] = result

    if len(failed_axes) > config.max_failed_axes:
        raise TooManyAxisFailures(
            axis_errors,
            axis_summaries={
                name: axis_summaries[name] for name in axis_names if name not in axis_errors
            },
            usage=usage,
        )
    if failed_axes:
        logger.warning("Axis failed but continuing: %s", ", ".join(failed_axes))

    return AxisSummariesResult(
        axis_summaries=axis_summaries,
        failed_axes=failed_axes,
        axis_errors=axis_errors,
        usage=usage,
    )


async def summarize_meta(
    *,
    client: AsyncOpenAI,
    model: str,
    axis_summaries: Mapping[str, str],
    prompts: PromptPack,
    limiter: AbstractAsyncContextManager[Any] | None = None,
    config: SummarizeConfig | None = None,
) -> MetaResult:
    """Combine ready axis summaries into one meta-summary."""
    if not axis_summaries:
        raise ConfigurationError("axis_summaries cannot be empty")

    resolved_config, resolved_limiter = _resolve_resources(limiter=limiter, config=config)
    return await _summarize_meta_resolved(
        client=client,
        model=model,
        axis_summaries=axis_summaries,
        prompts=prompts,
        limiter=resolved_limiter,
        config=resolved_config,
        usage=Usage(),
        failed_axes=[],
        axis_errors={},
    )


async def _summarize_meta_resolved(
    *,
    client: AsyncOpenAI,
    model: str,
    axis_summaries: Mapping[str, str],
    prompts: PromptPack,
    limiter: AbstractAsyncContextManager[Any],
    config: SummarizeConfig,
    usage: Usage,
    failed_axes: list[str],
    axis_errors: dict[str, str],
) -> MetaResult:
    """Run the meta stage with already-resolved resources and error context."""
    rendered_blocks = prompts._render_axis_blocks(axis_summaries)
    meta_messages = [
        {"role": "system", "content": prompts._render_meta_system()},
        {
            "role": "user",
            "content": prompts._render_meta_user(axis_blocks=rendered_blocks),
        },
    ]
    async with limiter:
        meta = await _call_with_retry(
            client=client,
            model=model,
            messages=meta_messages,
            temperature=config.temperature,
            usage=usage,
            max_attempts=config.max_attempts,
            retry_delays=config.retry_delays,
            axis_name="meta",
        )
    if meta is None:
        raise MetaSummaryError(
            axis_summaries=axis_summaries,
            failed_axes=failed_axes,
            axis_errors=axis_errors,
            usage=usage,
        )

    return MetaResult(meta=meta, usage=usage)
