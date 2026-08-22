"""Orchestration for two-stage news summarization."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from amnesiac.exceptions import ConfigurationError, SummarizeError, TooManyAxisFailures
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

    concurrency_is_explicit = config is not None and "concurrency" in config.model_fields_set
    resolved_config = config if config is not None else SummarizeConfig()
    resolved_limiter = _resolve_limiter(
        limiter=limiter,
        concurrency=resolved_config.concurrency,
        concurrency_is_explicit=concurrency_is_explicit,
    )
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
        async with resolved_limiter:
            summary = await _call_with_retry(
                client=client,
                model=model,
                messages=messages,
                temperature=resolved_config.temperature,
                usage=usage,
                max_attempts=resolved_config.max_attempts,
                retry_delays=resolved_config.retry_delays,
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
            axis_summaries[name] = resolved_config.failed_axis_placeholder
        else:
            axis_summaries[name] = result

    if len(failed_axes) > resolved_config.max_failed_axes:
        raise TooManyAxisFailures(axis_errors)
    if failed_axes:
        logger.warning("Axis failed but continuing: %s", ", ".join(failed_axes))

    rendered_blocks = prompts._render_axis_blocks(axis_summaries)
    meta_messages = [
        {"role": "system", "content": prompts._render_meta_system()},
        {
            "role": "user",
            "content": prompts._render_meta_user(axis_blocks=rendered_blocks),
        },
    ]
    async with resolved_limiter:
        meta = await _call_with_retry(
            client=client,
            model=model,
            messages=meta_messages,
            temperature=resolved_config.temperature,
            usage=usage,
            max_attempts=resolved_config.max_attempts,
            retry_delays=resolved_config.retry_delays,
            axis_name="meta",
        )
    if meta is None:
        raise SummarizeError("Model returned empty content for meta summary")

    return SummarizeResult(
        meta=meta,
        axis_summaries=axis_summaries,
        failed_axes=failed_axes,
        axis_errors=axis_errors,
        usage=usage,
    )
