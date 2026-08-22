"""Tests for concurrency limiter resolution."""

import asyncio
from types import TracebackType
from typing import Self

import pytest

import amnesiac.llm
from amnesiac.exceptions import AmnesiacError, ConfigurationError
from amnesiac.llm import _resolve_limiter


class CustomLimiter:
    """A non-semaphore async context manager for limiter tests."""

    def __init__(self) -> None:
        self.entries = 0

    async def __aenter__(self) -> Self:
        self.entries += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


async def test_limiter_and_explicit_concurrency_raise_configuration_error() -> None:
    assert issubclass(ConfigurationError, AmnesiacError)

    with pytest.raises(ConfigurationError):
        _resolve_limiter(limiter=CustomLimiter(), concurrency=3, concurrency_is_explicit=True)


async def test_supplied_limiter_is_returned_by_identity() -> None:
    supplied = CustomLimiter()

    resolved = _resolve_limiter(
        limiter=supplied,
        concurrency=3,
        concurrency_is_explicit=False,
    )

    assert resolved is supplied


async def test_default_limiter_is_semaphore_with_requested_capacity() -> None:
    concurrency = 3
    resolved = _resolve_limiter(
        limiter=None,
        concurrency=concurrency,
        concurrency_is_explicit=False,
    )

    assert isinstance(resolved, asyncio.Semaphore)
    assert not resolved.locked()
    for _ in range(concurrency):
        await resolved.acquire()
    assert resolved.locked()
    for _ in range(concurrency):
        resolved.release()


async def test_explicit_concurrency_without_limiter_returns_semaphore() -> None:
    resolved = _resolve_limiter(limiter=None, concurrency=2, concurrency_is_explicit=True)

    assert isinstance(resolved, asyncio.Semaphore)


async def test_resolved_limiters_are_async_context_managers() -> None:
    supplied = CustomLimiter()
    resolved_supplied = _resolve_limiter(
        limiter=supplied,
        concurrency=2,
        concurrency_is_explicit=False,
    )
    resolved_default = _resolve_limiter(
        limiter=None,
        concurrency=2,
        concurrency_is_explicit=False,
    )

    async with resolved_supplied:
        assert supplied.entries == 1
    async with resolved_default:
        pass


def test_module_has_no_limiter_instance() -> None:
    def is_limiter_instance(value: object) -> bool:
        return (
            not isinstance(value, type)
            and callable(getattr(value, "__aenter__", None))
            and callable(getattr(value, "__aexit__", None))
        )

    module_values = vars(amnesiac.llm).values()

    assert not any(isinstance(value, asyncio.Semaphore) for value in module_values)
    assert not any(is_limiter_instance(value) for value in module_values)
