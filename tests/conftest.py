"""Shared test doubles for provider calls."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import pytest


@dataclass
class ProviderUsage:
    """Token usage returned by the fake provider."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass
class Message:
    """A fake provider response message."""

    content: str | None


@dataclass
class Choice:
    """A fake provider response choice."""

    message: Message


@dataclass
class ProviderResponse:
    """A minimal fake provider response."""

    choices: list[Choice]
    usage: ProviderUsage | None = None


class FailingChoicesResponse:
    """A fake response that fails while its choices are extracted."""

    def __init__(self, usage: ProviderUsage, error: BaseException) -> None:
        self.usage = usage
        self._error = error

    @property
    def choices(self) -> list[Choice]:
        """Raise the scripted extraction error."""
        raise self._error


class FakeCreate:
    """Scripted implementation of ``chat.completions.create``."""

    def __init__(
        self,
        outcomes: Sequence[object],
        recorded: list[dict[str, object]],
    ) -> None:
        self._outcomes = iter(outcomes)
        self._recorded = recorded

    async def __call__(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> object:
        self._recorded.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
        )
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeCompletions:
    """The fake client's completions namespace."""

    def __init__(self, create: FakeCreate) -> None:
        self.create = create


class FakeChat:
    """The fake client's chat namespace."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeClient:
    """A provider client exposing only ``chat.completions.create``."""

    def __init__(self, outcomes: Sequence[object]) -> None:
        self.recorded: list[dict[str, object]] = []
        create = FakeCreate(outcomes, self.recorded)
        self.chat = FakeChat(FakeCompletions(create))

    def __getattr__(self, name: str) -> object:
        """Reject attributes outside the fake client's explicit public surface."""
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


@pytest.fixture
def fake_client_factory() -> Callable[[Sequence[object]], FakeClient]:
    """Build a fake provider client from scripted outcomes."""
    return FakeClient


@pytest.fixture
def response_factory() -> Callable[..., ProviderResponse]:
    """Build a provider response with optional token usage."""

    def make_response(
        content: str | None,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        include_usage: bool = False,
    ) -> ProviderResponse:
        usage = None
        if include_usage:
            usage = ProviderUsage(prompt_tokens, completion_tokens, total_tokens)
        return ProviderResponse(choices=[Choice(message=Message(content))], usage=usage)

    return make_response


@pytest.fixture
def failing_choices_response_factory() -> Callable[..., FailingChoicesResponse]:
    """Build a response with usage that fails during choice extraction."""

    def make_response(
        error: BaseException,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> FailingChoicesResponse:
        usage = ProviderUsage(prompt_tokens, completion_tokens, total_tokens)
        return FailingChoicesResponse(usage, error)

    return make_response


@pytest.fixture
def recording_sleep() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    """Return an injected sleep function and its recorded delays."""
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    return delays, sleep
