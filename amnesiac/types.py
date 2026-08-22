"""Shared data models for the amnesiac package."""

from pydantic import BaseModel, ConfigDict


class Doc(BaseModel):
    """A document supplied to amnesiac processing functions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    channel: str
    day_number: int
    doc_id: str | int | None = None


class Usage(BaseModel):
    """Accumulated token usage and provider call count."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


def _add_usage(target: Usage, other: Usage) -> None:
    """Add every field of ``other`` into ``target`` in place."""
    target.prompt_tokens += other.prompt_tokens
    target.completion_tokens += other.completion_tokens
    target.total_tokens += other.total_tokens
    target.calls += other.calls
