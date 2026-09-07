"""Shared data models for the amnesiac package."""

from __future__ import annotations

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

    def __add__(self, other: Usage) -> Usage:
        """Return the pairwise sum of two usage values."""
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )


def _add_usage(target: Usage, other: Usage) -> None:
    """Add every field of ``other`` into ``target`` in place."""
    target.prompt_tokens += other.prompt_tokens
    target.completion_tokens += other.completion_tokens
    target.total_tokens += other.total_tokens
    target.calls += other.calls
