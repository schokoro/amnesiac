"""Exception hierarchy for the amnesiac package."""

from amnesiac.types import Usage


class AmnesiacError(Exception):
    """Base exception for errors raised by amnesiac."""


class ConfigurationError(AmnesiacError):
    """Raised when arguments form an invalid configuration."""


class PromptRenderError(AmnesiacError):
    """Raised when a prompt template cannot be rendered."""


class SummarizeError(AmnesiacError):
    """Base exception for summarization errors."""


class TooManyAxisFailures(SummarizeError):
    """Raised when more axes fail than summarization permits."""

    def __init__(
        self,
        failures: dict[str, str],
        *,
        axis_summaries: dict[str, str] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.failures = failures
        self.axis_summaries = axis_summaries if axis_summaries is not None else {}
        self.usage = usage if usage is not None else Usage()
        super().__init__(f"Too many axis failures: {failures!r}")


class MetaSummaryError(SummarizeError):
    """Raised when the meta-summary call exhausts empty-content retries."""

    def __init__(
        self,
        *,
        axis_summaries: dict[str, str],
        failed_axes: list[str],
        axis_errors: dict[str, str],
        usage: Usage,
    ) -> None:
        self.axis_summaries = axis_summaries
        self.failed_axes = failed_axes
        self.axis_errors = axis_errors
        self.usage = usage
        super().__init__("Model returned empty content for meta summary")
