"""Exception hierarchy for the amnesiac package."""


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

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        super().__init__(f"Too many axis failures: {failures!r}")
