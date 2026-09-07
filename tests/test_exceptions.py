"""Tests for the package exception hierarchy."""

from amnesiac.exceptions import (
    AmnesiacError,
    ConfigurationError,
    MetaSummaryError,
    PromptRenderError,
    SummarizeError,
    TooManyAxisFailures,
)
from amnesiac.types import Usage


def test_exception_hierarchy_edges() -> None:
    assert issubclass(ConfigurationError, AmnesiacError)
    assert issubclass(PromptRenderError, AmnesiacError)
    assert issubclass(SummarizeError, AmnesiacError)
    assert issubclass(TooManyAxisFailures, SummarizeError)
    assert issubclass(TooManyAxisFailures, AmnesiacError)


def test_exception_hierarchy_has_no_cross_branch_inheritance() -> None:
    assert not issubclass(PromptRenderError, SummarizeError)
    assert not issubclass(ConfigurationError, SummarizeError)
    assert ConfigurationError.__bases__ == (AmnesiacError,)
    assert PromptRenderError.__bases__ == (AmnesiacError,)
    assert SummarizeError.__bases__ == (AmnesiacError,)
    assert TooManyAxisFailures.__bases__ == (SummarizeError,)


def test_too_many_axis_failures_preserves_failures() -> None:
    failures = {"inflation": "TimeoutError('timed out')"}

    error = TooManyAxisFailures(failures)

    assert error.failures == failures


def test_meta_summary_error_hierarchy() -> None:
    assert issubclass(MetaSummaryError, SummarizeError)
    assert issubclass(MetaSummaryError, AmnesiacError)
    assert MetaSummaryError.__bases__ == (SummarizeError,)
    assert not issubclass(MetaSummaryError, PromptRenderError)
    assert not issubclass(MetaSummaryError, ConfigurationError)


def test_summarize_error_handler_catches_meta_summary_error() -> None:
    caught = False

    try:
        raise MetaSummaryError(
            axis_summaries={"axis": "summary"},
            failed_axes=[],
            axis_errors={},
            usage=Usage(),
        )
    except SummarizeError:
        caught = True

    assert caught
