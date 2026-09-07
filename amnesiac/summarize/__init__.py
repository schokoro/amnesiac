"""Two-stage news summarization."""

from amnesiac.exceptions import (
    MetaSummaryError,
    PromptRenderError,
    SummarizeError,
    TooManyAxisFailures,
)
from amnesiac.types import Usage

from .config import SummarizeConfig
from .prompts import PromptPack
from .summarizer import (
    AxisSummariesResult,
    MetaResult,
    SummarizeResult,
    summarize,
    summarize_axes,
    summarize_meta,
)

__all__ = [
    "AxisSummariesResult",
    "MetaResult",
    "MetaSummaryError",
    "PromptPack",
    "PromptRenderError",
    "SummarizeConfig",
    "SummarizeError",
    "SummarizeResult",
    "TooManyAxisFailures",
    "Usage",
    "summarize",
    "summarize_axes",
    "summarize_meta",
]
