"""Tests for the package root API."""

import subprocess
import sys

import amnesiac


def test_root_exports_exact_public_api() -> None:
    assert amnesiac.__all__ == ["AmnesiacError", "ConfigurationError", "Doc"]
    assert all(getattr(amnesiac, name) is not None for name in amnesiac.__all__)


def test_root_does_not_expose_non_root_api() -> None:
    hidden_names = (
        "Usage",
        "SummarizeError",
        "PromptRenderError",
        "TooManyAxisFailures",
    )

    assert all(not hasattr(amnesiac, name) for name in hidden_names)


def test_root_import_does_not_import_subsystems() -> None:
    code = """
import sys
import amnesiac

forbidden = ("amnesiac.summarize", "amnesiac.select", "amnesiac.llm")
loaded = [name for name in forbidden if name in sys.modules]
print(repr(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "[]\n"
    assert result.stderr == ""
