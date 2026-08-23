"""Tests for the package root API."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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
    repository_root = Path(__file__).resolve().parent.parent
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
        cwd=repository_root,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "[]\n"
    assert result.stderr == ""


def test_root_import_succeeds_when_numpy_is_blocked() -> None:
    """Guard D-010 by proving the root imports while numpy is unavailable."""
    repository_root = Path(__file__).resolve().parent.parent
    code = """
import importlib.abc
import sys

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError(f"blocked import: {fullname}")
        return None

for name in list(sys.modules):
    if name == "numpy" or name.startswith("numpy."):
        del sys.modules[name]
sys.meta_path.insert(0, BlockNumpy())

try:
    import numpy
except ModuleNotFoundError:
    print("numpy blocked")
else:
    raise AssertionError("numpy import was not blocked")

import amnesiac

forbidden = ("amnesiac.summarize", "amnesiac.select", "amnesiac.llm")
assert not any(name in sys.modules for name in forbidden)
print("D-010 root ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        cwd=repository_root,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "numpy blocked\nD-010 root ok\n"
    assert result.stderr == ""


def test_public_surface_contains_no_private_names() -> None:
    """Section 10 excludes underscore-prefixed names from the public API."""
    assert all(not name.startswith("_") for name in amnesiac.__all__)


def test_root_does_not_expose_internal_module_layout() -> None:
    """Section 10 keeps internal modules and helper functions off the root."""
    internal_names = (
        "summarizer",
        "prompts",
        "config",
        "core",
        "summarize_axis",
        "summarize_meta",
        "_call_with_retry",
    )

    assert all(not hasattr(amnesiac, name) for name in internal_names)


def test_root_does_not_expose_subsystem_names_after_bare_import() -> None:
    """Section 10 keeps subsystem names off the package root."""
    repository_root = Path(__file__).resolve().parent.parent
    # Use a subprocess because importing a submodule elsewhere adds it to the package object.
    code = """
import amnesiac

assert not hasattr(amnesiac, "select")
assert not hasattr(amnesiac, "summarize")
print("subsystems absent")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        cwd=repository_root,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "subsystems absent\n"
    assert result.stderr == ""


def test_neuter_subpackage_is_absent() -> None:
    """Section 10 states that amnesiac.neuter does not exist in v0.1."""
    with pytest.raises(ModuleNotFoundError):
        __import__("amnesiac.neuter")

    assert importlib.util.find_spec("amnesiac.neuter") is None
