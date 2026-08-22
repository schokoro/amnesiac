"""Tests for package logging behavior."""

import ast
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import amnesiac
from amnesiac.llm import _call_with_retry
from amnesiac.types import Usage


async def test_exactly_one_warning_is_logged_per_retry(
    caplog: Any,
    fake_client_factory: Callable[[Sequence[object]], Any],
    response_factory: Callable[..., object],
    recording_sleep: tuple[list[float], Callable[[float], Awaitable[None]]],
) -> None:
    client = fake_client_factory(
        [json.JSONDecodeError("msg", "doc", 0), response_factory("success")]
    )
    delays, sleep = recording_sleep

    with caplog.at_level(logging.WARNING):
        result = await _call_with_retry(
            client=client,
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.3,
            usage=Usage(),
            max_attempts=2,
            retry_delays=(6,),
            axis_name="test-axis",
            sleep=sleep,
        )

    assert result == "success"
    assert delays == [6]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.name == "amnesiac.llm" or record.name.startswith("amnesiac.")


def test_package_configures_no_logging() -> None:
    package_logger = logging.getLogger("amnesiac")

    assert package_logger.handlers == []
    assert package_logger.level == logging.NOTSET


def test_package_contains_no_print_calls() -> None:
    package_file = amnesiac.__file__
    assert package_file is not None
    package_dir = Path(package_file).resolve().parent

    print_locations: list[str] = []
    for source_path in package_dir.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                print_locations.append(f"{source_path}:{node.lineno}")

    assert print_locations == []


def test_package_contains_no_forbidden_exception_handlers() -> None:
    package_file = amnesiac.__file__
    assert package_file is not None
    package_dir = Path(package_file).resolve().parent

    handler_locations: list[str] = []
    for source_path in package_dir.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and (
                node.type is None
                or (
                    isinstance(node.type, ast.Name)
                    and node.type.id in {"Exception", "BaseException"}
                )
            ):
                handler_locations.append(f"{source_path}:{node.lineno}")

    assert handler_locations == []
