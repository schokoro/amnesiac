import ast
import importlib
import re
from pathlib import Path
from types import ModuleType

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
PYTHON_BLOCK = re.compile(r"^```python\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _python_blocks() -> list[str]:
    return PYTHON_BLOCK.findall(README.read_text(encoding="utf-8"))


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    parts.append(value.id)
    return list(reversed(parts))


def _assert_imported_names_resolve(source: str) -> None:
    tree = ast.parse(source)
    imported_amnesiac_modules: dict[str, ModuleType] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is not None
            module = importlib.import_module(node.module)
            for imported_name in node.names:
                getattr(module, imported_name.name)
        elif isinstance(node, ast.Import):
            for imported_module in node.names:
                module = importlib.import_module(imported_module.name)
                if imported_module.name == "amnesiac" or imported_module.name.startswith(
                    "amnesiac."
                ):
                    if imported_module.asname:
                        imported_amnesiac_modules[imported_module.asname] = module
                    else:
                        root_name = imported_module.name.partition(".")[0]
                        imported_amnesiac_modules[root_name] = importlib.import_module(root_name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = _attribute_chain(node)
        if parts is None or parts[0] not in imported_amnesiac_modules:
            continue
        value: object = imported_amnesiac_modules[parts[0]]
        for part in parts[1:]:
            value = getattr(value, part)


def test_python_blocks_are_valid_syntax() -> None:
    blocks = _python_blocks()
    assert blocks
    for block in blocks:
        compile(block, "<readme>", "exec")


def test_amnesiac_names_resolve() -> None:
    blocks = _python_blocks()
    assert blocks
    for block in blocks:
        _assert_imported_names_resolve(block)


def test_imported_attribute_names_resolve() -> None:
    source = """
from amnesiac.summarize import summarize_axes
import amnesiac.summarize

amnesiac.summarize.summarize_meta
"""
    _assert_imported_names_resolve(source)

    invalid_source = source.replace("summarize_meta", "no_such_name")
    with pytest.raises(AttributeError):
        _assert_imported_names_resolve(invalid_source)
