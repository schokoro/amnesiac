"""Pure vector math for document selection; requires the select extra."""

from importlib.util import find_spec

if find_spec("numpy") is None:
    raise ImportError(
        "amnesiac.select requires numpy, which is not installed. "
        "Install it with: pip install 'amnesiac[select]'"
    )

from amnesiac.select.core import select_by_axis

__all__ = ["select_by_axis"]
