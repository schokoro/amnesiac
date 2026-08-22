"""Package root without subsystem re-exports.

Subsystem re-exports would make ``import amnesiac`` depend on optional extras.
"""

from .exceptions import AmnesiacError, ConfigurationError
from .types import Doc

__all__ = ["AmnesiacError", "ConfigurationError", "Doc"]
