# scripts/__init__.py
"""
scripts package

Provides sweep entrypoints (vision / GLUE / etc.) with lazy submodule loading so
`import scripts` stays lightweight.

Typical usage:
  python -m scripts.sweep_vision ...
  python -m scripts.sweep_glue  ...
"""

from __future__ import annotations

from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from typing import List as _List

# Core modules expected to exist in this package.
_CORE = ["sweep_common", "sweep_glue", "sweep_vision"]

# Optional modules (export only if present).
_OPTIONAL_CANDIDATES = ["sweep_e2e"]

_OPTIONAL = [m for m in _OPTIONAL_CANDIDATES if _find_spec(f"{__name__}.{m}") is not None]

__all__ = _CORE + _OPTIONAL


def __getattr__(name: str):
    # Allow `unittest.mock.patch("scripts.sweep_vision.*")` etc.
    if name in __all__:
        try:
            mod = _import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as e:
            # Convert "missing optional module" into a clean attribute error.
            if e.name in {f"{__name__}.{name}", name}:
                raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
            raise
        globals()[name] = mod  # cache
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> _List[str]:
    return sorted(set(list(globals().keys()) + __all__))
