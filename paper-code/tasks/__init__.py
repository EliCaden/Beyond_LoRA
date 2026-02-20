# tasks/__init__.py
from __future__ import annotations

from .glue_text import GlueTextTask
from .vision_image import VisionClassificationTask

__all__ = [
    "GlueTextTask",
    "VisionClassificationTask",
]

# Optional: only available if tasks/causal_lm.py exists in this repo.
# Catch ImportError so real bugs in causal_lm aren't silently hidden.
try:
    from .causal_lm import CausalLMTask  # type: ignore
except ImportError:
    CausalLMTask = None  # type: ignore
else:
    __all__.append("CausalLMTask")