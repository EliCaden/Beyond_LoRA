# methods/__init__.py
from __future__ import annotations

"""
Public API for the `methods` package.

- Core method classes are imported eagerly (so any registration/side-effects run).
- Optional helpers (LoRA/PaCA adapters + recipe utilities) are imported defensively.
  We catch ImportError only (so real bugs inside those modules are not silently hidden).
"""

# -----------------------------
# Optional re-exports (helpers)
# -----------------------------
try:
    from models.lora_layers import LoRALinearAdapter
except ImportError:
    LoRALinearAdapter = None  # type: ignore[assignment]

try:
    from models.lora_recipes import (
        apply_recipe,
        chain_step,
        merge_all_lora,
        strip_all_lora,
        merge_and_strip_all_lora,
        finalize_ba_sparse_all,
        find_lora_adapters,
        notify_optimizer_step_all,
    )
except ImportError:
    apply_recipe = None  # type: ignore[assignment]
    chain_step = None  # type: ignore[assignment]
    merge_all_lora = None  # type: ignore[assignment]
    strip_all_lora = None  # type: ignore[assignment]
    merge_and_strip_all_lora = None  # type: ignore[assignment]
    finalize_ba_sparse_all = None  # type: ignore[assignment]
    find_lora_adapters = None  # type: ignore[assignment]
    notify_optimizer_step_all = None  # type: ignore[assignment]

# PaCA helpers are optional (depending on whether PaCA is included in the install)
try:
    from models.paca_layers import PaCALinearAdapter
except ImportError:
    PaCALinearAdapter = None  # type: ignore[assignment]

try:
    from models.paca_recipes import (
        is_paca_recipe,
        has_paca_adapters,
        find_paca_adapters,
        merge_all_paca,
        apply_paca_recipe,
        paca_chain_step,
        after_optimizer_step_all,  # PaCA post-step patch hook
    )
except ImportError:
    is_paca_recipe = None  # type: ignore[assignment]
    has_paca_adapters = None  # type: ignore[assignment]
    find_paca_adapters = None  # type: ignore[assignment]
    merge_all_paca = None  # type: ignore[assignment]
    apply_paca_recipe = None  # type: ignore[assignment]
    paca_chain_step = None  # type: ignore[assignment]
    after_optimizer_step_all = None  # type: ignore[assignment]

# -----------------------------
# Methods in this package
# -----------------------------
from .full import FullFineTune
from .head_only import HeadOnlyFineTune
from .lora_qv import LoRAQV, LoRAQVLoRAOnly, LoRAQVWithHead
from .lora_only_freeze_head import LoRAOnlyFreezeHead

# PaCA method(s) are optional
try:
    from .paca_qv import PaCAConfig, PaCAQV
except ImportError:
    PaCAConfig = None  # type: ignore[assignment]
    PaCAQV = None  # type: ignore[assignment]

__all__ = [
    # LoRA helpers (optional)
    "LoRALinearAdapter",
    "apply_recipe",
    "chain_step",
    "merge_all_lora",
    "strip_all_lora",
    "merge_and_strip_all_lora",
    "finalize_ba_sparse_all",
    "find_lora_adapters",
    "notify_optimizer_step_all",
    # PaCA helpers (optional)
    "PaCALinearAdapter",
    "is_paca_recipe",
    "has_paca_adapters",
    "find_paca_adapters",
    "merge_all_paca",
    "apply_paca_recipe",
    "paca_chain_step",
    "after_optimizer_step_all",
    # methods
    "FullFineTune",
    "HeadOnlyFineTune",
    "LoRAQV",
    "LoRAQVLoRAOnly",
    "LoRAQVWithHead",
    "LoRAOnlyFreezeHead",
    "PaCAConfig",
    "PaCAQV",
]