# models/__init__.py
from __future__ import annotations

from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec

# -----------------------------
# LoRA layers + recipe helpers
# -----------------------------
from .lora_layers import LoRALinearAdapter
from .lora_recipes import (
    find_lora_adapters,
    merge_all_lora,
    strip_all_lora,
    merge_and_strip_all_lora,
    apply_recipe,
    chain_step,
    finalize_ba_sparse_all,
    notify_optimizer_step_all,
)

# -----------------------------
# PaCA layers + recipe helpers (optional)
# -----------------------------
try:
    from .paca_layers import PaCALinearAdapter, cpaca_merge_and_replace_
    from .paca_recipes import (
        is_paca_recipe,
        find_paca_adapters,
        has_paca_adapters,
        merge_all_paca,
        apply_paca_recipe,
        paca_chain_step,
        after_optimizer_step_all,
    )
except ImportError:
    PaCALinearAdapter = None  # type: ignore[assignment]
    cpaca_merge_and_replace_ = None  # type: ignore[assignment]
    is_paca_recipe = None  # type: ignore[assignment]
    find_paca_adapters = None  # type: ignore[assignment]
    has_paca_adapters = None  # type: ignore[assignment]
    merge_all_paca = None  # type: ignore[assignment]
    apply_paca_recipe = None  # type: ignore[assignment]
    paca_chain_step = None  # type: ignore[assignment]
    after_optimizer_step_all = None  # type: ignore[assignment]

# -----------------------------
# Lazy model wrappers (Transformers-heavy)
# -----------------------------
_LAZY = {
    "ViTImageModel": (".vit_classifier", "ViTImageModel"),
    "RobertaGLUEModel": (".roberta", "RobertaGLUEModel"),
}

def __getattr__(name: str):
    if name in _LAZY:
        mod_rel, attr = _LAZY[name]
        mod_name = __name__ + mod_rel
        if _find_spec(mod_name) is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        mod = _import_module(mod_name)
        val = getattr(mod, attr)
        globals()[name] = val  # cache
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # LoRA
    "LoRALinearAdapter",
    "find_lora_adapters",
    "merge_all_lora",
    "strip_all_lora",
    "merge_and_strip_all_lora",
    "apply_recipe",
    "chain_step",
    "finalize_ba_sparse_all",
    "notify_optimizer_step_all",
    # PaCA (optional)
    "PaCALinearAdapter",
    "cpaca_merge_and_replace_",
    "is_paca_recipe",
    "find_paca_adapters",
    "has_paca_adapters",
    "merge_all_paca",
    "apply_paca_recipe",
    "paca_chain_step",
    "after_optimizer_step_all",
    # Models (lazy)
    "ViTImageModel",
    "RobertaGLUEModel",
]