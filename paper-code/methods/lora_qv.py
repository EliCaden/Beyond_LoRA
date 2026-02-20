# methods/lora_qv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn

from models.lora_layers import LoRALinearAdapter
from models.lora_recipes import apply_recipe
from utils.targets import normalize_targets
from utils.injectors import (
    unfreeze_head_modules,
    inject_lora_fallback,
    inject_lora_roberta_head,
    inject_lora_generic_head,
)


@dataclass
class _Cfg:
    lr: float
    weight_decay: float


def _call_add_lora(model: nn.Module, *, r: int, alpha: int, targets, raw_target_modules):
    fn = getattr(model, "add_lora", None)
    if not callable(fn):
        return None
    try:
        return int(fn(r=r, alpha=alpha, target_modules=targets))
    except TypeError:
        return int(fn(r=r, alpha=alpha, target_modules=raw_target_modules))


def _call_add_lora_head(model: nn.Module, *, r: int, alpha: int, head_modules):
    fn = getattr(model, "add_lora_head", None)
    if not callable(fn):
        return 0
    try:
        return int(fn(r=r, alpha=alpha, target_modules=head_modules))
    except TypeError:
        return int(fn(r=r, alpha=alpha))


def _inject_backbone_lora(model: nn.Module, *, r: int, alpha: int, targets, raw_target_modules) -> int:
    injected = _call_add_lora(model, r=r, alpha=alpha, targets=targets, raw_target_modules=raw_target_modules)
    if injected is not None:
        return injected

    if hasattr(model, "add_lora_qv") and callable(getattr(model, "add_lora_qv")) and targets == ["q", "v"]:
        return int(model.add_lora_qv(r=r, alpha=alpha))

    return int(inject_lora_fallback(model, AdapterCls=LoRALinearAdapter, r=r, alpha=alpha, targets=targets))


class LoRAQV:
    """
    LoRA on selected backbone targets (default: q,v) + train RAW classifier head weights.

    Trainable:
      - head weights/bias (raw, not LoRA-wrapped)
      - LoRA adapter params per recipe
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        recipe: str = "base",
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        mask_alpha: float | None = None,
        mask_ties_keep: bool = True,
        mask_rounding: str = "ceil",
        mask_recompute_every: int = 1,
        target_modules: Sequence[str] | str | None = None,
    ):
        self.r = int(r)
        self.alpha = int(alpha)
        self.recipe = str(recipe).lower()
        self.cfg = _Cfg(lr=float(lr), weight_decay=float(weight_decay))

        self.mask_alpha = mask_alpha
        self.mask_ties_keep = bool(mask_ties_keep)
        self.mask_rounding = str(mask_rounding)
        self.mask_recompute_every = int(mask_recompute_every)

        self.target_modules = target_modules

    def configure_model(self, model: nn.Module):
        if self.r < 1:
            raise ValueError(f"LoRAQV: r must be >= 1, got {self.r}")

        # Freeze backbone; train heads + LoRA
        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        # Train raw classifier head weights/biases
        if hasattr(model, "unfreeze_heads") and callable(getattr(model, "unfreeze_heads")):
            model.unfreeze_heads()
        else:
            unfreeze_head_modules(model)

        targets = normalize_targets(self.target_modules, default=("q", "v"), allow_head=False)

        injected = _inject_backbone_lora(
            model,
            r=self.r,
            alpha=self.alpha,
            targets=targets,
            raw_target_modules=self.target_modules,
        )
        if injected <= 0:
            raise RuntimeError(
                f"LoRAQV: injected 0 adapters for targets={targets} "
                f"(raw target_modules={self.target_modules!r}, model={type(model).__name__})"
            )

        apply_recipe(
            model,
            self.recipe,
            mask_alpha=self.mask_alpha,
            mask_ties_keep=self.mask_ties_keep,
            mask_rounding=self.mask_rounding,
            mask_recompute_every=self.mask_recompute_every,
        )

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)


class LoRAQVLoRAOnly:
    """
    LoRA on selected targets, train ONLY LoRA adapter params (head remains frozen).
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        recipe: str = "base",
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        mask_alpha: float | None = None,
        mask_ties_keep: bool = True,
        mask_rounding: str = "ceil",
        mask_recompute_every: int = 1,
        target_modules: Sequence[str] | str | None = None,
    ):
        self.r = int(r)
        self.alpha = int(alpha)
        self.recipe = str(recipe).lower()
        self.cfg = _Cfg(lr=float(lr), weight_decay=float(weight_decay))

        self.mask_alpha = mask_alpha
        self.mask_ties_keep = bool(mask_ties_keep)
        self.mask_rounding = str(mask_rounding)
        self.mask_recompute_every = int(mask_recompute_every)

        self.target_modules = target_modules

    def configure_model(self, model: nn.Module):
        if self.r < 1:
            raise ValueError(f"LoRAQVLoRAOnly: r must be >= 1, got {self.r}")

        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        targets = normalize_targets(self.target_modules, default=("q", "v"), allow_head=False)

        injected = _inject_backbone_lora(
            model,
            r=self.r,
            alpha=self.alpha,
            targets=targets,
            raw_target_modules=self.target_modules,
        )
        if injected <= 0:
            raise RuntimeError(
                f"LoRAQVLoRAOnly: injected 0 adapters for targets={targets} "
                f"(raw target_modules={self.target_modules!r}, model={type(model).__name__})"
            )

        apply_recipe(
            model,
            self.recipe,
            mask_alpha=self.mask_alpha,
            mask_ties_keep=self.mask_ties_keep,
            mask_rounding=self.mask_rounding,
            mask_recompute_every=self.mask_recompute_every,
        )

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)


class LoRAQVWithHead:
    """
    LoRA on backbone targets AND LoRA-wrap the classifier head linears.

    Trainable:
      - LoRA adapter params on backbone + head (per recipe)
    Frozen:
      - raw head weights (since we never unfreeze heads)
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        recipe: str = "base",
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        mask_alpha: float | None = None,
        mask_ties_keep: bool = True,
        mask_rounding: str = "ceil",
        mask_recompute_every: int = 1,
        target_modules: Sequence[str] | str | None = None,
        head_modules: Sequence[str] | str | None = None,
    ):
        self.r = int(r)
        self.alpha = int(alpha)
        self.recipe = str(recipe).lower()
        self.cfg = _Cfg(lr=float(lr), weight_decay=float(weight_decay))

        self.mask_alpha = mask_alpha
        self.mask_ties_keep = bool(mask_ties_keep)
        self.mask_rounding = str(mask_rounding)
        self.mask_recompute_every = int(mask_recompute_every)

        self.target_modules = target_modules
        self.head_modules = head_modules

    def configure_model(self, model: nn.Module):
        if self.r < 1:
            raise ValueError(f"LoRAQVWithHead: r must be >= 1, got {self.r}")

        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        targets = normalize_targets(self.target_modules, default=("q", "v"), allow_head=False)

        injected = _inject_backbone_lora(
            model,
            r=self.r,
            alpha=self.alpha,
            targets=targets,
            raw_target_modules=self.target_modules,
        )
        if injected <= 0:
            raise RuntimeError(
                f"LoRAQVWithHead: injected 0 backbone adapters for targets={targets} "
                f"(raw target_modules={self.target_modules!r}, model={type(model).__name__})"
            )

        # Head wrapping
        head_added = _call_add_lora_head(model, r=self.r, alpha=self.alpha, head_modules=self.head_modules)
        if head_added <= 0:
            head_added = inject_lora_roberta_head(model, AdapterCls=LoRALinearAdapter, r=self.r, alpha=self.alpha)
            if head_added <= 0:
                head_added = inject_lora_generic_head(model, AdapterCls=LoRALinearAdapter, r=self.r, alpha=self.alpha)

        if head_added <= 0:
            print("[WARN] LoRAQVWithHead: could not wrap any head Linear layers with LoRA")

        apply_recipe(
            model,
            self.recipe,
            mask_alpha=self.mask_alpha,
            mask_ties_keep=self.mask_ties_keep,
            mask_rounding=self.mask_rounding,
            mask_recompute_every=self.mask_recompute_every,
        )

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)