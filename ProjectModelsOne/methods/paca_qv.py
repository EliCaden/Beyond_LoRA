# methods/paca_qv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch.nn as nn

from models.paca_recipes import apply_paca_recipe, is_paca_recipe
from utils.targets import normalize_targets
from utils.injectors import unfreeze_head_modules, inject_paca_fallback


@dataclass
class PaCAConfig:
    r: int = 8
    alpha: int = 16
    seed: int = 0
    k_per_row: Optional[int] = None
    recipe: Optional[str] = None
    lr: float = 5e-4
    weight_decay: float = 0.0
    train_head: bool = False
    train_head_only: bool = False
    target_modules: Sequence[str] | str | None = None


def _call_add_paca(model: nn.Module, *, cfg: PaCAConfig, targets):
    fn = getattr(model, "add_paca", None)
    if not callable(fn):
        return None
    try:
        return int(
            fn(
                r=cfg.r,
                alpha=cfg.alpha,
                seed=cfg.seed,
                k_per_row=cfg.k_per_row,
                target_modules=targets,  # prefer normalized
            )
        )
    except TypeError:
        return int(
            fn(
                r=cfg.r,
                alpha=cfg.alpha,
                seed=cfg.seed,
                k_per_row=cfg.k_per_row,
                target_modules=cfg.target_modules,  # raw fallback
            )
        )


class PaCAQV:
    """
    PaCA on selected targets (default: q,v).

    Trainable:
      - paca_cols_weight for injected adapters
      - optionally classifier head weights (train_head=True)
    Frozen:
      - everything else
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        seed: int = 0,
        k_per_row: int | None = None,
        recipe: str | None = None,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        train_head: bool = False,
        train_head_only: bool = False,
        target_modules: Sequence[str] | str | None = None,
    ):
        self.cfg = PaCAConfig(
            r=int(r),
            alpha=int(alpha),
            seed=int(seed),
            k_per_row=k_per_row,
            recipe=recipe,
            lr=float(lr),
            weight_decay=float(weight_decay),
            train_head=bool(train_head),
            train_head_only=bool(train_head_only),
            target_modules=target_modules,
        )

    def configure_model(self, model: nn.Module):
        if self.cfg.r < 1:
            raise ValueError(f"PaCAQV: r must be >= 1, got {self.cfg.r}")

        # Validate flag combinations
        if self.cfg.train_head_only:
            if self.cfg.recipe is not None:
                raise ValueError("PaCAQV: train_head_only=True cannot be combined with a PaCA recipe.")
            # train_head is redundant in head-only mode; disallow to avoid silent confusion
            if self.cfg.train_head:
                raise ValueError("PaCAQV: train_head_only=True implies head training; set train_head=False.")
            # 1) Freeze everything
            if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
                model.freeze_all()
            else:
                for p in model.parameters():
                    p.requires_grad = False
            # 2) Unfreeze heads only
            if hasattr(model, "unfreeze_heads") and callable(getattr(model, "unfreeze_heads")):
                model.unfreeze_heads()
            else:
                unfreeze_head_modules(model)
            return

        # If recipe provided, require it to be valid (catch typos)
        if self.cfg.recipe is not None and not is_paca_recipe(self.cfg.recipe):
            raise ValueError(
                f"PaCAQV: unknown PaCA recipe {self.cfg.recipe!r}. Allowed: 'dpaca', 'cpaca', 'dcpaca' (or None)."
            )

        # 1) Freeze everything
        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        targets = normalize_targets(self.cfg.target_modules, default=("q", "v"), allow_head=False)

        # 2) Insert PaCA adapters on selected targets
        injected = _call_add_paca(model, cfg=self.cfg, targets=targets)

        if injected is None:
            if hasattr(model, "add_paca_qv") and callable(getattr(model, "add_paca_qv")) and targets == ["q", "v"]:
                injected = int(
                    model.add_paca_qv(
                        r=self.cfg.r,
                        alpha=self.cfg.alpha,
                        seed=self.cfg.seed,
                        k_per_row=self.cfg.k_per_row,
                    )
                )
            else:
                injected = int(
                    inject_paca_fallback(
                        model,
                        r=self.cfg.r,
                        alpha=self.cfg.alpha,
                        seed=self.cfg.seed,
                        k_per_row=self.cfg.k_per_row,
                        targets=targets,
                    )
                )

        if injected <= 0:
            raise RuntimeError(
                f"PaCAQV: injected 0 adapters for targets={targets} "
                f"(raw target_modules={self.cfg.target_modules!r}, model={type(model).__name__})"
            )

        # 3) Apply PaCA recipe if requested (validated above)
        if self.cfg.recipe is not None:
            apply_paca_recipe(model, self.cfg.recipe, seed=self.cfg.seed, offset_step=0)

        # 4) Optionally train classifier head too
        if self.cfg.train_head:
            if hasattr(model, "unfreeze_heads") and callable(getattr(model, "unfreeze_heads")):
                model.unfreeze_heads()
            else:
                unfreeze_head_modules(model)

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)