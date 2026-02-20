# methods/lora_only_freeze_head.py
from __future__ import annotations

import torch.nn as nn

from models.lora_layers import LoRALinearAdapter
from models.lora_recipes import apply_recipe
from utils.targets import normalize_targets
from utils.injectors import inject_lora_fallback


class LoRAOnlyFreezeHead:
    """
    Train only LoRA modules while keeping classifier head frozen.

    Trainable params:
      - LoRA adapter params (A/B) according to recipe (e.g., asym_a freezes A, etc.)
    Frozen:
      - everything else, including head(s)
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        recipe: str = "base",
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        target_modules=None,  # optional, defaults to q,v behavior
    ):
        self.r = int(r)
        self.alpha = int(alpha)
        self.recipe = str(recipe).lower()
        self.target_modules = target_modules
        self.cfg = type("cfg", (), {"lr": lr, "weight_decay": weight_decay})

    def _freeze_head(self, model: nn.Module) -> None:
        net = getattr(model, "net", model)
        for attr in ("classifier", "head", "lm_head"):
            m = getattr(net, attr, None)
            if callable(m):
                try:
                    m = m()
                except Exception:
                    m = None
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

    def _call_add_lora(self, model: nn.Module, *, targets):
        fn = getattr(model, "add_lora", None)
        if not callable(fn):
            return None
        # Prefer normalized targets for consistency with validation/fallback
        try:
            return int(fn(r=self.r, alpha=self.alpha, target_modules=targets))
        except TypeError:
            # Backward compatibility: some implementations may want raw input
            return int(fn(r=self.r, alpha=self.alpha, target_modules=self.target_modules))

    def configure_model(self, model: nn.Module):
        if self.r < 1:
            raise ValueError(f"LoRAOnlyFreezeHead: r must be >= 1, got {self.r}")

        # 1) Freeze everything
        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        targets = normalize_targets(self.target_modules, default=("q", "v"), allow_head=False)

        # 2) Inject LoRA
        injected = None
        injected = self._call_add_lora(model, targets=targets)

        if injected is None:
            if hasattr(model, "add_lora_qv") and callable(getattr(model, "add_lora_qv")) and targets == ["q", "v"]:
                injected = int(model.add_lora_qv(r=self.r, alpha=self.alpha))
            else:
                injected = int(
                    inject_lora_fallback(
                        model,
                        AdapterCls=LoRALinearAdapter,
                        r=self.r,
                        alpha=self.alpha,
                        targets=targets,
                    )
                )

        if injected <= 0:
            raise RuntimeError(
                f"LoRAOnlyFreezeHead: injected 0 adapters for targets={targets} "
                f"(raw target_modules={self.target_modules!r}, model={type(model).__name__})"
            )

        # 3) Apply recipe (must succeed; apply_recipe will raise on unknown recipe)
        apply_recipe(model, self.recipe)

        # 4) Ensure head remains frozen
        self._freeze_head(model)

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)