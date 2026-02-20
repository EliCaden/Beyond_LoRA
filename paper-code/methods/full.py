# methods/full.py
from __future__ import annotations

import torch.nn as nn


class FullFineTune:
    """
    Full fine-tuning: make the *entire* model trainable.

    Correctness guard:
      - If LoRA/PaCA adapters are still attached, "full" can silently be incorrect:
          * LoRA: detach_base=True blocks base grads.
          * PaCA: base weights are frozen by design.
      - Default behavior here is to MERGE+STRIP adapters before unfreezing everything.
        This makes "full" a true baseline.
    """

    def __init__(self, lr: float = 2e-5, weight_decay: float = 0.01, *, merge_and_strip_adapters: bool = True):
        self.cfg = type("cfg", (), {"lr": lr, "weight_decay": weight_decay})
        self.merge_and_strip_adapters = bool(merge_and_strip_adapters)

    def _merge_and_strip_any_adapters(self, model: nn.Module) -> None:
        # ---- LoRA: merge + strip if available ----
        try:
            from models.lora_recipes import merge_and_strip_all_lora
        except Exception:
            merge_and_strip_all_lora = None

        if callable(merge_and_strip_all_lora):
            try:
                merge_and_strip_all_lora(model)
            except Exception as e:
                raise RuntimeError(f"FullFineTune: failed to merge/strip LoRA adapters: {e}") from e

        # ---- PaCA: merge into base and strip wrapper nodes (no global helper in your chunk) ----
        try:
            from models.paca_layers import PaCALinearAdapter
        except Exception:
            PaCALinearAdapter = None  # type: ignore[assignment]

        if PaCALinearAdapter is None:
            return

        def _strip_paca(module: nn.Module) -> int:
            cnt = 0
            for name, child in list(module.named_children()):
                if isinstance(child, PaCALinearAdapter):
                    # Ensure base weight reflects latest adapter params
                    child.merge_into_base(zero_adapter=False, disable_patching=True)
                    setattr(module, name, child.base)
                    cnt += 1
                else:
                    cnt += _strip_paca(child)
            return cnt

        _strip_paca(model)

    def configure_model(self, model: nn.Module):
        if self.merge_and_strip_adapters:
            self._merge_and_strip_any_adapters(model)

        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True

        # If user opted out of stripping adapters, at least ensure LoRA base grads aren't blocked.
        if not self.merge_and_strip_adapters:
            try:
                from models.lora_layers import LoRALinearAdapter
            except Exception:
                LoRALinearAdapter = None  # type: ignore[assignment]

            if LoRALinearAdapter is not None:
                for m in model.modules():
                    if isinstance(m, LoRALinearAdapter):
                        m.detach_base = False  # allow base grads if base params are trainable

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)