# methods/head_only.py
from __future__ import annotations

import torch.nn as nn


class HeadOnlyFineTune:
    """
    Fine-tune only the classification head(s).

    - Freezes all parameters first.
    - Then unfreezes head modules.
    - Works best with wrappers that expose .freeze_all() and .unfreeze_heads().
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.01):
        self.cfg = type("cfg", (), {"lr": lr, "weight_decay": weight_decay})

    def configure_model(self, model: nn.Module):
        # Freeze everything
        if hasattr(model, "freeze_all") and callable(getattr(model, "freeze_all")):
            model.freeze_all()
        else:
            for p in model.parameters():
                p.requires_grad = False

        # Unfreeze only heads / classifier
        if hasattr(model, "unfreeze_heads") and callable(getattr(model, "unfreeze_heads")):
            model.unfreeze_heads()
            return

        # Fallbacks: try common patterns
        net = getattr(model, "net", model)

        head = getattr(net, "classifier", None)
        if callable(head):  # e.g., wrapper with classifier() -> nn.Module
            try:
                head = head()
            except Exception:
                head = None

        if isinstance(head, nn.Module):
            for p in head.parameters():
                p.requires_grad = True
            return

        # last resort: try ".head"
        head2 = getattr(net, "head", None)
        if isinstance(head2, nn.Module):
            for p in head2.parameters():
                p.requires_grad = True

    def parameters(self, model: nn.Module):
        return (p for p in model.parameters() if p.requires_grad)