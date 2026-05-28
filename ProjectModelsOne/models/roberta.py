# models/roberta.py
from __future__ import annotations

from typing import Any, List, Optional

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from .lora_layers import LoRALinearAdapter
from .lora_recipes import merge_and_strip_all_lora

_MODEL_MAP = {"base": "roberta-base", "large": "roberta-large"}

_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "rte": 2,
    "qqp": 2,
    "qnli": 2,
    "wnli": 2,
    "mnli": 3,
    "stsb": 1,
}


def resolve_pretrained_id(variant: str) -> str:
    v = str(variant).lower().strip()
    return _MODEL_MAP.get(v, _MODEL_MAP["base"])


class RobertaGLUEModel(nn.Module):
    """
    Notes on adapter caching:

    - Recipes / methods often want "all LoRA modules" repeatedly.
    - Scanning `self.net.modules()` is expensive if done per-step.
    - We keep a cache and only rescan when the model's adapter topology changes.

    Cache invalidation policy:
      - add_* methods set dirty and force-refresh
      - merge/strip sets dirty and clears cache
      - methods that mutate adapters directly can call mark_*_dirty()
    """

    def __init__(
        self,
        task: str,
        variant: str = "base",
        pretrained: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.task = str(task).lower().strip()
        if self.task not in _NUM_LABELS:
            raise ValueError(f"Unknown GLUE task: {task}")
        self.variant = str(variant).lower().strip()

        model_id = resolve_pretrained_id(self.variant)
        num_labels = int(_NUM_LABELS[self.task])

        # Env override for pretrained toggle
        env = os.environ.get("ROBERTA_PRETRAINED", "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            pretrained = False
        elif env in {"1", "true", "yes", "on"}:
            pretrained = True

        # Optional: Transformers attention implementation (newer versions)
        # - Can pass attn_implementation=... via kwargs, or set ROBERTA_ATTN_IMPL.
        attn_impl = kwargs.pop("attn_implementation", None)
        env_attn = os.environ.get("ROBERTA_ATTN_IMPL", "").strip()
        if env_attn:
            attn_impl = env_attn

        init_kwargs: dict[str, Any] = {"num_labels": num_labels}
        if self.task == "stsb":
            init_kwargs["problem_type"] = "regression"
        init_kwargs.update(kwargs)

        if pretrained:
            # Some transformers versions accept `attn_implementation`, some don't.
            if attn_impl:
                try:
                    self.net = AutoModelForSequenceClassification.from_pretrained(
                        model_id, attn_implementation=attn_impl, **init_kwargs
                    )
                except TypeError:
                    self.net = AutoModelForSequenceClassification.from_pretrained(model_id, **init_kwargs)
            else:
                self.net = AutoModelForSequenceClassification.from_pretrained(model_id, **init_kwargs)
        else:
            cfg = AutoConfig.from_pretrained(model_id)
            cfg.num_labels = num_labels
            if self.task == "stsb":
                cfg.problem_type = "regression"
            self.net = AutoModelForSequenceClassification.from_config(cfg)

        self.num_labels = num_labels

        # Adapter caches (real caches, not "refresh every call")
        self._lora_cache: List[LoRALinearAdapter] = []
        self._paca_cache: list[nn.Module] = []
        self._lora_cache_dirty: bool = True
        self._paca_cache_dirty: bool = True

    def forward(self, **kwargs):
        return self.net(**kwargs)

    # ---------------------------
    # cache invalidation helpers
    # ---------------------------
    def mark_lora_dirty(self) -> None:
        self._lora_cache_dirty = True

    def mark_paca_dirty(self) -> None:
        self._paca_cache_dirty = True

    def mark_adapters_dirty(self) -> None:
        self._lora_cache_dirty = True
        self._paca_cache_dirty = True

    def _refresh_lora_cache(self, *, force: bool = False) -> None:
        if not (force or self._lora_cache_dirty):
            return
        self._lora_cache = [m for m in self.net.modules() if isinstance(m, LoRALinearAdapter)]
        self._lora_cache_dirty = False

    def _refresh_paca_cache(self, *, force: bool = False) -> None:
        if not (force or self._paca_cache_dirty):
            return
        try:
            from .paca_layers import PaCALinearAdapter
        except Exception:
            self._paca_cache = []
            self._paca_cache_dirty = False
            return
        self._paca_cache = [m for m in self.net.modules() if isinstance(m, PaCALinearAdapter)]
        self._paca_cache_dirty = False

    def lora_modules(self, *, refresh: bool = False) -> List[LoRALinearAdapter]:
        self._refresh_lora_cache(force=refresh)
        return self._lora_cache

    def paca_modules(self, *, refresh: bool = False) -> list[nn.Module]:
        self._refresh_paca_cache(force=refresh)
        return self._paca_cache

    # ---------------------------
    # helpers used by methods
    # ---------------------------
    def freeze_all(self):
        for p in self.net.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.net.parameters():
            p.requires_grad = True

    def unfreeze_heads(self):
        head = getattr(self.net, "classifier", None)
        if not isinstance(head, nn.Module):
            raise RuntimeError("Expected RoBERTa classification head at .classifier")
        for p in head.parameters():
            p.requires_grad = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # convenience alias used elsewhere in the repo
        return cls(*args, **kwargs)

    def classifier(self) -> Optional[nn.Module]:
        return getattr(self.net, "classifier", None)

    def backbone(self) -> Optional[nn.Module]:
        return getattr(self.net, "roberta", None)

    # ---------------------------
    # internal traversal
    # ---------------------------
    def _iter_self_attention(self):
        try:
            layers = self.net.roberta.encoder.layer
        except AttributeError as e:
            raise RuntimeError("Expected .roberta.encoder.layer on RoBERTa") from e
        for layer in layers:
            yield layer.attention.self

    # ---------------------------
    # adapter injection
    # ---------------------------
    @torch.no_grad()
    def add_lora_qv(self, r: int = 8, alpha: int = 16) -> int:
        replaced = 0
        for sa in self._iter_self_attention():
            if isinstance(sa.query, nn.Linear) and not isinstance(sa.query, LoRALinearAdapter):
                sa.query = LoRALinearAdapter(sa.query, r=r, alpha=alpha)
                replaced += 1
            if isinstance(sa.value, nn.Linear) and not isinstance(sa.value, LoRALinearAdapter):
                sa.value = LoRALinearAdapter(sa.value, r=r, alpha=alpha)
                replaced += 1
        self._lora_cache_dirty = True
        self._refresh_lora_cache(force=True)
        return replaced

    @torch.no_grad()
    def add_paca_qv(
        self,
        r: int = 8,
        alpha: int = 16,
        *,
        seed: int = 0,
        k_per_row: int | None = None,
    ) -> int:
        from .paca_layers import PaCALinearAdapter

        replaced = 0
        for i, sa in enumerate(self._iter_self_attention()):
            q_seed = int(seed + 1000 * i + 1)
            v_seed = int(seed + 1000 * i + 2)

            if isinstance(sa.query, nn.Linear) and not isinstance(sa.query, PaCALinearAdapter):
                sa.query = PaCALinearAdapter(sa.query, r=r, alpha=alpha, seed=q_seed, k_per_row=k_per_row)
                replaced += 1
            if isinstance(sa.value, nn.Linear) and not isinstance(sa.value, PaCALinearAdapter):
                sa.value = PaCALinearAdapter(sa.value, r=r, alpha=alpha, seed=v_seed, k_per_row=k_per_row)
                replaced += 1

        self._paca_cache_dirty = True
        self._refresh_paca_cache(force=True)
        return replaced

    @torch.no_grad()
    def merge_and_strip_lora(self) -> int:
        n = merge_and_strip_all_lora(self.net)
        self._lora_cache = []
        self._lora_cache_dirty = True
        return n
