# models/vit_classifier.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from data import get_data
from .lora_layers import LoRALinearAdapter

_MODEL_MAP = {
    "tiny": "facebook/deit-tiny-patch16-224",
    "base": "google/vit-base-patch16-224",
    "large": "google/vit-large-patch16-224",
    "xxl": "google/vit-huge-patch14-224-in21k",
}

_PRETRAIN_VARIANT_TO_DATASET = {
    "tiny": "imagenet1k",
    "base": "imagenet1k",
    "large": "imagenet1k",
    "xxl": "imagenet21k",
}


def resolve_pretrained_id(variant: str) -> str:
    return _MODEL_MAP.get(str(variant).lower(), _MODEL_MAP["base"])


def _resolve_classes(
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    classes: Optional[int] = None,
    n_classes: Optional[int] = None,
) -> int:
    for cand in (num_classes, num_labels, classes, n_classes):
        if cand is not None:
            return int(cand)
    raise ValueError("ViTImageModel: one of {num_classes, num_labels, classes, n_classes} must be provided.")


class ViTImageModel(nn.Module):
    """
    HF ViT image classifier wrapper.

    Key behavior:
      - Caches lists of LoRA/PaCA adapters (for methods that need fast access).
      - External injectors must call mark_lora_dirty / mark_paca_dirty / mark_adapters_dirty.
    """

    def __init__(
        self,
        variant: str = "base",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        classes: Optional[int] = None,
        n_classes: Optional[int] = None,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        ignore_mismatched_sizes: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.variant = str(variant).lower()
        model_id = resolve_pretrained_id(self.variant)
        cls_count = _resolve_classes(num_classes, num_labels, classes, n_classes)

        env = os.environ.get("VIT_PRETRAINED", "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            pretrained = False
        elif env in {"1", "true", "yes", "on"}:
            pretrained = True

        if pretrained:
            init_kwargs: dict[str, Any] = {
                "num_labels": cls_count,
                "ignore_mismatched_sizes": ignore_mismatched_sizes,
            }
            if id2label is not None:
                init_kwargs["id2label"] = id2label
            if label2id is not None:
                init_kwargs["label2id"] = label2id
            init_kwargs.update(kwargs)
            self.net = AutoModelForImageClassification.from_pretrained(model_id, **init_kwargs)
        else:
            cfg = AutoConfig.from_pretrained(model_id)
            cfg.num_labels = cls_count
            if id2label is not None:
                cfg.id2label = id2label
            if label2id is not None:
                cfg.label2id = label2id
            for k, v in kwargs.items():
                setattr(cfg, k, v)
            self.net = AutoModelForImageClassification.from_config(cfg)

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except TypeError:
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)

        self.num_classes = cls_count

        self._lora_cache: list[LoRALinearAdapter] = []
        self._paca_cache: list[nn.Module] = []
        self._lora_cache_valid = False
        self._paca_cache_valid = False

    def forward(self, *args, **kwargs):
        # convenience aliases
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            kwargs["pixel_values"] = args[0]
        if "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
            kwargs["pixel_values"] = kwargs.pop("x")
        if "images" in kwargs and isinstance(kwargs["images"], torch.Tensor):
            kwargs["pixel_values"] = kwargs.pop("images")
        return self.net(**kwargs)

    # ---- dirty hooks (external injectors rely on these) ----
    def mark_lora_dirty(self) -> None:
        self._lora_cache_valid = False

    def mark_paca_dirty(self) -> None:
        self._paca_cache_valid = False

    def mark_adapters_dirty(self) -> None:
        self._lora_cache_valid = False
        self._paca_cache_valid = False

    # ---- caching helpers ----
    def _refresh_lora_cache(self) -> None:
        self._lora_cache = [m for m in self.modules() if isinstance(m, LoRALinearAdapter)]
        self._lora_cache_valid = True

    def _refresh_paca_cache(self) -> None:
        try:
            from .paca_layers import PaCALinearAdapter
        except ImportError:
            self._paca_cache = []
            self._paca_cache_valid = True
            return
        self._paca_cache = [m for m in self.modules() if isinstance(m, PaCALinearAdapter)]
        self._paca_cache_valid = True

    def lora_modules(self) -> list[LoRALinearAdapter]:
        if not self._lora_cache_valid:
            self._refresh_lora_cache()
        return self._lora_cache

    def paca_modules(self) -> list[nn.Module]:
        if not self._paca_cache_valid:
            self._refresh_paca_cache()
        return self._paca_cache

    # ---- helpers used by methods ----
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_heads(self):
        head = getattr(self.net, "classifier", None)
        if isinstance(head, nn.Module):
            for p in head.parameters():
                p.requires_grad = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _backbone_and_layers(self):
        bb = getattr(self.net, "vit", None) or getattr(self.net, "deit", None)
        if bb is None:
            return None, None
        enc = getattr(bb, "encoder", None)
        layers = getattr(enc, "layer", None)
        return bb, layers

    @torch.no_grad()
    def add_paca_qv(self, r: int = 8, alpha: int = 16, *, seed: int = 0, k_per_row: int | None = None) -> int:
        from .paca_layers import PaCALinearAdapter

        n = 0
        _, layers = self._backbone_and_layers()
        if layers is None:
            return 0

        for i, blk in enumerate(layers):
            attn_wrap = getattr(blk, "attention", None)
            if attn_wrap is None:
                continue
            attn = getattr(attn_wrap, "attention", None)
            if attn is None:
                continue

            q = getattr(attn, "query", None)
            v = getattr(attn, "value", None)

            if isinstance(q, nn.Linear) and not isinstance(q, PaCALinearAdapter):
                attn.query = PaCALinearAdapter(q, r=r, alpha=alpha, seed=int(seed + 1000 * i + 1), k_per_row=k_per_row)
                n += 1
            if isinstance(v, nn.Linear) and not isinstance(v, PaCALinearAdapter):
                attn.value = PaCALinearAdapter(v, r=r, alpha=alpha, seed=int(seed + 1000 * i + 2), k_per_row=k_per_row)
                n += 1

        self.mark_paca_dirty()
        self._refresh_paca_cache()
        return n

    @torch.no_grad()
    def add_lora_qv(self, r: int = 8, alpha: int = 16) -> int:
        n = 0
        _, layers = self._backbone_and_layers()
        if layers is None:
            return 0

        for blk in layers:
            attn_wrap = getattr(blk, "attention", None)
            if attn_wrap is None:
                continue
            attn = getattr(attn_wrap, "attention", None)
            if attn is None:
                continue

            q = getattr(attn, "query", None)
            v = getattr(attn, "value", None)

            if isinstance(q, nn.Linear) and not isinstance(q, LoRALinearAdapter):
                attn.query = LoRALinearAdapter(q, r=r, alpha=alpha)
                n += 1
            if isinstance(v, nn.Linear) and not isinstance(v, LoRALinearAdapter):
                attn.value = LoRALinearAdapter(v, r=r, alpha=alpha)
                n += 1

        self.mark_lora_dirty()
        self._refresh_lora_cache()
        return n

    @torch.no_grad()
    def add_lora_head(self, r: int = 8, alpha: int = 16) -> int:
        head = getattr(self.net, "classifier", None)
        if isinstance(head, nn.Linear) and not isinstance(head, LoRALinearAdapter):
            self.net.classifier = LoRALinearAdapter(head, r=r, alpha=alpha)
            self.mark_lora_dirty()
            self._refresh_lora_cache()
            return 1
        return 0

    def classifier(self) -> Optional[nn.Module]:
        return getattr(self.net, "classifier", None)

    def backbone(self) -> Optional[nn.Module]:
        return getattr(self.net, "vit", None) or getattr(self.net, "deit", None)

    # -------------------------------------------------------------------------
    # Helpers to choose & build a "pretrain" dataset/dataloader
    # -------------------------------------------------------------------------

    def _infer_img_size_and_norm(self) -> Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]:
        proc = getattr(self, "image_processor", None)
        img_size = 224
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        if proc is not None:
            size = getattr(proc, "size", None)
            if isinstance(size, int):
                img_size = int(size)
            elif isinstance(size, dict) and size:
                if "shortest_edge" in size:
                    img_size = int(size["shortest_edge"])
                else:
                    try:
                        img_size = int(next(iter(size.values())))
                    except Exception:
                        pass
            elif isinstance(size, (tuple, list)) and len(size) > 0:
                img_size = int(size[-1])

            try:
                if hasattr(proc, "image_mean"):
                    mean = tuple(float(x) for x in proc.image_mean)
            except Exception:
                pass
            try:
                if hasattr(proc, "image_std"):
                    std = tuple(float(x) for x in proc.image_std)
            except Exception:
                pass

        return img_size, mean, std

    def _resolve_pretrain_dataset_name(self) -> str:
        v = getattr(self, "variant", "base")
        return _PRETRAIN_VARIANT_TO_DATASET.get(str(v).lower(), "caltech101")

    def return_pretrained_dataset(
        self,
        data_root: str,
        *,
        batch_size: int = 128,
        seed: int = 0,
        num_workers: Optional[int] = None,
        split: str = "test",
        download: bool = True,
        **data_kwargs: Any,
    ) -> DataLoader:
        dataset_name = self._resolve_pretrain_dataset_name()
        img_size, mean, std = self._infer_img_size_and_norm()

        base_cfg: dict[str, Any] = dict(
            data_root=data_root,
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=mean,
            std=std,
            download=download,
        )

        if dataset_name == "imagenet1k":
            base_cfg["train_dir_name"] = "imagenet-val"
            base_cfg["val_dir_name"] = "imagenet-val"
            base_cfg["test_dir_name"] = "imagenet-val"
            base_cfg["download"] = False

        base_cfg.update(data_kwargs)

        try:
            dm = get_data(dataset_name, **base_cfg)
        except TypeError as e:
            print(f"[WARN] ViTImageModel.return_pretrained_dataset: retrying '{dataset_name}' init without extras due to: {e}")
            minimal_cfg: dict[str, Any] = {
                "data_root": data_root,
                "batch_size": batch_size,
                "img_size": img_size,
                "seed": seed,
                "download": False if dataset_name == "imagenet1k" else download,
            }
            if dataset_name == "imagenet1k":
                minimal_cfg.update(train_dir_name="imagenet-val", val_dir_name="imagenet-val", test_dir_name="imagenet-val")
            dm = get_data(dataset_name, **minimal_cfg)

        dm.setup()
        s = split.lower()
        if s == "train":
            return dm.train_dataloader()
        if s == "val":
            return dm.val_dataloader()
        return dm.test_dataloader()