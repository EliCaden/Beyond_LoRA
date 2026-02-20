# data/vtab_helpers.py
from __future__ import annotations

import torch


def _to_rgb(img):
    """Convert PIL Image to RGB to ensure 3 channels."""
    return img.convert("RGB")


def _collate(batch):
    """Collate a batch of (img, label) into dict for trainer."""
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0).contiguous()
    ys = torch.tensor(ys, dtype=torch.long)
    return {"pixel_values": xs, "labels": ys}
