# models/convit_classifier.py
import torch
import torch.nn as nn
from types import SimpleNamespace

try:
    import timm
except ImportError as e:
    raise RuntimeError("timm is required for ConViT backbones. pip install timm") from e


_VIANT_MAP = {
    "tiny":  "convit_tiny",
    "small": "convit_small",
    "base":  "convit_base",
    # allow passing a full timm name via --vit-variant too (falls back to key)
}

class ConViTImageModel(nn.Module):
    def __init__(self, variant: str = "tiny", num_classes: int = 10, pretrained: bool = True):
        super().__init__()  # <<< IMPORTANT: call before assigning any submodule
        model_name = _VIANT_MAP.get(variant, variant)  # 'tiny' -> 'convit_tiny', else pass-through
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, pixel_values=None, **kwargs):
        # Trainer builds inputs from the batch with key 'pixel_values'
        x = pixel_values
        if x is None:
            # be tolerant to other key names
            x = kwargs.get("images") or kwargs.get("x")
        logits = self.net(x)  # timm classifiers return raw logits
        return SimpleNamespace(logits=logits)
