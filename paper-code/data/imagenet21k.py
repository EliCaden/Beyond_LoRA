# data/imagenet21k.py
from __future__ import annotations

from .base import register_data
from .imagenet1k import ImageNet1KDataModule


@register_data("imagenet21k")
class ImageNet21KDataModule(ImageNet1KDataModule):
    """
    ImageNet-21k DataModule.

    This is a thin alias around ImageNet1KDataModule so you can do:
        dm = get_data("imagenet21k", data_root="...", ...)

    It assumes the same ImageFolder-style layout:
        data_root/
          train/
            class1/
            ...
          [optional] val/
          [optional] test/
    """
    # Inherit everything (init, transforms, setup, loaders) from ImageNet1KDataModule.
    pass
