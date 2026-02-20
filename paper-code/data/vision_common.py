# data/vision_common.py
from __future__ import annotations

from typing import Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_train_tfms(size: int, *, fast_transform: bool = True) -> T.Compose:
    if fast_transform:
        return T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(size + 32),
        T.RandomResizedCrop(size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_tfms(size: int, *, fast_transform: bool = True) -> T.Compose:
    if fast_transform:
        return T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(size + 32),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageFolderWithPaths(Dataset):
    """
    Generic folder dataset (class subfolders). If 'domains' are used, the structure is:
      root/domain/class/*.jpg
    Else: root/class/*.jpg
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": label}
