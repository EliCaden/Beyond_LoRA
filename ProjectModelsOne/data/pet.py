# data/pet.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T

from .base import BaseDataModule, register_data
from .vtab_helpers import _to_rgb, _collate


def _maybe_add_pin_memory_device(kwargs: dict) -> None:
    try:
        import inspect
        sig = inspect.signature(DataLoader.__init__)
        if kwargs.get("pin_memory") and torch.cuda.is_available() and "pin_memory_device" in sig.parameters:
            kwargs["pin_memory_device"] = "cuda"
    except Exception:
        pass


@register_data("oxford_iiit_pet")
class OxfordPetDataModule(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        img_size: int = 224,
        batch_size: int = 128,
        val_split: float = 0.10,
        seed: int = 0,
        num_workers: Optional[int] = None,
        drop_last: bool = False,
        download: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        # loader knobs
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
    ):
        super().__init__(data_root=data_root, batch_size=batch_size, img_size=img_size, seed=seed)

        self.data_root = data_root
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.val_split = float(val_split)
        self.seed = int(seed)

        self.num_workers = 4 if num_workers is None else int(num_workers)
        self.drop_last = bool(drop_last)
        self.download = bool(download)

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        self.pin_memory = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)

        self.num_classes = 37

        self.mean = tuple(mean) if mean is not None else (0.5, 0.5, 0.5)
        self.std = tuple(std) if std is not None else (0.5, 0.5, 0.5)

        self.t_train = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            _to_rgb,
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        self.t_eval = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            _to_rgb,
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def on_epoch_start(self, epoch: int) -> None:
        return

    def setup(self):
        # Two views so val uses eval transforms
        full_train = OxfordIIITPet(self.data_root, split="trainval", download=self.download, transform=self.t_train)
        full_eval = OxfordIIITPet(self.data_root, split="trainval", download=self.download, transform=self.t_eval)
        test = OxfordIIITPet(self.data_root, split="test", download=self.download, transform=self.t_eval)

        N = len(full_train)
        val_count = max(1, int(self.val_split * N))
        train_count = N - val_count
        g = torch.Generator().manual_seed(self.seed)

        tr_idx_subset, va_idx_subset = random_split(range(N), [train_count, val_count], generator=g)
        tr_idx = list(tr_idx_subset)
        va_idx = list(va_idx_subset)

        self.train_ds = Subset(full_train, tr_idx)
        self.val_ds = Subset(full_eval, va_idx)
        self.test_ds = test

    def _loader(self, ds, train: bool = False) -> DataLoader:
        pf = self.prefetch_factor if self.num_workers > 0 else None
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=bool(train),
            num_workers=self.num_workers,
            drop_last=(self.drop_last if train else False),
            pin_memory=self.pin_memory,
            collate_fn=_collate,
            persistent_workers=(self.persistent_workers if self.num_workers > 0 else False),
        )
        if pf is not None:
            kwargs["prefetch_factor"] = pf
        _maybe_add_pin_memory_device(kwargs)
        return DataLoader(ds, **kwargs)

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds, False)
    def test_dataloader(self):  return self._loader(self.test_ds, False)
