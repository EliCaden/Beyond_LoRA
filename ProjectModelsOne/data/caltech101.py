# data/caltech101.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import Caltech101
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


@register_data("caltech101")
class Caltech101DataModule(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        img_size: int = 224,
        batch_size: int = 128,
        val_split: float = 0.10,
        test_split: float = 0.10,
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
        super().__init__(data_root=data_root, img_size=img_size, batch_size=batch_size, seed=seed, val_split=val_split, test_split=test_split)

        self.data_root = data_root
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.seed = int(seed)

        self.num_workers = 4 if num_workers is None else int(num_workers)
        self.drop_last = bool(drop_last)
        self.download = bool(download)

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        self.pin_memory = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)

        self.num_classes = 101

        self.mean = tuple(mean) if mean is not None else (0.5, 0.5, 0.5)
        self.std = tuple(std) if std is not None else (0.5, 0.5, 0.5)

        base = [
            T.Resize((self.img_size, self.img_size)),
            _to_rgb,
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ]
        self.t_train = T.Compose(base)
        self.t_eval = T.Compose(base)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def on_epoch_start(self, epoch: int) -> None:
        return

    def setup(self):
        base_train = Caltech101(self.data_root, download=self.download, transform=self.t_train)
        base_eval = Caltech101(self.data_root, download=self.download, transform=self.t_eval)

        N = len(base_train)
        test_count = max(1, min(N - 2, int(self.test_split * N)))
        trainval_count = N - test_count

        g = torch.Generator().manual_seed(self.seed)
        trainval_split, test_split = random_split(range(N), [trainval_count, test_count], generator=g)

        val_count = max(1, min(trainval_count - 1, int(self.val_split * trainval_count)))
        train_count = trainval_count - val_count

        tv_idx = list(trainval_split)
        train_idx_split, val_idx_split = random_split(tv_idx, [train_count, val_count], generator=g)

        train_idx = list(train_idx_split)
        val_idx = list(val_idx_split)
        test_idx = list(test_split)

        self.train_ds = Subset(base_train, train_idx)
        self.val_ds = Subset(base_eval, val_idx)
        self.test_ds = Subset(base_eval, test_idx)

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
