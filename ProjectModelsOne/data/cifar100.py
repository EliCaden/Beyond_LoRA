# data/cifar100.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .base import register_data, BaseDataModule


def _maybe_add_pin_memory_device(kwargs: dict) -> None:
    try:
        import inspect
        sig = inspect.signature(DataLoader.__init__)
        if kwargs.get("pin_memory") and torch.cuda.is_available() and "pin_memory_device" in sig.parameters:
            kwargs["pin_memory_device"] = "cuda"
    except Exception:
        pass


def _collate(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0).contiguous()
    ys = torch.tensor(ys, dtype=torch.long)
    return {"pixel_values": xs, "labels": ys}


def _build_transforms(
    img_size: int,
    train: bool,
    *,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    fast_transform: bool = True,
) -> T.Compose:
    interp = InterpolationMode.BILINEAR if fast_transform else InterpolationMode.BICUBIC
    t = [T.Resize((img_size, img_size), interpolation=interp)]
    if train:
        t.append(T.RandomHorizontalFlip(p=0.5))
    t.extend([T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose(t)


@register_data("cifar100")
class CIFAR100DataModule(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        img_size: int = 224,
        val_split: float = 0.05,
        seed: int = 0,
        num_workers: Optional[int] = None,
        drop_last: bool = False,
        download: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        # speed knobs
        fast_transform: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
    ):
        super().__init__(data_root=data_root, batch_size=batch_size, img_size=img_size, seed=seed)

        self.data_root = data_root
        self.batch_size = int(batch_size)
        self.img_size = int(img_size)
        self.val_split = float(val_split)
        self.seed = int(seed)

        self.num_workers = 4 if num_workers is None else int(num_workers)
        self.drop_last = bool(drop_last)
        self.download = bool(download)

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        self.pin_memory = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)

        self.fast_transform = bool(fast_transform)

        self.num_classes = 100
        self.class_names = [str(i) for i in range(self.num_classes)]

        self.mean = tuple(mean) if mean is not None else (0.5, 0.5, 0.5)
        self.std = tuple(std) if std is not None else (0.5, 0.5, 0.5)

        self.t_train = _build_transforms(self.img_size, True, mean=self.mean, std=self.std, fast_transform=self.fast_transform)
        self.t_eval = _build_transforms(self.img_size, False, mean=self.mean, std=self.std, fast_transform=True)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def on_epoch_start(self, epoch: int) -> None:
        return

    def setup(self):
        tr_train = CIFAR100(root=self.data_root, train=True, download=self.download, transform=self.t_train)
        tr_eval = CIFAR100(root=self.data_root, train=True, download=self.download, transform=self.t_eval)
        te = CIFAR100(root=self.data_root, train=False, download=self.download, transform=self.t_eval)

        N = len(tr_train)
        n_val = max(1, int(self.val_split * N))
        n_tr = N - n_val
        g = torch.Generator().manual_seed(self.seed)

        tr_idx_subset, va_idx_subset = random_split(range(N), [n_tr, n_val], generator=g)
        tr_idx = list(tr_idx_subset)
        va_idx = list(va_idx_subset)

        self.train_ds = Subset(tr_train, tr_idx)
        self.val_ds = Subset(tr_eval, va_idx)
        self.test_ds = te

    def _loader(self, ds, train: bool = False) -> DataLoader:
        pf = self.prefetch_factor if self.num_workers > 0 else None
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=bool(train),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(self.drop_last if train else False),
            collate_fn=_collate,
            persistent_workers=(self.persistent_workers if self.num_workers > 0 else False),
        )
        if pf is not None:
            kwargs["prefetch_factor"] = pf
        _maybe_add_pin_memory_device(kwargs)
        return DataLoader(ds, **kwargs)

    def train_dataloader(self): return self._loader(self.train_ds, train=True)
    def val_dataloader(self):   return self._loader(self.val_ds, train=False)
    def test_dataloader(self):  return self._loader(self.test_ds, train=False)
