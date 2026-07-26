# data/imagenet1k.py
from __future__ import annotations

import os
import math
import random
from typing import Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
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


def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros((y.shape[0], num_classes), device=y.device, dtype=torch.float)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh


def _rand_bbox(H: int, W: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = math.sqrt(1.0 - lam)
    rw = int(W * cut_rat)
    rh = int(H * cut_rat)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = max(cx - rw // 2, 0)
    y1 = max(cy - rh // 2, 0)
    x2 = min(cx + rw // 2, W)
    y2 = min(cy + rh // 2, H)
    return x1, y1, x2, y2


def _build_transforms(
    img_size: int,
    train: bool,
    *,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    use_randaugment: bool = False,
    ra_n: int = 2,
    ra_m: int = 15,
    random_erasing_p: float = 0.0,
    fast_transform: bool = True,
) -> T.Compose:
    t = []
    if train:
        if fast_transform:
            t.append(T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR))
            t.append(T.RandomHorizontalFlip(p=0.5))
            if use_randaugment:
                t.append(T.RandAugment(num_ops=int(ra_n), magnitude=int(ra_m)))
        else:
            t.append(
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.08, 1.0),
                    ratio=(0.67, 1.5),
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
            t.append(T.RandomHorizontalFlip(p=0.5))
            t.append(T.ColorJitter(0.4, 0.4, 0.4, 0.4))
            if use_randaugment:
                t.append(T.RandAugment(num_ops=int(ra_n), magnitude=int(ra_m)))
    else:
        if fast_transform:
            t.append(T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR))
        else:
            t.append(T.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC))
            t.append(T.CenterCrop(img_size))

    t.append(T.ToTensor())
    if train and random_erasing_p > 0.0:
        t.append(
            T.RandomErasing(
                p=float(random_erasing_p),
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    t.append(T.Normalize(mean, std))
    return T.Compose(t)


@register_data("imagenet1k")
class ImageNet1KDataModule(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        img_size: int = 224,
        seed: int = 0,
        num_workers: Optional[int] = None,
        drop_last: bool = False,
        val_split: float = 0.05,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        train_dir_name: str = "train",
        val_dir_name: str = "val",
        test_dir_name: Optional[str] = None,
        download: bool = False,
        use_randaugment: bool = False,
        ra_n: int = 2,
        ra_m: int = 15,
        random_erasing_p: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 0.0,
        switch_prob: float = 0.5,
        # speed knobs
        fast_transform: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
        **_: Any,
    ):
        super().__init__(data_root=data_root, batch_size=batch_size, img_size=img_size, seed=seed)

        self.data_root = data_root
        self.batch_size = int(batch_size)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.val_split = float(val_split)

        self.num_workers = 4 if num_workers is None else int(num_workers)
        self.drop_last = bool(drop_last)

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        self.pin_memory = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)

        if download:
            print("[WARN] ImageNet1KDataModule: 'download' ignored; ImageNet must be on disk.")

        self.mean = tuple(mean) if mean is not None else (0.485, 0.456, 0.406)
        self.std = tuple(std) if std is not None else (0.229, 0.224, 0.225)

        self.train_dir_name = train_dir_name
        self.val_dir_name = val_dir_name
        self.test_dir_name = test_dir_name

        self.use_randaugment = bool(use_randaugment)
        self.ra_n = int(ra_n)
        self.ra_m = int(ra_m)
        self.random_erasing_p = float(random_erasing_p)

        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.mix_prob = float(mix_prob)
        self.switch_prob = float(switch_prob)

        self.fast_transform = bool(fast_transform)

        self.t_train = _build_transforms(
            self.img_size,
            train=True,
            mean=self.mean,
            std=self.std,
            use_randaugment=self.use_randaugment,
            ra_n=self.ra_n,
            ra_m=self.ra_m,
            random_erasing_p=self.random_erasing_p,
            fast_transform=self.fast_transform,
        )
        self.t_eval = _build_transforms(
            self.img_size,
            train=False,
            mean=self.mean,
            std=self.std,
            use_randaugment=False,
            random_erasing_p=0.0,
            fast_transform=True,
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.num_classes: Optional[int] = None
        self.class_names = None

    def on_epoch_start(self, epoch: int) -> None:
        return

    def setup(self):
        train_dir = os.path.join(self.data_root, self.train_dir_name)
        val_dir = os.path.join(self.data_root, self.val_dir_name)
        test_dir = os.path.join(self.data_root, self.test_dir_name) if self.test_dir_name else None

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")

        if os.path.isdir(val_dir):
            train_base = ImageFolder(root=train_dir, transform=self.t_train)
            val_base = ImageFolder(root=val_dir, transform=self.t_eval)
            if test_dir and os.path.isdir(test_dir):
                test_base = ImageFolder(root=test_dir, transform=self.t_eval)
            else:
                test_base = val_base
            self.train_ds = train_base
            self.val_ds = val_base
            self.test_ds = test_base
        else:
            full_train_train = ImageFolder(root=train_dir, transform=self.t_train)
            full_train_eval = ImageFolder(root=train_dir, transform=self.t_eval)

            n_val = max(1, int(self.val_split * len(full_train_train)))
            n_tr = len(full_train_train) - n_val
            g = torch.Generator().manual_seed(self.seed)

            tr_idx_subset, va_idx_subset = random_split(range(len(full_train_train)), [n_tr, n_val], generator=g)
            tr_idx = list(tr_idx_subset)
            va_idx = list(va_idx_subset)

            self.train_ds = Subset(full_train_train, tr_idx)
            self.val_ds = Subset(full_train_eval, va_idx)
            self.test_ds = self.val_ds

        base = self.train_ds.dataset if isinstance(self.train_ds, Subset) else self.train_ds
        if hasattr(base, "classes"):
            self.num_classes = len(base.classes)
            self.class_names = list(base.classes)

    def _collate_train(self, batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0).contiguous()
        ys = torch.tensor(ys, dtype=torch.long)

        if (
            self.mix_prob <= 0.0
            or (self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0)
            or random.random() >= self.mix_prob
        ):
            return {"pixel_values": xs, "labels": ys}

        B, C, H, W = xs.shape
        perm = torch.randperm(B)
        xs2 = xs[perm]
        if self.num_classes is None:
            raise RuntimeError("num_classes not set (setup() must run).")

        y1 = _one_hot(ys, self.num_classes)
        y2 = _one_hot(ys[perm], self.num_classes)

        can_cutmix = self.cutmix_alpha > 0.0
        can_mixup = self.mixup_alpha > 0.0
        use_cutmix = ((can_cutmix and can_mixup and (random.random() < self.switch_prob)) or (can_cutmix and not can_mixup))

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            x1b, y1b, x2b, y2b = _rand_bbox(H, W, lam)
            xs[:, :, y1b:y2b, x1b:x2b] = xs2[:, :, y1b:y2b, x1b:x2b]
            lam = 1.0 - ((x2b - x1b) * (y2b - y1b) / float(H * W))
            y = lam * y1 + (1.0 - lam) * y2
        else:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            xs = lam * xs + (1.0 - lam) * xs2
            y = lam * y1 + (1.0 - lam) * y2

        return {"pixel_values": xs, "labels": y}

    def _collate_eval(self, batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0).contiguous()
        ys = torch.tensor(ys, dtype=torch.long)
        return {"pixel_values": xs, "labels": ys}

    def _loader(self, ds, train: bool = False) -> DataLoader:
        pf = self.prefetch_factor if self.num_workers > 0 else None
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=bool(train),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(self.drop_last if train else False),
            collate_fn=(self._collate_train if train else self._collate_eval),
            persistent_workers=(self.persistent_workers if self.num_workers > 0 else False),
        )
        if pf is not None:
            kwargs["prefetch_factor"] = pf
        _maybe_add_pin_memory_device(kwargs)
        return DataLoader(ds, **kwargs)

    def train_dataloader(self): return self._loader(self.train_ds, train=True)
    def val_dataloader(self):   return self._loader(self.val_ds, train=False)
    def test_dataloader(self):  return self._loader(self.test_ds, train=False)
