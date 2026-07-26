# data/officehome.py
from __future__ import annotations

import os
import random
import tarfile
import zipfile
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image

from .base import register_data, BaseDataModule

from .vision_common import IMAGENET_MEAN, IMAGENET_STD

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _maybe_add_pin_memory_device(kwargs: dict) -> None:
    try:
        import inspect
        sig = inspect.signature(DataLoader.__init__)
        if kwargs.get("pin_memory") and torch.cuda.is_available() and "pin_memory_device" in sig.parameters:
            kwargs["pin_memory_device"] = "cuda"
    except Exception:
        pass


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def _maybe_descend_one_level(root: str) -> str:
    if not os.path.isdir(root):
        return root
    kids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if len(kids) == 1:
        inner = os.path.join(root, kids[0])
        if any(os.path.isdir(os.path.join(inner, d)) for d in os.listdir(inner)):
            return inner
    return root


def _extract_any(archive_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive_path}")


def _download_to(url: str, dest_path: str):
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _maybe_download_officehome(root: str, url: Optional[str]) -> str:
    if os.path.isdir(root) and any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root)):
        return root

    url = url or os.environ.get("OFFICEHOME_URL")
    if not url:
        return root

    print(f"[OfficeHome] No data found under {root}. Attempting download: {url}")
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, os.path.basename(url))
        _download_to(url, arc_path)
        _extract_any(arc_path, root)

    return _maybe_descend_one_level(root)


def _scan_officehome(root: str, domains: Optional[List[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return out

    root = _maybe_descend_one_level(root)
    doms = domains or [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    for dom in doms:
        ddir = os.path.join(root, dom)
        if not os.path.isdir(ddir):
            continue
        for cls in sorted(os.listdir(ddir)):
            cdir = os.path.join(ddir, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                fp = os.path.join(cdir, fn)
                if _is_image(fp):
                    out.append((fp, cls))
    return out


def _build_transforms(
    img_size: int,
    train: bool,
    *,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
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
        t.append(T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR))

    t.append(T.ToTensor())
    if train and random_erasing_p > 0.0:
        t.append(T.RandomErasing(p=float(random_erasing_p), scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"))
    t.append(T.Normalize(mean, std))
    return T.Compose(t)


def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros((y.shape[0], num_classes), device=y.device, dtype=torch.float)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh


def _rand_bbox(H: int, W: int, lam: float):
    import math, random as _r
    cut_rat = math.sqrt(1.0 - lam)
    rw = int(W * cut_rat)
    rh = int(H * cut_rat)
    cx = _r.randint(0, W - 1)
    cy = _r.randint(0, H - 1)
    x1 = max(cx - rw // 2, 0)
    y1 = max(cy - rh // 2, 0)
    x2 = min(cx + rw // 2, W)
    y2 = min(cy + rh // 2, H)
    return x1, y1, x2, y2


@dataclass
class _OHItem:
    path: str
    label: int


class _OHImageList(Dataset):
    def __init__(self, items: List[_OHItem], tfm: T.Compose):
        self.items = items
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = Image.open(it.path).convert("RGB")
        x = self.tfm(img) if self.tfm else img
        y = torch.tensor(it.label, dtype=torch.long)
        return x, y


@register_data("officehome")
class OfficeHomeDataModule(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        img_size: int = 224,
        split_ratios=(0.7, 0.15, 0.15),
        domains: Optional[List[str]] = None,
        seed: int = 0,
        num_workers: Optional[int] = None,
        drop_last: bool = False,
        mean: Optional[tuple] = None,
        std: Optional[tuple] = None,
        use_randaugment: bool = False,
        ra_n: int = 2,
        ra_m: int = 15,
        random_erasing_p: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 0.0,
        switch_prob: float = 0.5,
        download: bool = False,
        download_url: Optional[str] = None,
        pretrained: Optional[bool] = None,
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

        self.split_ratios = tuple(float(x) for x in split_ratios)
        s = sum(self.split_ratios)
        if not (abs(s - 1.0) < 1e-6):
            raise ValueError(f"split_ratios must sum to 1.0, got {self.split_ratios} (sum={s})")


        self.domains = domains
        self.seed = int(seed)

        self.num_workers = 4 if num_workers is None else int(num_workers)
        self.drop_last = bool(drop_last)

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        self.pin_memory = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)

        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together (or neither).")

        if mean is None and std is None:
            # Default to ImageNet unless user explicitly says pretrained=False
            if pretrained is False:
                mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            else:
                mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.use_randaugment = bool(use_randaugment)
        self.ra_n = int(ra_n)
        self.ra_m = int(ra_m)
        self.random_erasing_p = float(random_erasing_p)

        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.mix_prob = float(mix_prob)
        self.switch_prob = float(switch_prob)

        self.download = bool(download)
        self.download_url = download_url

        self.fast_transform = bool(fast_transform)

        self.t_train = _build_transforms(
            self.img_size,
            True,
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
            False,
            mean=self.mean,
            std=self.std,
            use_randaugment=False,
            random_erasing_p=0.0,
            fast_transform=True,
        )

        self.train_ds: Optional[_OHImageList] = None
        self.val_ds: Optional[_OHImageList] = None
        self.test_ds: Optional[_OHImageList] = None

        self.num_classes: Optional[int] = None
        self.class_names: Optional[List[str]] = None

    def on_epoch_start(self, epoch: int) -> None:
        return

    def setup(self):
        root = self.data_root
        pairs = _scan_officehome(root, self.domains)

        if not pairs and self.download:
            root = _maybe_download_officehome(root, self.download_url)
            pairs = _scan_officehome(root, self.domains)

        if not pairs:
            raise FileNotFoundError(
                f"[OfficeHome] No images found under: {self.data_root}. "
                f"If you want auto-download, pass download=True and provide download_url or set OFFICEHOME_URL."
            )

        classes = sorted({cls for _, cls in pairs})
        cls2id = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)
        self.class_names = classes

        items: List[_OHItem] = [_OHItem(p, cls2id[c]) for p, c in pairs]

        rng = random.Random(self.seed)
        items.sort(key=lambda it: it.path)
        rng.shuffle(items)

        n = len(items)
        n_tr = max(1, min(n - 2, int(self.split_ratios[0] * n)))
        n_v = max(1, min(n - n_tr - 1, int(self.split_ratios[1] * n)))

        train_items = items[:n_tr]
        val_items = items[n_tr:n_tr + n_v]
        test_items = items[n_tr + n_v:]

        self.train_ds = _OHImageList(train_items, self.t_train)
        self.val_ds = _OHImageList(val_items, self.t_eval)
        self.test_ds = _OHImageList(test_items, self.t_eval)

    def _collate_train(self, batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0).contiguous()
        ys = torch.tensor(ys, dtype=torch.long)

        if self.mix_prob <= 0.0 or (self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0):
            return {"pixel_values": xs, "labels": ys}
        if random.random() >= self.mix_prob:
            return {"pixel_values": xs, "labels": ys}

        B, C, H, W = xs.shape
        perm = torch.randperm(B)
        xs2 = xs[perm]
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

    def _loader(self, ds: Dataset, train: bool = False) -> DataLoader:
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

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self) -> DataLoader:   return self._loader(self.val_ds, False)
    def test_dataloader(self) -> DataLoader:  return self._loader(self.test_ds, False)
