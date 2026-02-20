# tasks/vision_image.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import inspect

import torch
import torch.nn.functional as F

_FAST_EVAL_METRICS = os.getenv("FAST_EVAL_METRICS", "0").lower() in {"1", "true", "yes", "y"}
_HAS_CE_LABEL_SMOOTHING = "label_smoothing" in inspect.signature(F.cross_entropy).parameters


def _as_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _normalize_hard_labels(
    labels,
    *,
    device: torch.device,
    ignore_index: int,
) -> torch.Tensor:
    """
    Normalize hard labels to (B,) long on device.
    Accepts (B,), (B,1).
    """
    y = _as_tensor(labels)
    if y.device != device:
        y = y.to(device=device, non_blocking=True)
    if y.dtype != torch.long:
        y = y.to(dtype=torch.long)

    if y.ndim == 0:
        y = y.view(1)
    elif y.ndim == 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    elif y.ndim != 1:
        raise ValueError(f"Expected labels shape (B,) or (B,1); got {tuple(y.shape)}")

    # Optional: treat negative as ignore_index without sync (pure GPU op).
    if ignore_index is not None and ignore_index < 0:
        y = torch.where(y < 0, torch.full_like(y, ignore_index), y)

    return y


def _valid_mask(y: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if ignore_index is None:
        return torch.ones_like(y, dtype=torch.bool)
    return y != int(ignore_index)


class _AsyncVecAdder:
    """
    Accumulate small vectors on CPU, fed from GPU via async D2H copies + CUDA events.
    """
    _MAX_PENDING = 256

    def __init__(self, dim: int, dtype: torch.dtype = torch.float64, acc_dtype: torch.dtype = torch.float64):
        self.dim = int(dim)
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self._acc = torch.zeros(self.dim, dtype=self.acc_dtype)  # CPU accumulator

        self._pending: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        self._pool: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []

    def reset(self):
        self._acc.zero_()
        self._pending.clear()

    def _alloc_host_event(self) -> Tuple[torch.Tensor, "torch.cuda.Event"]:
        if self._pool:
            return self._pool.pop()
        host = torch.empty((self.dim,), device="cpu", dtype=self.dtype, pin_memory=True)
        ev = torch.cuda.Event(enable_timing=False)
        return host, ev

    def _recycle(self, host: torch.Tensor, ev: "torch.cuda.Event") -> None:
        self._pool.append((host, ev))

    @torch.no_grad()
    def update(self, v: torch.Tensor):
        v = v.detach()
        if v.is_cuda and torch.cuda.is_available():
            vv = v.to(dtype=self.dtype)
            host, ev = self._alloc_host_event()
            host.copy_(vv, non_blocking=True)
            ev.record(torch.cuda.current_stream())
            self._pending.append((host, ev))
            if len(self._pending) > self._MAX_PENDING:
                self._drain(sync=False)
        else:
            self._acc.add_(v.to(dtype=self.acc_dtype).cpu())

    def _drain(self, sync: bool):
        if not self._pending:
            return

        if sync:
            try:
                _, last_ev = self._pending[-1]
                last_ev.synchronize()
                all_ready = True
            except Exception:
                all_ready = False

            if not all_ready:
                for _, ev in self._pending:
                    try:
                        ev.synchronize()
                    except Exception:
                        pass

            for host, ev in self._pending:
                self._acc.add_(host.to(dtype=self.acc_dtype))
                self._recycle(host, ev)
            self._pending.clear()
            return

        keep: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        for host, ev in self._pending:
            try:
                if ev.query():
                    self._acc.add_(host.to(dtype=self.acc_dtype))
                    self._recycle(host, ev)
                else:
                    keep.append((host, ev))
            except Exception:
                keep.append((host, ev))
        self._pending = keep

    def sync(self):
        self._drain(sync=True)

    def snapshot(self) -> torch.Tensor:
        self._drain(sync=(not _FAST_EVAL_METRICS))
        return self._acc.clone()


class _AccMeter:
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)
        self._v = _AsyncVecAdder(dim=2, dtype=torch.float64, acc_dtype=torch.float64)  # [correct, total]

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels_hard: torch.Tensor):
        dev = logits.device
        preds = logits.argmax(dim=-1)
        preds = preds.view(-1)

        y = _normalize_hard_labels(labels_hard, device=dev, ignore_index=self.ignore_index).view(-1)
        if preds.numel() != y.numel():
            raise ValueError(f"preds/labels size mismatch: {preds.numel()} vs {y.numel()}")

        valid = _valid_mask(y, self.ignore_index)
        correct = ((preds == y) & valid).sum(dtype=torch.float64)
        total = valid.sum(dtype=torch.float64)

        vec = torch.empty((2,), device=dev, dtype=torch.float64)
        vec[0] = correct
        vec[1] = total
        self._v.update(vec)

    def sync(self):
        self._v.sync()

    def compute(self) -> float:
        c, t = self._v.snapshot().tolist()
        return 0.0 if t <= 0 else float(c / t)


class _MeanLossMeter:
    def __init__(self):
        self._v = _AsyncVecAdder(dim=2, dtype=torch.float64, acc_dtype=torch.float64)  # [sum_loss, count]

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, loss_mean: torch.Tensor, count: Union[int, torch.Tensor]):
        dev = loss_mean.device
        if torch.is_tensor(count):
            c = count.detach().to(device=dev, dtype=torch.float64)
        else:
            c = torch.full((), float(int(count)), device=dev, dtype=torch.float64)

        s = loss_mean.detach().to(dtype=torch.float64) * c

        vec = torch.empty((2,), device=dev, dtype=torch.float64)
        vec[0] = s
        vec[1] = c
        self._v.update(vec)

    def sync(self):
        self._v.sync()

    def compute(self) -> float:
        s, c = self._v.snapshot().tolist()
        return 0.0 if c <= 0 else float(s / c)


class _MCCMeter:
    """
    Multi-class MCC with CPU confusion matrix, updated via async D2H copies.
    """
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.k = int(num_classes)
        self.ignore_index = int(ignore_index)

        self.cm_cpu = torch.zeros(self.k, self.k, dtype=torch.int64)  # CPU
        self._pending: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        self._pool: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []

    def reset(self):
        self.cm_cpu.zero_()
        self._pending.clear()

    def _alloc_host_event(self) -> Tuple[torch.Tensor, "torch.cuda.Event"]:
        if self._pool:
            return self._pool.pop()
        host = torch.empty((self.k, self.k), device="cpu", dtype=torch.int64, pin_memory=True)
        ev = torch.cuda.Event(enable_timing=False)
        return host, ev

    def _recycle(self, host: torch.Tensor, ev: "torch.cuda.Event") -> None:
        self._pool.append((host, ev))

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels_hard: torch.Tensor):
        dev = logits.device
        preds = logits.argmax(dim=-1).view(-1)
        y = _normalize_hard_labels(labels_hard, device=dev, ignore_index=self.ignore_index).view(-1)

        if preds.numel() != y.numel():
            raise ValueError(f"preds/labels size mismatch: {preds.numel()} vs {y.numel()}")

        valid = _valid_mask(y, self.ignore_index)
        if valid.dtype != torch.bool:
            valid = valid.to(dtype=torch.bool)

        # Filter to valid labels only (pure GPU ops; no sync).
        yv = y[valid]
        pv = preds[valid]

        k = self.k
        if yv.numel() == 0:
            return

        idx = (yv * k + pv).to(dtype=torch.long)
        counts = torch.bincount(idx, minlength=k * k).to(dtype=torch.int64).reshape(k, k)

        if counts.is_cuda and torch.cuda.is_available():
            host, ev = self._alloc_host_event()
            host.copy_(counts, non_blocking=True)
            ev.record(torch.cuda.current_stream())
            self._pending.append((host, ev))
        else:
            self.cm_cpu.add_(counts.cpu())

    def _drain(self, sync: bool):
        if not self._pending:
            return

        if sync:
            try:
                _, last_ev = self._pending[-1]
                last_ev.synchronize()
                all_ready = True
            except Exception:
                all_ready = False

            if not all_ready:
                for _, ev in self._pending:
                    try:
                        ev.synchronize()
                    except Exception:
                        pass

            for host, ev in self._pending:
                self.cm_cpu.add_(host)
                self._recycle(host, ev)
            self._pending.clear()
            return

        keep: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        for host, ev in self._pending:
            try:
                if ev.query():
                    self.cm_cpu.add_(host)
                    self._recycle(host, ev)
                else:
                    keep.append((host, ev))
            except Exception:
                keep.append((host, ev))
        self._pending = keep

    def sync(self):
        self._drain(sync=True)

    def compute(self) -> float:
        self._drain(sync=(not _FAST_EVAL_METRICS))

        cm = self.cm_cpu.to(dtype=torch.float64)
        s = cm.sum()
        if float(s.item()) == 0.0:
            return 0.0
        t = cm.sum(dim=1)
        p = cm.sum(dim=0)
        c = torch.diag(cm).sum()
        num = (c * s) - (t * p).sum()
        den = torch.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
        den_v = float(den.item())
        if den_v <= 0.0:
            return 0.0
        return float((num / den).item())


class VisionClassificationTask:
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        primary_metric: str = "acc",
        maximize: bool | None = None,
        track_mcc: bool = False,
        ignore_index: int = -100,
    ):
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)

        self._acc = _AccMeter(ignore_index=self.ignore_index)
        self._loss_meter = _MeanLossMeter()
        self._track_mcc = bool(track_mcc)
        self._mcc = _MCCMeter(self.num_classes, ignore_index=self.ignore_index) if self._track_mcc else None

        self._primary = primary_metric.lower().strip()
        self._maximize = (
            bool(maximize)
            if maximize is not None
            else (self._primary not in {"loss", "val_loss", "ce", "cross_entropy"})
        )

        self.mode = "classification"
        self._cached_eval_loss: Optional[torch.Tensor] = None
        self._cached_eval_count: Optional[torch.Tensor] = None

    def default_metrics_and_primary(self):
        return self._primary, self._maximize

    def reset_metrics(self):
        self._acc.reset()
        self._loss_meter.reset()
        if self._track_mcc and self._mcc is not None:
            self._mcc.reset()
        self._cached_eval_loss = None
        self._cached_eval_count = None

    def sync_metrics(self):
        self._acc.sync()
        self._loss_meter.sync()
        if self._track_mcc and self._mcc is not None:
            self._mcc.sync()

    def metrics(self) -> Dict[str, Any]:
        d = {"acc": self._acc, "loss": self._loss_meter}
        if self._track_mcc and self._mcc is not None:
            d["mcc"] = self._mcc
        return d

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor, smoothing: float) -> torch.Tensor:
        # Soft labels: expect (B, C) float probabilities
        if labels.ndim == 2 and torch.is_floating_point(labels):
            if labels.shape != logits.shape:
                raise ValueError(f"Soft labels shape must match logits. labels={tuple(labels.shape)} logits={tuple(logits.shape)}")
            log_probs = F.log_softmax(logits, dim=-1)
            return -(labels * log_probs).sum(dim=-1).mean()

        # Hard labels
        y = _normalize_hard_labels(labels, device=logits.device, ignore_index=self.ignore_index)

        if smoothing > 0.0 and _HAS_CE_LABEL_SMOOTHING:
            return F.cross_entropy(
                logits,
                y,
                ignore_index=self.ignore_index,
                label_smoothing=float(smoothing),
            )

        if smoothing > 0.0:
            s = float(smoothing)
            log_probs = F.log_softmax(logits, dim=-1)

            # nll handles ignore_index internally.
            nll = F.nll_loss(log_probs, y, reduction="mean", ignore_index=self.ignore_index)

            # Smooth term should also exclude ignored labels (no sync).
            smooth_per = (-log_probs.mean(dim=-1))  # (B,)
            m = _valid_mask(y, self.ignore_index).to(dtype=log_probs.dtype)
            denom = m.sum().clamp(min=1.0)
            smooth = (smooth_per * m).sum() / denom

            return (1.0 - s) * nll + s * smooth

        return F.cross_entropy(logits, y, ignore_index=self.ignore_index)

    def loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch["labels"]
        labels = _as_tensor(labels)
        if labels.device != logits.device:
            labels = labels.to(device=logits.device, non_blocking=True)
        return self._ce(logits, labels, smoothing=self.label_smoothing)

    def loss_eval(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch["labels"]
        labels = _as_tensor(labels)
        if labels.device != logits.device:
            labels = labels.to(device=logits.device, non_blocking=True)

        loss = self._ce(logits, labels, smoothing=0.0)
        self._cached_eval_loss = loss.detach()

        # Count examples contributing to loss (no sync).
        if labels.ndim == 2 and torch.is_floating_point(labels):
            count = torch.full((), float(labels.shape[0]), device=logits.device, dtype=torch.float64)
        else:
            y = _normalize_hard_labels(labels, device=logits.device, ignore_index=self.ignore_index)
            count = _valid_mask(y, self.ignore_index).sum(dtype=torch.float64)

        self._cached_eval_count = count.detach()
        return loss

    @torch.no_grad()
    def update_metrics(self, outputs: Any, batch: Dict[str, Any]):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = _as_tensor(batch["labels"])
        if labels.device != logits.device:
            labels = labels.to(device=logits.device, non_blocking=True)

        # Hard labels for acc/mcc
        if torch.is_floating_point(labels):
            labels_hard = labels.argmax(dim=-1)
            count = torch.full((), float(labels.shape[0]), device=logits.device, dtype=torch.float64)
        else:
            labels_hard = _normalize_hard_labels(labels, device=logits.device, ignore_index=self.ignore_index)
            count = _valid_mask(labels_hard, self.ignore_index).sum(dtype=torch.float64)

        self._acc.update(logits, labels_hard)
        if self._track_mcc and self._mcc is not None:
            self._mcc.update(logits, labels_hard)

        loss = self._ce(logits, labels, smoothing=0.0)
        self._loss_meter.update(loss, count)

    @torch.no_grad()
    def update_metrics_eval(self, outputs: Any, batch: Dict[str, Any]):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = _as_tensor(batch["labels"])
        if labels.device != logits.device:
            labels = labels.to(device=logits.device, non_blocking=True)

        if torch.is_floating_point(labels):
            labels_hard = labels.argmax(dim=-1)
            count = torch.full((), float(labels.shape[0]), device=logits.device, dtype=torch.float64)
        else:
            labels_hard = _normalize_hard_labels(labels, device=logits.device, ignore_index=self.ignore_index)
            count = _valid_mask(labels_hard, self.ignore_index).sum(dtype=torch.float64)

        self._acc.update(logits, labels_hard)
        if self._track_mcc and self._mcc is not None:
            self._mcc.update(logits, labels_hard)

        if self._cached_eval_loss is not None and self._cached_eval_count is not None:
            loss = self._cached_eval_loss
            count = self._cached_eval_count
            self._cached_eval_loss = None
            self._cached_eval_count = None
        else:
            loss = self._ce(logits, labels, smoothing=0.0)

        self._loss_meter.update(loss, count)