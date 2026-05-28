# tasks/glue_text.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import os

import torch
import torch.nn.functional as F

_FAST_EVAL_METRICS = os.getenv("FAST_EVAL_METRICS", "0").lower() in {"1", "true", "yes", "y"}


class _Meter:
    def reset(self): ...
    def update(self, *args, **kwargs): ...
    def compute(self): return None
    def sync(self): ...


class _AsyncVecAdder:
    """
    Accumulate a small vector on CPU, fed from GPU via async D2H copies.
    """
    _MAX_PENDING = 256

    def __init__(self, dim: int, dtype: torch.dtype = torch.float64):
        self.dim = int(dim)
        self.dtype = dtype
        self._acc = torch.zeros(self.dim, dtype=self.dtype)  # CPU
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
            self._acc.add_(v.to(dtype=self.dtype).cpu())

    def _drain(self, sync: bool):
        if not self._pending:
            return

        if sync:
            try:
                _, last_ev = self._pending[-1]
                last_ev.synchronize()
            except Exception:
                for _, ev in self._pending:
                    try:
                        ev.synchronize()
                    except Exception:
                        pass

            for host, ev in self._pending:
                self._acc.add_(host)
                self._recycle(host, ev)
            self._pending.clear()
            return

        keep: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        for host, ev in self._pending:
            try:
                if ev.query():
                    self._acc.add_(host)
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


def _as_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _normalize_class_labels(
    labels,
    *,
    device: torch.device,
    ignore_index: int,
) -> torch.Tensor:
    """
    Normalize hard classification labels to shape (B,) on `device`, dtype long.

    Accepts labels shaped:
      - (B,)
      - (B, 1)
    Any other shape is likely a caller/collator bug and we raise early (correctness).
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

    # If someone uses -1 (or any negative) for "no label", map it to ignore_index
    # without synchronizing (pure GPU op).
    if ignore_index is not None and ignore_index < 0:
        y = torch.where(y < 0, torch.full_like(y, ignore_index), y)

    return y


def _valid_mask(y: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if ignore_index is None:
        return torch.ones_like(y, dtype=torch.bool)
    return y != int(ignore_index)


class Accuracy(_Meter):
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)
        self._v = _AsyncVecAdder(dim=2, dtype=torch.float64)  # [correct, total]

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, logits, labels):
        dev = logits.device
        preds = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0).to(torch.long)

        y = _normalize_class_labels(labels, device=dev, ignore_index=self.ignore_index)
        preds = preds.view(-1)
        y = y.view(-1)

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

    def compute(self):
        c, t = self._v.snapshot().tolist()
        return 0.0 if t <= 0 else float(c / t)


class MCC(_Meter):
    # binary MCC for CoLA: accum [tp, tn, fp, fn]
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)
        self._v = _AsyncVecAdder(dim=4, dtype=torch.float64)

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, logits, labels):
        dev = logits.device
        preds = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0).to(torch.long)

        y = _normalize_class_labels(labels, device=dev, ignore_index=self.ignore_index)
        preds = preds.view(-1)
        y = y.view(-1)

        if preds.numel() != y.numel():
            raise ValueError(f"preds/labels size mismatch: {preds.numel()} vs {y.numel()}")

        valid = _valid_mask(y, self.ignore_index)

        tp = ((preds == 1) & (y == 1) & valid).sum(dtype=torch.float64)
        tn = ((preds == 0) & (y == 0) & valid).sum(dtype=torch.float64)
        fp = ((preds == 1) & (y == 0) & valid).sum(dtype=torch.float64)
        fn = ((preds == 0) & (y == 1) & valid).sum(dtype=torch.float64)

        vec = torch.empty((4,), device=dev, dtype=torch.float64)
        vec[0] = tp
        vec[1] = tn
        vec[2] = fp
        vec[3] = fn
        self._v.update(vec)

    def sync(self):
        self._v.sync()

    def compute(self):
        tp, tn, fp, fn = self._v.snapshot().tolist()
        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if den <= 0:
            return 0.0
        return float(num / (den ** 0.5))


class F1Binary(_Meter):
    # accum [tp, fp, fn]
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)
        self._v = _AsyncVecAdder(dim=3, dtype=torch.float64)

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, logits, labels):
        dev = logits.device
        preds = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0).to(torch.long)

        y = _normalize_class_labels(labels, device=dev, ignore_index=self.ignore_index)
        preds = preds.view(-1)
        y = y.view(-1)

        if preds.numel() != y.numel():
            raise ValueError(f"preds/labels size mismatch: {preds.numel()} vs {y.numel()}")

        valid = _valid_mask(y, self.ignore_index)

        tp = ((preds == 1) & (y == 1) & valid).sum(dtype=torch.float64)
        fp = ((preds == 1) & (y == 0) & valid).sum(dtype=torch.float64)
        fn = ((preds == 0) & (y == 1) & valid).sum(dtype=torch.float64)

        vec = torch.empty((3,), device=dev, dtype=torch.float64)
        vec[0] = tp
        vec[1] = fp
        vec[2] = fn
        self._v.update(vec)

    def sync(self):
        self._v.sync()

    def compute(self):
        tp, fp, fn = self._v.snapshot().tolist()
        denom = 2 * tp + fp + fn
        return 0.0 if denom <= 0 else float((2 * tp) / denom)


class PearsonR(_Meter):
    """
    Streaming Pearson: accum [n, sx, sy, sxx, syy, sxy] as float64 scalars.
    """
    def __init__(self):
        self._v = _AsyncVecAdder(dim=6, dtype=torch.float64)

    def reset(self):
        self._v.reset()

    @torch.no_grad()
    def update(self, preds, labels):
        x0 = _as_tensor(preds)
        y0 = _as_tensor(labels)

        dev = x0.device
        x = x0.to(device=dev, non_blocking=True).view(-1)
        y = y0.to(device=dev, non_blocking=True).view(-1)

        if x.numel() != y.numel():
            raise ValueError(f"preds/labels size mismatch: {x.numel()} vs {y.numel()}")

        n = torch.full((), x.numel(), device=dev, dtype=torch.float64)
        sx = x.sum(dtype=torch.float64)
        sy = y.sum(dtype=torch.float64)
        sxx = (x * x).sum(dtype=torch.float64)
        syy = (y * y).sum(dtype=torch.float64)
        sxy = (x * y).sum(dtype=torch.float64)

        vec = torch.empty((6,), device=dev, dtype=torch.float64)
        vec[0] = n
        vec[1] = sx
        vec[2] = sy
        vec[3] = sxx
        vec[4] = syy
        vec[5] = sxy
        self._v.update(vec)

    def sync(self):
        self._v.sync()

    def compute(self):
        n, sx, sy, sxx, syy, sxy = self._v.snapshot().tolist()
        if n <= 0:
            return 0.0
        num = n * sxy - sx * sy
        denx = n * sxx - sx * sx
        deny = n * syy - sy * sy
        den = denx * deny
        if den <= 1e-12:
            return 0.0
        return float(num / (den ** 0.5))


class GlueTextTask:
    """
    Auto-selects loss/metrics per GLUE task.
    """
    def __init__(self, task_name: str, *, ignore_index: int = -100):
        self.task_name = task_name.lower()
        self.ignore_index = int(ignore_index)

        if self.task_name == "stsb":
            self.mode = "regression"
            self._metrics: Dict[str, _Meter] = {"pearson": PearsonR()}
        elif self.task_name == "cola":
            self.mode = "classification"
            self._metrics = {"mcc": MCC(ignore_index=self.ignore_index)}
        elif self.task_name in {"mrpc", "qqp"}:
            self.mode = "classification"
            self._metrics = {
                "f1": F1Binary(ignore_index=self.ignore_index),
                "acc": Accuracy(ignore_index=self.ignore_index),
            }
        else:
            self.mode = "classification"
            self._metrics = {"acc": Accuracy(ignore_index=self.ignore_index)}

    def loss(self, outputs, batch):
        if "labels" not in batch:
            raise KeyError("batch missing 'labels'")

        if self.mode == "regression":
            preds = outputs.logits.squeeze(-1).view(-1)
            y = _as_tensor(batch["labels"]).to(preds.device, dtype=torch.float32, non_blocking=True).view(-1)
            if preds.numel() != y.numel():
                raise ValueError(f"preds/labels size mismatch: {preds.numel()} vs {y.numel()}")
            return F.mse_loss(preds, y)

        logits = outputs.logits
        y = _normalize_class_labels(batch["labels"], device=logits.device, ignore_index=self.ignore_index)
        return F.cross_entropy(logits, y, ignore_index=self.ignore_index)

    def loss_eval(self, outputs, batch):
        return self.loss(outputs, batch)

    def metrics(self):
        return self._metrics

    def reset_metrics(self):
        for m in self._metrics.values():
            m.reset()

    def sync_metrics(self):
        for m in self._metrics.values():
            try:
                m.sync()
            except Exception:
                pass

    @torch.no_grad()
    def update_metrics(self, outputs, batch):
        if "labels" not in batch:
            return

        if self.mode == "regression":
            preds = outputs.logits.squeeze(-1)
            self._metrics["pearson"].update(preds, batch["labels"])
        elif self.task_name == "cola":
            self._metrics["mcc"].update(outputs.logits, batch["labels"])
        elif self.task_name in {"mrpc", "qqp"}:
            self._metrics["f1"].update(outputs.logits, batch["labels"])
            self._metrics["acc"].update(outputs.logits, batch["labels"])
        else:
            self._metrics["acc"].update(outputs.logits, batch["labels"])

    def update_metrics_eval(self, outputs, batch):
        self.update_metrics(outputs, batch)

    def default_metrics_and_primary(self):
        t = self.task_name
        if t == "stsb":
            return ("pearson", True)
        if t == "cola":
            return ("mcc", True)
        if t in {"mrpc", "qqp"}:
            return ("f1", True)
        return ("acc", True)