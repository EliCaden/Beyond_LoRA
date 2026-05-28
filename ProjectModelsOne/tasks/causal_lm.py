# tasks/causal_lm.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import math
import os

import torch
import torch.nn.functional as F

__all__ = ["CausalLMTask"]

_FAST_EVAL_METRICS = os.getenv("FAST_EVAL_METRICS", "0").lower() in {"1", "true", "yes", "y"}


def _shifted_ce_sum_and_count(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - sum CE over valid tokens (ignore_index=-100), scalar float tensor on logits.device
      - number of valid tokens, int64 scalar tensor on logits.device
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Count valid tokens (int64 scalar on device)
    n_valid = shift_labels.ne(-100).sum(dtype=torch.int64)

    # Sum loss directly (avoid mean + multiply)
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, n_valid


def _safe_mean(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    # count is int64 scalar on same device; avoid div-by-zero
    denom = count.to(dtype=loss_sum.dtype).clamp_min_(1.0)
    return loss_sum / denom


class _AsyncScalarSumCount:
    """
    Accumulate (sum, count) without per-batch GPU sync.

    FAST_EVAL_METRICS=1: compute() drains only READY copies, never blocks on GPU.
    Otherwise: compute() waits for all pending copies (event.synchronize()).

    Notes:
      - Uses pinned host buffers + CUDA events for async D2H.
      - Keeps pending queue bounded and reuses host buffers via a small pool.
    """
    _MAX_PENDING = 256

    def __init__(self):
        self._sum_cpu: float = 0.0
        self._count_cpu: int = 0
        self._pending: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        self._pool: List[torch.Tensor] = []  # pinned CPU buffers of shape [2], float64

    def reset(self):
        self._sum_cpu = 0.0
        self._count_cpu = 0
        self._pending.clear()
        self._pool.clear()

    def _alloc_host2(self) -> torch.Tensor:
        if self._pool:
            return self._pool.pop()
        return torch.empty((2,), device="cpu", dtype=torch.float64, pin_memory=True)

    @torch.no_grad()
    def update(self, sum_value: torch.Tensor, count_value: torch.Tensor):
        # both should be scalar tensors on same device
        sv = sum_value.detach()
        cv = count_value.detach()

        if sv.is_cuda:
            # Pack into float64 on device without stack overhead.
            v = torch.empty((2,), device=sv.device, dtype=torch.float64)
            v[0] = sv.to(dtype=torch.float64)
            v[1] = cv.to(dtype=torch.float64)

            host = self._alloc_host2()
            host.copy_(v, non_blocking=True)

            ev = torch.cuda.Event(enable_timing=False)
            ev.record(torch.cuda.current_stream())
            self._pending.append((host, ev))

            # keep the pending list bounded without blocking
            if len(self._pending) > self._MAX_PENDING:
                self._drain(sync=False)
        else:
            # CPU path (may sync if sv/cv are CPU tensors requiring item(), but no GPU)
            self._sum_cpu += float(sv.to(dtype=torch.float64).item())
            self._count_cpu += int(cv.to(dtype=torch.int64).item())

    def _drain(self, sync: bool):
        if not self._pending:
            return

        keep: List[Tuple[torch.Tensor, "torch.cuda.Event"]] = []
        for host, ev in self._pending:
            if sync:
                ev.synchronize()
                self._sum_cpu += float(host[0].item())
                self._count_cpu += int(host[1].item())
                self._pool.append(host)
            else:
                if ev.query():
                    self._sum_cpu += float(host[0].item())
                    self._count_cpu += int(host[1].item())
                    self._pool.append(host)
                else:
                    keep.append((host, ev))

        self._pending = keep

    def sync(self):
        self._drain(sync=True)

    def compute(self) -> float:
        self._drain(sync=(not _FAST_EVAL_METRICS))
        if self._count_cpu <= 0:
            return 0.0
        return float(self._sum_cpu / self._count_cpu)


class _PPLFromLoss:
    def __init__(self, loss_meter: _AsyncScalarSumCount):
        self._loss_meter = loss_meter

    def sync(self):
        self._loss_meter.sync()

    def compute(self) -> float:
        l = float(self._loss_meter.compute())
        return float(math.exp(l)) if l < 50 else float("inf")


class CausalLMTask:
    def __init__(self):
        self._loss = _AsyncScalarSumCount()
        # Cache eval-only scalars so update_metrics_eval doesn't recompute CE.
        self._cached_eval_sum: Optional[torch.Tensor] = None
        self._cached_eval_ntok: Optional[torch.Tensor] = None

    def default_metrics_and_primary(self):
        return "loss", False  # minimize loss

    def reset_metrics(self):
        self._loss.reset()
        self._cached_eval_sum = None
        self._cached_eval_ntok = None

    def sync_metrics(self):
        self._loss.sync()

    def metrics(self) -> Dict[str, Any]:
        return {"loss": self._loss, "ppl": _PPLFromLoss(self._loss)}

    def _labels_on_device(self, labels: torch.Tensor, dev: torch.device) -> torch.Tensor:
        if labels.device == dev:
            return labels
        return labels.to(device=dev, non_blocking=True)

    def loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        logits = getattr(outputs, "logits", outputs)
        labels = self._labels_on_device(batch["labels"], logits.device)
        loss_sum, n_valid = _shifted_ce_sum_and_count(logits, labels)
        return _safe_mean(loss_sum, n_valid)

    def loss_eval(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Eval loss used by trainer; also caches (sum,count) so update_metrics_eval is cheap.
        """
        logits = getattr(outputs, "logits", outputs)
        labels = self._labels_on_device(batch["labels"], logits.device)
        loss_sum, n_valid = _shifted_ce_sum_and_count(logits, labels)
        self._cached_eval_sum = loss_sum.detach()
        self._cached_eval_ntok = n_valid.detach()
        return _safe_mean(loss_sum, n_valid)

    @torch.no_grad()
    def update_metrics(self, outputs: Any, batch: Dict[str, Any]):
        logits = getattr(outputs, "logits", outputs)
        labels = self._labels_on_device(batch["labels"], logits.device)
        loss_sum, n_valid = _shifted_ce_sum_and_count(logits, labels)
        self._loss.update(loss_sum, n_valid)

    @torch.no_grad()
    def update_metrics_eval(self, outputs: Any, batch: Dict[str, Any]):
        if self._cached_eval_sum is not None and self._cached_eval_ntok is not None:
            loss_sum = self._cached_eval_sum
            n_valid = self._cached_eval_ntok
            self._cached_eval_sum = None
            self._cached_eval_ntok = None
        else:
            logits = getattr(outputs, "logits", outputs)
            labels = self._labels_on_device(batch["labels"], logits.device)
            loss_sum, n_valid = _shifted_ce_sum_and_count(logits, labels)

        self._loss.update(loss_sum, n_valid)
