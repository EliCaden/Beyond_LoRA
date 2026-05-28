# trainers/generic_trainer.py
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .callbacks import Callback


# ------------------------------
# helpers
# ------------------------------
def _now() -> float:
    return time.perf_counter()


class NonFiniteError(RuntimeError):
    """Raised when a non-finite loss is detected and stop_on_nan=True."""


def _set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    try:
        import numpy as np
    except Exception:
        np = None

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (can be expensive).
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = (not deterministic)
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass


def _configure_fast_cuda_perf_flags() -> None:
    """
    Best-effort perf flags for fast_mode (may slightly change numerics).
    """
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _get_git_sha() -> Optional[str]:
    try:
        import subprocess

        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or None
    except Exception:
        return None


def _recursive_to_device(x: Any, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        items = [_recursive_to_device(v, device) for v in x]
        return tuple(items) if isinstance(x, tuple) else type(x)(items)
    return x


def _compute_grad_norm_tensor(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Global grad norm across trainable params.
    Returns a DEVICE scalar tensor (no CPU sync).
    """
    total = None
    dev = None
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach()
            dev = g.device
            if total is None:
                total = torch.zeros((), device=dev, dtype=torch.float32)
            n = g.float().norm(norm_type)
            total += n.pow(norm_type)

    if total is None:
        dev = dev or torch.device("cpu")
        return torch.zeros((), device=dev, dtype=torch.float32)

    return total.pow(1.0 / norm_type)


def _is_lora_key(name: str) -> bool:
    ln = name.lower()
    last = ln.rsplit(".", 1)[-1]
    return (last in {"a", "b", "a_q", "b_q", "a_v", "b_v"}) or ("lora" in ln)


def _is_paca_key(name: str) -> bool:
    ln = name.lower()
    return ("paca_cols_weight" in ln) or ("paca_w" in ln)


def _is_head_key(name: str) -> bool:
    ln = name.lower()
    return (
        "classifier" in ln
        or ln.endswith("lm_head.weight")
        or ln.endswith("lm_head.bias")
        or ".lm_head." in ln
        or ".classifier." in ln
        or ln.endswith(".score.weight")
        or ln.endswith(".score.bias")
        or ".score." in ln
        or ln.endswith(".head.weight")
        or ln.endswith(".head.bias")
        or ".head." in ln
    )


class _AsyncScalarD2H:
    """
    Deferred GPU->CPU scalar meter:
      - Enqueue non_blocking D2H copy into pinned host scalar
      - Record CUDA event
      - poll() returns latest completed value without synchronizing the GPU
    """

    def __init__(self, *, enabled: bool, nbuf: int = 4, dtype: torch.dtype = torch.float32):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.dtype = dtype
        self._latest: Optional[float] = None

        self._nbuf = max(2, int(nbuf))
        self._buf: List[Optional[torch.Tensor]] = []
        self._ev: List[Optional[torch.cuda.Event]] = []
        self._idx = 0
        self._pending_idx: Optional[int] = None
        self._has_pending = False

        if self.enabled:
            try:
                self._buf = [
                    torch.empty((), device="cpu", dtype=self.dtype, pin_memory=True)
                    for _ in range(self._nbuf)
                ]
                self._ev = [torch.cuda.Event(enable_timing=False) for _ in range(self._nbuf)]
            except Exception:
                self.enabled = False
                self._buf = []
                self._ev = []

    def push(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.numel() != 1:
            return

        if not self.enabled:
            # Only safe to read if CPU; avoid accidental CUDA sync here.
            if x.device.type == "cpu":
                try:
                    self._latest = float(x.detach().to(dtype=self.dtype).item())
                except Exception:
                    self._latest = None
            return

        if x.device.type != "cuda":
            try:
                self._latest = float(x.detach().to(dtype=self.dtype).item())
            except Exception:
                self._latest = None
            return

        i = self._idx
        self._idx = (self._idx + 1) % self._nbuf
        buf = self._buf[i]
        ev = self._ev[i]
        if buf is None or ev is None:
            return

        try:
            buf.copy_(x.detach().to(dtype=self.dtype), non_blocking=True)
            ev.record()
            self._pending_idx = i
            self._has_pending = True
        except Exception:
            pass

    def poll(self) -> Optional[float]:
        if not self.enabled:
            return self._latest

        if self._has_pending and self._pending_idx is not None:
            ev = self._ev[self._pending_idx]
            buf = self._buf[self._pending_idx]
            if ev is not None and buf is not None:
                try:
                    if ev.query():
                        self._latest = float(buf.item())
                        self._has_pending = False
                        self._pending_idx = None
                except Exception:
                    pass
        return self._latest


class _CudaScalarSync:
    """
    Coalesce many CUDA scalar reads into ONE synchronization per boundary.

    Usage:
      vals = syncer.read({"loss": loss_t, "gn": gn_t, ...})
      -> returns python floats
    """

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._cpu_buf: Optional[torch.Tensor] = None
        self._ev: Optional[torch.cuda.Event] = None

        if self.enabled:
            try:
                self._ev = torch.cuda.Event(enable_timing=False)
            except Exception:
                self.enabled = False
                self._ev = None

    def read(self, scalars: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not scalars:
            return out

        cuda_items: List[Tuple[str, torch.Tensor]] = []
        for k, t in scalars.items():
            if not isinstance(t, torch.Tensor) or t.numel() != 1:
                continue
            if t.device.type == "cuda" and self.enabled:
                cuda_items.append((str(k), t.detach()))
            elif t.device.type == "cpu":
                try:
                    out[str(k)] = float(t.detach().float().item())
                except Exception:
                    pass

        if not cuda_items:
            return out

        keys = [k for (k, _) in cuda_items]
        ts = [t.float().reshape(()) for (_, t) in cuda_items]

        try:
            vec = torch.stack(ts, dim=0)  # CUDA tensor
        except Exception:
            # Fallback (may sync multiple times), only at boundary.
            for k, t in cuda_items:
                try:
                    out[k] = float(t.float().item())
                except Exception:
                    pass
            return out

        n = int(vec.numel())
        try:
            if self._cpu_buf is None or int(self._cpu_buf.numel()) != n:
                self._cpu_buf = torch.empty((n,), device="cpu", dtype=torch.float32, pin_memory=True)
        except Exception:
            cpu = vec.detach().cpu()
            vals = cpu.tolist()
            for k, v in zip(keys, vals):
                out[k] = float(v)
            return out

        try:
            self._cpu_buf.copy_(vec, non_blocking=True)
            if self._ev is not None:
                self._ev.record()
                self._ev.synchronize()  # exactly one sync for this boundary
            vals = self._cpu_buf.tolist()
            for k, v in zip(keys, vals):
                out[k] = float(v)
        except Exception:
            cpu = vec.detach().cpu()
            vals = cpu.tolist()
            for k, v in zip(keys, vals):
                out[k] = float(v)

        return out


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_amp: bool = True
    prefer_bf16: Optional[bool] = None
    grad_accum_steps: int = 1
    scheduler: str | None = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    save_dir: str = "ckpts"
    save_name_prefix: str = "model"

    # Saving policy (model-only checkpoints; no resume support)
    save_all_epochs: bool = True
    save_epoch_every: Optional[int] = None
    epochs_dir_name: str = "epochs"
    save_checkpoints: bool = True
    save_best_to_disk: bool = True
    save_init_checkpoint: bool = True

    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    # Logging frequency controls (respected even in fast_mode)
    train_log_interval: float | int | None = 0.2
    eval_log_interval: float | int | None = 0.2

    # Non-finite detection
    detect_nonfinite: bool = True
    nonfinite_sync: str = "epoch"  # "epoch" | "off" | "batch"
    stop_on_nan: bool = True
    nonfinite_skip_eval: bool = True
    nonfinite_skip_save: bool = True
    nonfinite_skip_test: bool = True

    seed: Optional[int] = 42
    min_lr: Optional[float] = None
    warmup_lr: Optional[float] = 1e-6

    # trims expensive logging/diagnostics
    fast_mode: bool = False
    fast_mode_print_every: Optional[int] = None

    # If True, avoid explicit GPU sync at eval boundaries. (Metrics/score may be NaN/unavailable.)
    fast_mode_no_gpu_sync: bool = False

    # fast-mode eval schedule:
    #   0  -> skip eval during training (but may still eval on last epoch if needed)
    #   k>0 -> eval every k epochs (and always on last epoch)
    fast_mode_eval_every: int = -1

    # fast-mode eval: if True compute only loss (skips update_metrics_eval + compute()) unless needed.
    fast_mode_eval_loss_only: bool = True

    # If True, auto turn off epoch-level nonfinite sync in fast_mode (keeps batch option if user set it).
    # SAFETY: will NOT disable nonfinite sync if stop_on_nan=True.
    fast_mode_disable_nonfinite_sync: bool = True

    # If True, disable callback-driven early stopping checks in fast_mode.
    fast_mode_disable_early_stopping: bool = True

    # If True, do one synced eval after training ends (fast_mode only).
    fast_mode_final_sync_eval: bool = False

    # Best-model selection policy
    best_metric: str = "primary"  # "loss" | "acc" | "primary" | "<metric_key>"

    fast_mode_eval_on_last_epoch_for_best: bool = True
    fast_mode_force_sync_for_best: bool = True

    # If True, in fast_mode, only keep primary/selection/acc (still enough for correct best selection).
    # Set to False to log full metrics in fast_mode.
    fast_mode_primary_metric_only: bool = False

    always_compute_acc: bool = True


class GenericTrainer:
    """
    Trainer with explicit GPU/CPU sync minimization.

    Key guarantees:
      - fast_mode does NOT silently disable NaN/Inf aborts when stop_on_nan=True.
      - nonfinite_sync='batch' influences epoch-level skip/save/test policy and aborts safely.
      - Best-model selection is correct even when best_metric is a custom metric key.
      - Fast mode still produces useful progress logs without per-step CUDA synchronizations.
      - Checkpoints are model-only (no resume support).
    """

    def __init__(
        self,
        model: nn.Module,
        method,
        task_logic,
        train_loader,
        val_loader,
        device: torch.device,
        tcfg: Optional[TrainConfig] = None,
        method_cfg_lr_override: Optional[float] = None,
        test_loader=None,
        callbacks: Optional[List[Callback]] = None,
        data_module=None,
    ):
        self.model = model
        self.method = method
        self.task = task_logic
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_module = data_module
        self.device = device
        self.cfg = tcfg or TrainConfig()
        self.fast_mode = bool(getattr(self.cfg, "fast_mode", False))

        self._cuda_enabled = (self.device.type == "cuda" and torch.cuda.is_available())
        self._scalar_sync = _CudaScalarSync(enabled=self._cuda_enabled)

        # Seeds + perf flags
        if self.cfg.seed is not None:
            _set_global_seed(int(self.cfg.seed), deterministic=(False))
        if self.fast_mode:
            _configure_fast_cuda_perf_flags()

        # SAFETY: don't let fast mode silently disable nonfinite detection if stop_on_nan=True
        if self.fast_mode and bool(getattr(self.cfg, "fast_mode_disable_nonfinite_sync", True)):
            mode = str(getattr(self.cfg, "nonfinite_sync", "epoch")).lower().strip()
            if mode == "epoch" and (not bool(getattr(self.cfg, "stop_on_nan", True))):
                self.cfg.nonfinite_sync = "off"

        self.git_commit = _get_git_sha()
        self.callbacks: List[Callback] = callbacks or []

        # Configure model then move to device
        self.method.configure_model(self.model)
        self.model.to(self.device)
        self._refresh_notify_modules()

        # AMP device selection MUST match training device
        self.amp_device_type = "cuda" if self._cuda_enabled else "cpu"
        if self.cfg.prefer_bf16 is None:
            use_bf16 = bool(self._cuda_enabled and torch.cuda.is_bf16_supported())
        else:
            use_bf16 = bool(self.cfg.prefer_bf16)
        self._amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # counts
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.trainable_fraction = (self.trainable_params / self.total_params) if self.total_params > 0 else 0.0
        self.reduction_factor_vs_full = (self.total_params / self.trainable_params) if self.trainable_params > 0 else None

        # runtime series
        self.epoch_durations: List[float] = []
        self.epoch_train_seen: List[int] = []
        self.epoch_train_tokens: List[int] = []
        self.epoch_train_opt_steps: List[int] = []
        self.epoch_train_wall_sec: List[float] = []
        self.epoch_train_samples_per_s: List[float] = []
        self.epoch_train_tokens_per_s: List[float] = []

        self.global_step = 0  # microbatches processed (batches), across epochs
        self._opt_step = 0    # optimizer steps, across epochs

        # grad-norm cache
        self._last_grad_norm_t: Optional[torch.Tensor] = None
        self._last_grad_norm: Optional[float] = None
        self._last_amp_scale: Optional[float] = None

        # wall timing fields
        self._last_epoch_sec: Optional[float] = None
        self._last_epoch_train_sec: Optional[float] = None
        self._last_eval_sec: Optional[float] = None
        self._last_save_sec: Optional[float] = None

        self._last_epoch_train_seen: Optional[int] = None
        self._last_epoch_train_tokens: Optional[int] = None
        self._last_epoch_train_opt_steps: Optional[int] = None

        self._last_val_loss: Optional[float] = None
        self._test_loss: Optional[float] = None
        self._test_metrics: Optional[Dict[str, Any]] = None

        # Cache trainable params list (refreshable)
        self._refresh_trainable_params_list()

        # GPU-resident nonfinite flag (no sync on update)
        self._nonfinite_flag = torch.zeros((), device=self.device, dtype=torch.uint8)
        self._ever_nonfinite = False
        self._nonfinite_epochs: List[int] = []
        self._nonfinite_hit_this_epoch: bool = False

        # async meters (progress prints; sync-safe)
        self._train_loss_meter = _AsyncScalarD2H(enabled=self._cuda_enabled)
        self._eval_loss_meter = _AsyncScalarD2H(enabled=self._cuda_enabled)
        self._grad_norm_meter = _AsyncScalarD2H(enabled=self._cuda_enabled)

        # base lr/wd
        lr = self.cfg.lr
        if hasattr(method, "cfg") and hasattr(method.cfg, "lr"):
            try:
                lr = float(method.cfg.lr)
            except Exception:
                pass
        if method_cfg_lr_override is not None:
            lr = float(method_cfg_lr_override)

        wd = self.cfg.weight_decay
        if hasattr(method, "cfg") and hasattr(method.cfg, "weight_decay"):
            try:
                wd = float(method.cfg.weight_decay)
            except Exception:
                pass

        self._base_lr = float(lr)
        self._base_wd = float(wd)

        # Build optimizer + AMP scaler
        self.opt = self._make_optimizer()

        # GradScaler is useful for FP16; typically unnecessary for BF16.
        scaler_enabled = bool(
            self.cfg.use_amp
            and (self.amp_device_type == "cuda")
            and (self._amp_dtype == torch.float16)
        )
        self.scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=scaler_enabled,
        )

        # Task primary metric
        self.primary_metric, self.maximize = self.task.default_metrics_and_primary()

        # Best selection policy
        self.best_selection_name, self.best_selection_maximize = self._resolve_best_selection_policy()
        self.best_selection_value = (-float("inf") if self.best_selection_maximize else float("inf"))
        self.best_score = float(self.best_selection_value)  # backward-compatible

        self.best_path: Optional[str] = None
        self.best_val_loss: Optional[float] = None
        self.best_val_metrics: Optional[Dict[str, Any]] = None
        self.best_epoch: Optional[int] = None
        self.last_train_epoch_loss: Optional[float] = None

        self._stop_training = False

        # Scheduler build
        self._scheduler = None
        self._sched_meta: Dict[str, Any] = {"name": self.cfg.scheduler, "total_steps": None, "warmup_steps": None}

        try:
            train_total_iters = len(self.train_loader)
        except TypeError:
            train_total_iters = None

        sched_reason = None
        if not self.cfg.scheduler:
            sched_reason = "disabled_by_config"
        elif train_total_iters is None:
            sched_reason = "unknown_total_train_iters"

        base_lr0 = float(self.opt.param_groups[0]["lr"])
        warmup_lr = float(self.cfg.warmup_lr if self.cfg.warmup_lr is not None else 1e-6)
        start_factor = max(1e-8, min(1.0, warmup_lr / max(1e-12, base_lr0)))

        if self.cfg.scheduler and train_total_iters is not None:
            max_train_iters = train_total_iters
            if self.cfg.limit_train_batches is not None:
                max_train_iters = min(max_train_iters, int(self.cfg.limit_train_batches))
            steps_per_epoch = math.ceil(max_train_iters / max(1, int(self.cfg.grad_accum_steps)))
            total_steps = steps_per_epoch * int(self.cfg.epochs)
            warmup_steps = (
                int(self.cfg.warmup_steps)
                if self.cfg.warmup_steps is not None
                else int(float(self.cfg.warmup_ratio) * total_steps)
            )

            name = (self.cfg.scheduler or "").lower()
            if name == "cosine":
                scheds = []
                milestones = []
                if warmup_steps > 0:
                    scheds.append(LinearLR(self.opt, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps))
                    milestones.append(warmup_steps)
                eta_min = float(self.cfg.min_lr) if self.cfg.min_lr is not None else 0.0
                T_max = max(1, total_steps - warmup_steps)
                scheds.append(CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min))
                self._scheduler = SequentialLR(self.opt, schedulers=scheds, milestones=milestones) if milestones else scheds[0]

            elif name == "linear":
                end_factor = 0.0
                if self.cfg.min_lr is not None:
                    end_factor = max(0.0, float(self.cfg.min_lr) / max(1e-12, base_lr0))
                scheds = []
                milestones = []
                if warmup_steps > 0:
                    scheds.append(LinearLR(self.opt, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps))
                    milestones.append(warmup_steps)
                remain = max(1, total_steps - warmup_steps)
                scheds.append(LinearLR(self.opt, start_factor=1.0, end_factor=end_factor, total_iters=remain))
                self._scheduler = SequentialLR(self.opt, schedulers=scheds, milestones=milestones) if milestones else scheds[0]
            else:
                self._scheduler = None

            self._sched_meta = {
                "name": self.cfg.scheduler,
                "total_steps": int(total_steps),
                "warmup_steps": int(warmup_steps),
                "min_lr": float(self.cfg.min_lr) if self.cfg.min_lr is not None else None,
                "reason": None,
            }
        else:
            self._sched_meta = {
                "name": self.cfg.scheduler,
                "total_steps": None,
                "warmup_steps": None,
                "min_lr": float(self.cfg.min_lr) if self.cfg.min_lr is not None else None,
                "reason": sched_reason,
            }

        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.run_dir = os.path.join(self.cfg.save_dir, self.cfg.save_name_prefix)
        os.makedirs(self.run_dir, exist_ok=True)

        # cheap run metadata
        self.run_meta = self._build_run_meta()
        try:
            with open(os.path.join(self.run_dir, "run_meta.json"), "w") as f:
                json.dump(self.run_meta, f, indent=2)
        except Exception:
            pass

    # ------------------------------
    # utilities
    # ------------------------------
    def _refresh_trainable_params_list(self) -> None:
        self._trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]

    def _refresh_notify_modules(self) -> None:
        mods = []
        for m in self.model.modules():
            fn = getattr(m, "notify_optimizer_step", None)
            if callable(fn):
                mods.append(m)
        self._notify_modules = mods

    def _autocast_ctx(self):
        return torch.amp.autocast(
            self.amp_device_type,
            enabled=bool(self.cfg.use_amp and self.amp_device_type == "cuda"),
            dtype=self._amp_dtype,
        )

    def _split_batch(self, batch: Dict[str, Any]):
        labels = _recursive_to_device(batch["labels"], self.device)
        inputs = {k: _recursive_to_device(v, self.device) for k, v in batch.items() if k != "labels"}
        return inputs, labels

    @staticmethod
    def _fmt_eta(seconds: Optional[float]) -> str:
        if seconds is None:
            return ""
        try:
            total = int(max(0, round(seconds)))
        except Exception:
            return ""

        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)

        parts: List[str] = []
        if days:
            parts.append(f"{days}d")
            parts.append(f"{hours:02d}h")
            parts.append(f"{minutes:02d}m")
            parts.append(f"{secs:02d}s")
        elif hours:
            parts.append(f"{hours}h")
            parts.append(f"{minutes:02d}m")
            parts.append(f"{secs:02d}s")
        elif minutes:
            parts.append(f"{minutes}m")
            parts.append(f"{secs:02d}s")
        else:
            parts.append(f"{secs}s")
        return ", eta=" + " ".join(parts)

    def _resolve_log_every(self, setting: float | int | None, total_iters: int | None) -> int | None:
        if setting is None:
            return None
        if isinstance(setting, float):
            f = float(setting)
            if not (0.0 < f <= 1.0):
                return None
            if total_iters is None:
                return 50
            return max(1, int(math.ceil(f * max(1, total_iters))))
        if isinstance(setting, int):
            return max(1, setting)
        return None

    def _loader_info(self, loader) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if loader is None:
            return info
        try:
            info["num_batches"] = int(len(loader))
        except Exception:
            info["num_batches"] = None
        try:
            ds = getattr(loader, "dataset", None)
            info["num_samples"] = int(len(ds)) if ds is not None else None
        except Exception:
            info["num_samples"] = None
        for k in ["batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "drop_last"]:
            try:
                v = getattr(loader, k)
                info[k] = bool(v) if isinstance(v, bool) else (int(v) if isinstance(v, int) else v)
            except Exception:
                pass
        return info

    def _trainable_param_breakdown(self) -> Dict[str, Any]:
        out = {"total_trainable": int(self.trainable_params)}
        buckets = {"lora": 0, "paca": 0, "head": 0, "other": 0}
        mod_paths = set()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            try:
                mod_paths.add(n.rsplit(".", 1)[0])
            except Exception:
                pass
            if _is_lora_key(n):
                buckets["lora"] += int(p.numel())
            elif _is_paca_key(n):
                buckets["paca"] += int(p.numel())
            elif _is_head_key(n):
                buckets["head"] += int(p.numel())
            else:
                buckets["other"] += int(p.numel())
        out["by_kind_params"] = {k: int(v) for k, v in buckets.items()}
        out["by_kind_frac"] = {
            k: (float(v) / float(self.trainable_params) if self.trainable_params else 0.0) for k, v in buckets.items()
        }
        out["num_trainable_module_paths"] = int(len(mod_paths))
        out["trainable_module_paths"] = sorted(list(mod_paths))[:200]
        out["trainable_module_paths_truncated"] = bool(len(mod_paths) > 200)
        return out

    def _build_run_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        meta["git_commit"] = self.git_commit
        meta["device"] = str(self.device)
        meta["amp"] = bool(self.scaler.is_enabled())
        meta["amp_dtype"] = str(self._amp_dtype)
        meta["fast_mode"] = bool(self.fast_mode)
        meta["scheduler"] = dict(self._sched_meta)

        meta["loaders"] = {
            "train": self._loader_info(self.train_loader),
            "val": self._loader_info(self.val_loader),
            "test": self._loader_info(self.test_loader) if self.test_loader is not None else None,
        }

        bs = meta["loaders"]["train"].get("batch_size", None)
        gas = int(getattr(self.cfg, "grad_accum_steps", 1) or 1)
        try:
            meta["effective_batch_size"] = int(bs) * int(gas) if bs is not None else None
        except Exception:
            meta["effective_batch_size"] = None

        meta["params"] = {
            "total": int(self.total_params),
            "trainable": int(self.trainable_params),
            "trainable_fraction": float(self.trainable_fraction),
            "reduction_factor_vs_full": float(self.reduction_factor_vs_full) if self.reduction_factor_vs_full else None,
        }
        meta["trainable_breakdown"] = self._trainable_param_breakdown()

        meta["best_selection"] = {
            "best_metric_cfg": str(getattr(self.cfg, "best_metric", "loss")),
            "resolved_name": str(self.best_selection_name),
            "maximize": bool(self.best_selection_maximize),
            "primary_metric": str(self.primary_metric),
            "primary_maximize": bool(self.maximize),
        }

        gpu_info = {}
        if torch.cuda.is_available() and self._cuda_enabled:
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "cc": f"{props.major}.{props.minor}",
                    "total_memory": int(getattr(props, "total_memory", 0)),
                }
            except Exception:
                pass

        meta["env"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": getattr(__import__("transformers"), "__version__", None),
            "gpu": gpu_info or None,
        }
        return meta

    # ------------------------------
    # best selection policy
    # ------------------------------
    def _resolve_best_selection_policy(self) -> Tuple[str, bool]:
        m = str(getattr(self.cfg, "best_metric", "loss")).strip().lower()
        if m in {"loss", "val_loss"}:
            return "loss", False
        if m in {"acc", "accuracy"}:
            return "acc", True
        if m in {"primary", "score"}:
            return str(self.primary_metric), bool(self.maximize)
        return m, True  # user-provided key (assume maximize)

    @staticmethod
    def _lower_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(metrics, dict):
            return {}
        out = {}
        for k, v in metrics.items():
            try:
                out[str(k).lower()] = v
            except Exception:
                pass
        return out

    def _extract_selection_value(self, val_loss: float, metrics: Dict[str, Any], score: float) -> Optional[float]:
        name = str(self.best_selection_name).strip().lower()
        ml = self._lower_metrics(metrics or {})

        if name == "loss":
            try:
                v = float(val_loss)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        if name == "acc":
            try:
                v = float(ml.get("acc", None))
                return v if math.isfinite(v) else None
            except Exception:
                return None

        if name in ml:
            try:
                v = float(ml[name])
                return v if math.isfinite(v) else None
            except Exception:
                pass

        try:
            v = float(score)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    def _is_better(self, v: Optional[float]) -> bool:
        if v is None:
            return False
        try:
            vv = float(v)
        except Exception:
            return False
        if not math.isfinite(vv):
            return False
        cur = float(self.best_selection_value)
        return (vv > cur) if self.best_selection_maximize else (vv < cur)

    # ------------------------------
    # non-finite detection
    # ------------------------------
    def _nonfinite_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "detect_nonfinite", True)):
            return False
        mode = str(getattr(self.cfg, "nonfinite_sync", "epoch")).lower().strip()
        return mode != "off"

    def _nonfinite_reset(self) -> None:
        try:
            self._nonfinite_flag.zero_()
        except Exception:
            pass

    def _nonfinite_update_from_loss(self, loss_tensor: torch.Tensor) -> None:
        if not self._nonfinite_enabled():
            return
        try:
            bad = (~torch.isfinite(loss_tensor.detach())).any()
            self._nonfinite_flag |= bad.to(dtype=self._nonfinite_flag.dtype)
        except Exception:
            pass

    def _nonfinite_sync_and_reset(self) -> bool:
        """
        Exactly one GPU->CPU sync boundary (scalar read). Use sparingly.
        """
        if not self._nonfinite_enabled():
            return False
        try:
            hit = bool(self._nonfinite_flag.item())
        except Exception:
            hit = False
        self._nonfinite_reset()
        if hit:
            self._ever_nonfinite = True
        return hit

    # ------------------------------
    # optimizer grouping / rebuild
    # ------------------------------
    def _build_param_groups(self, *, lr: float, wd: float, old_lrs: Optional[Dict[str, float]] = None):
        old_lrs = old_lrs or {}

        lora_a_params, lora_b_params, paca_params, head_params, other_params = [], [], [], [], []
        has_lora = False
        has_paca = False

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            last = name.rsplit(".", 1)[-1].lower()

            if _is_lora_key(name):
                has_lora = True
                if last in {"b", "b_q", "b_v"} or "lora_b" in name.lower() or last.startswith("b_"):
                    lora_b_params.append(p)
                else:
                    lora_a_params.append(p)
            elif _is_paca_key(name):
                has_paca = True
                paca_params.append(p)
            elif _is_head_key(name):
                head_params.append(p)
            else:
                other_params.append(p)

        self.has_lora = bool(has_lora)
        self.has_paca = bool(has_paca)
        has_adapter = self.has_lora or self.has_paca

        if has_adapter:
            param_groups = []

            if has_lora:
                try:
                    gamma = float(
                        getattr(self.method, "plus_gamma", None)
                        or getattr(getattr(self.method, "cfg", object()), "plus_gamma", None)
                    )
                except Exception:
                    gamma = None
                gamma = gamma if (gamma is not None and gamma > 0) else 1.0

                lr_a = lr / gamma
                lr_b = lr * gamma

                if lora_a_params:
                    param_groups.append(
                        {"name": "lora_a", "params": lora_a_params, "lr": float(old_lrs.get("lora_a", lr_a)), "weight_decay": 0.0}
                    )
                if lora_b_params:
                    param_groups.append(
                        {"name": "lora_b", "params": lora_b_params, "lr": float(old_lrs.get("lora_b", lr_b)), "weight_decay": 0.0}
                    )

            if paca_params:
                param_groups.append({"name": "paca", "params": paca_params, "lr": float(old_lrs.get("paca", lr)), "weight_decay": 0.0})

            if head_params:
                param_groups.append({"name": "head", "params": head_params, "lr": float(old_lrs.get("head", lr)), "weight_decay": 0.0})

            if other_params:
                param_groups.append(
                    {
                        "name": "other",
                        "params": other_params,
                        "lr": float(old_lrs.get("other", lr)),
                        "weight_decay": float(old_lrs.get("other_wd", wd)) if "other_wd" in old_lrs else float(wd),
                    }
                )

            return param_groups

        # Full fine-tune path
        decay, no_decay, head = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            lname = n.lower()
            if _is_head_key(n):
                head.append(p)
                continue
            if (
                lname.endswith("bias")
                or "norm" in lname
                or "layernorm" in lname
                or ".ln" in lname
                or "position_embeddings" in lname
                or "pos_embed" in lname
                or "cls_token" in lname
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = []
        if decay:
            param_groups.append({"name": "decay", "params": decay, "lr": float(old_lrs.get("decay", lr)), "weight_decay": float(wd)})
        if no_decay:
            param_groups.append({"name": "no_decay", "params": no_decay, "lr": float(old_lrs.get("no_decay", lr)), "weight_decay": 0.0})
        if head:
            param_groups.append({"name": "head", "params": head, "lr": float(old_lrs.get("head", lr * 10.0)), "weight_decay": float(wd)})
        return param_groups

    def _make_optimizer(self, *, old_lrs: Optional[Dict[str, float]] = None) -> AdamW:
        groups = self._build_param_groups(lr=float(self._base_lr), wd=float(self._base_wd), old_lrs=old_lrs)
        return AdamW(groups)

    def rebuild_optimizer_after_chain(self, *, keep_group_lrs: bool = True, disable_scheduler: bool = True):
        old_lrs: Dict[str, float] = {}
        if keep_group_lrs and getattr(self, "opt", None) is not None:
            try:
                for g in self.opt.param_groups:
                    n = g.get("name", None)
                    if n is not None:
                        old_lrs[str(n)] = float(g.get("lr", 0.0))
                    if g.get("name", None) == "other":
                        old_lrs["other_wd"] = float(g.get("weight_decay", self._base_wd))
            except Exception:
                old_lrs = {}

        self.opt = self._make_optimizer(old_lrs=old_lrs if keep_group_lrs else None)
        self._refresh_notify_modules()
        self._refresh_trainable_params_list()

        if disable_scheduler:
            self._scheduler = None
            self._sched_meta = {
                "name": self.cfg.scheduler,
                "total_steps": None,
                "warmup_steps": None,
                "min_lr": float(self.cfg.min_lr) if self.cfg.min_lr is not None else None,
                "reason": "disabled_after_chain_reset_optimizer",
            }

        try:
            self.opt.zero_grad(set_to_none=True)
        except Exception:
            pass

    def _notify_after_opt_step(self) -> None:
        for m in getattr(self, "_notify_modules", []):
            try:
                m.notify_optimizer_step()
            except Exception:
                pass

    def _mul_grads_(self, factor: float) -> None:
        """
        Multiply in-place all trainable grads by a scalar (device-side, no CPU sync).
        Used to fix remainder scaling when an epoch ends mid-accumulation cycle.
        """
        try:
            f = float(factor)
        except Exception:
            return
        if not math.isfinite(f) or f == 1.0:
            return
        for p in self._trainable_params_list:
            if p.grad is not None:
                try:
                    p.grad.mul_(f)
                except Exception:
                    pass

    # ------------------------------
    # loss + eval
    # ------------------------------
    @torch.no_grad()
    def compute_loss(self, batch):
        inputs, labels = self._split_batch(batch)
        with self._autocast_ctx():
            outputs = self.model(**inputs)
            loss = self.task.loss_eval(outputs, {"labels": labels})
        return loss.detach()

    def _step_batch(self, batch, *, accum_div: Optional[int] = None) -> torch.Tensor:
        """
        One microbatch forward/backward.
        FLOPs: if a callback exposes train-step FLOPs hooks, we bracket forward/backward so it can
        measure forward+backward once per input signature (cached), without counting optimizer ops.
        """
        inputs, labels = self._split_batch(batch)

        self._flops_stage = "train"

        # ---- FLOPs train-step begin (optional; no-ops if callback doesn't implement) ----
        for cb in self.callbacks:
            fn = getattr(cb, "on_flops_train_step_start", None)
            if callable(fn):
                try:
                    fn(self, inputs)
                except Exception:
                    pass

        try:
            with self._autocast_ctx():
                outputs = self.model(**inputs)
                loss = self.task.loss(outputs, {"labels": labels})
                div = max(1, int(accum_div if accum_div is not None else self.cfg.grad_accum_steps))
                loss_to_backprop = loss / div

            # ---- FLOPs: mark forward complete (before any backward work starts) ----
            for cb in self.callbacks:
                fn = getattr(cb, "on_flops_train_step_after_forward", None)
                if callable(fn):
                    try:
                        fn(self)
                    except Exception:
                        pass

            # Nonfinite tracking should look at the *unscaled* loss
            self._nonfinite_update_from_loss(loss)

            mode = str(getattr(self.cfg, "nonfinite_sync", "epoch")).lower().strip()
            if self._nonfinite_enabled() and mode == "batch":
                hit = self._nonfinite_sync_and_reset()
                if hit:
                    self._nonfinite_hit_this_epoch = True
                    if bool(getattr(self.cfg, "stop_on_nan", True)):
                        # Abort BEFORE backward to avoid contaminating grads/weights.
                        raise NonFiniteError("NaN/Inf loss encountered (nonfinite_sync='batch', stop_on_nan=True).")

            if self.scaler.is_enabled():
                self.scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            return loss.detach()

        finally:
            # ---- FLOPs train-step end (optional) ----
            for cb in self.callbacks:
                fn = getattr(cb, "on_flops_train_step_end", None)
                if callable(fn):
                    try:
                        fn(self)
                    except Exception:
                        pass


    @torch.no_grad()
    def _evaluate(
        self,
        loader=None,
        *,
        limit_batches: int | None = None,
        stage: str = "val",
        compute_metrics: bool = True,
        no_gpu_sync: bool = False,
    ):
        prev_stage = getattr(self, "_flops_stage", None)
        self._flops_stage = str(stage)
        
        try:
            loader = loader or self.val_loader
            was_training = bool(self.model.training)
            self.model.eval()

            try:
                self.task.reset_metrics()
            except Exception:
                pass

            try:
                eval_total_iters = len(loader)
            except TypeError:
                eval_total_iters = None

            prefix = {"train": "[TRAIN]", "val": "[VAL]", "test": "[TEST]"}.get(stage, "[EVAL]")
            log_every_eval = self._resolve_log_every(self.cfg.eval_log_interval, eval_total_iters)

            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            total_n = 0

            t_last = _now()
            last_i = 0
            last_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            last_n = 0

            metrics_obj = self.task.metrics() if compute_metrics else None

            for i, batch in enumerate(loader, 1):
                inputs, labels = self._split_batch(batch)
                with self._autocast_ctx():
                    outputs = self.model(**inputs)
                    loss = self.task.loss_eval(outputs, {"labels": labels})

                bs = int(labels.shape[0]) if isinstance(labels, torch.Tensor) and labels.dim() >= 1 else 1
                loss_sum += loss.detach().float() * bs
                total_n += bs

                if compute_metrics:
                    try:
                        self.task.update_metrics_eval(outputs, {"labels": labels})
                    except Exception:
                        pass

                if log_every_eval is not None and (i % log_every_eval == 0):
                    dt = _now() - t_last
                    steps_since = max(1, i - last_i)

                    denom = max(1, total_n - last_n)
                    window_loss_t = (loss_sum - last_loss_sum) / float(denom)
                    self._eval_loss_meter.push(window_loss_t)
                    window_loss = self._eval_loss_meter.poll()

                    left = (eval_total_iters - i) if (eval_total_iters is not None) else None
                    eta_str = self._fmt_eta((left * (dt / steps_since)) if left is not None else None)

                    wl_str = f"{window_loss:.4f}" if isinstance(window_loss, (int, float)) else "..."
                    print(
                        f" {prefix} step {i}/{eval_total_iters if eval_total_iters is not None else '?'}: "
                        f"avg_time/batch={dt/steps_since:.3f}s, avg_loss={wl_str}"
                        f"{', left='+str(left) if left is not None else ''}{eta_str}"
                    )

                    t_last = _now()
                    last_i = i
                    last_loss_sum.copy_(loss_sum.detach())
                    last_n = total_n

                if limit_batches is not None and i >= int(limit_batches):
                    break

            val_loss_t = loss_sum / float(max(1, total_n))

            # ---- no-sync eval mode ----
            if bool(no_gpu_sync):
                if val_loss_t.device.type == "cpu":
                    try:
                        v = float(val_loss_t.detach().float().item())
                    except Exception:
                        v = float("nan")
                    if was_training:
                        self.model.train()
                    return v, {"loss": v}, v

                try:
                    self._eval_loss_meter.push(val_loss_t.detach().float())
                    v = self._eval_loss_meter.poll()
                    v = float(v) if v is not None else float("nan")
                except Exception:
                    v = float("nan")

                if was_training:
                    self.model.train()
                return v, {"loss": v}, v

            # ---- normal eval: flush async metrics + ONE coalesced sync for scalars ----
            if compute_metrics:
                try:
                    if hasattr(self.task, "sync_metrics"):
                        self.task.sync_metrics()
                except Exception:
                    pass

            metrics_py: Dict[str, float] = {}
            metrics_cuda: Dict[str, torch.Tensor] = {"loss": val_loss_t}

            only_primary = bool(self.fast_mode and getattr(self.cfg, "fast_mode_primary_metric_only", False))

            best_cfg = str(getattr(self.cfg, "best_metric", "loss")).lower().strip()
            want_acc = bool(getattr(self.cfg, "always_compute_acc", True)) or (best_cfg in {"acc", "accuracy"})

            best_key = str(getattr(self, "best_selection_name", "loss")).lower().strip()
            primary_key = str(getattr(self, "primary_metric", "loss")).lower().strip()
            keep_names_lower = {primary_key, best_key, "loss"}
            if want_acc:
                keep_names_lower.add("acc")

            for name, m in (metrics_obj or {}).items():
                if only_primary:
                    try:
                        if str(name).lower() not in keep_names_lower:
                            continue
                    except Exception:
                        continue
                try:
                    v = m.compute()
                except Exception:
                    continue

                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics_cuda[str(name)] = v.detach()
                else:
                    try:
                        metrics_py[str(name)] = float(v)
                    except Exception:
                        pass

            metrics = dict(metrics_py)
            metrics.update(self._scalar_sync.read(metrics_cuda))

            if "loss" not in metrics:
                try:
                    metrics["loss"] = float(val_loss_t.detach().float().item())
                except Exception:
                    metrics["loss"] = float("nan")

            ml = self._lower_metrics(metrics)
            score = float(ml.get(primary_key, metrics["loss"]))
            val_loss = float(metrics["loss"])

            if was_training:
                self.model.train()
            return val_loss, metrics, score
        finally:
            self._flops_stage = prev_stage

    # ------------------------------
    # checkpoint helpers (model-only)
    # ------------------------------
    def _save_best(self) -> None:
        fname = f"{self.cfg.save_name_prefix}-best.pt"
        fpath = os.path.join(self.run_dir, fname)
        torch.save(self.model.state_dict(), fpath)
        self.best_path = fpath

    def _load_best(self) -> None:
        if self.best_path and os.path.isfile(self.best_path):
            try:
                state = torch.load(self.best_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(self.best_path, map_location=self.device)
            self.model.load_state_dict(state)

    # ------------------------------
    # optimizer step
    # ------------------------------
    def _maybe_get_grad_norm_tensor(self, params) -> Optional[torch.Tensor]:
        try:
            maxn = float(self.cfg.max_grad_norm) if self.cfg.max_grad_norm is not None else 0.0
        except Exception:
            maxn = 0.0

        if maxn and maxn > 0:
            try:
                return torch.nn.utils.clip_grad_norm_(params, maxn).detach()
            except Exception:
                return None

        # In fast_mode, skip grad norm unless clipping is enabled.
        if self.fast_mode:
            return None
        try:
            return _compute_grad_norm_tensor(params).detach()
        except Exception:
            return None

    def _optimizer_step(self) -> None:
        params = self._trainable_params_list

        if self.scaler.is_enabled():
            # 1D: Unscale whenever we might read/clip grads, otherwise norms can be on scaled grads.
            need_unscale = False
            try:
                if self.cfg.max_grad_norm is not None and float(self.cfg.max_grad_norm) > 0:
                    need_unscale = True
                elif (not self.fast_mode):
                    need_unscale = True
            except Exception:
                pass
            if need_unscale:
                try:
                    self.scaler.unscale_(self.opt)
                except Exception:
                    pass

            gn_t = self._maybe_get_grad_norm_tensor(params)
            if gn_t is not None:
                self._last_grad_norm_t = gn_t
                self._grad_norm_meter.push(gn_t)
                gn = self._grad_norm_meter.poll()
                if gn is not None:
                    self._last_grad_norm = float(gn)

            if not self.fast_mode:
                try:
                    self._last_amp_scale = float(self.scaler.get_scale())
                except Exception:
                    self._last_amp_scale = None
            else:
                self._last_amp_scale = None

            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            gn_t = self._maybe_get_grad_norm_tensor(params)
            if gn_t is not None:
                self._last_grad_norm_t = gn_t
                self._grad_norm_meter.push(gn_t)
                gn = self._grad_norm_meter.poll()
                if gn is not None:
                    self._last_grad_norm = float(gn)

            self.opt.step()

        if self._scheduler is not None:
            try:
                self._scheduler.step()
            except Exception:
                pass

        if self.cfg.min_lr is not None:
            floor = float(self.cfg.min_lr)
            for pg in self.opt.param_groups:
                if pg["lr"] < floor:
                    pg["lr"] = floor

        try:
            self.opt.zero_grad(set_to_none=True)
        except Exception:
            pass

        self._opt_step += 1
        self._notify_after_opt_step()

        for cb in self.callbacks:
            try:
                cb.on_optimizer_step_end(self, self.global_step, self._opt_step)
            except Exception:
                pass

    # ------------------------------
    # grad accumulation planner
    # ------------------------------
    @staticmethod
    def _epoch_total_iters(train_total_iters: Optional[int], limit_train_batches: Optional[int]) -> Optional[int]:
        if train_total_iters is None:
            return int(limit_train_batches) if limit_train_batches is not None else None
        if limit_train_batches is None:
            return int(train_total_iters)
        return min(int(train_total_iters), int(limit_train_batches))

    @staticmethod
    def _next_cycle_target(
        *,
        gas: int,
        epoch_total_iters: Optional[int],
        it: int,
        accum_in_cycle: int,
    ) -> int:
        gas = max(1, int(gas))
        if accum_in_cycle != 0:
            return gas
        if epoch_total_iters is None:
            return gas
        remaining = int(epoch_total_iters) - (int(it) - 1)
        return max(1, min(gas, remaining))

    # ------------------------------
    # fit
    # ------------------------------
    def fit(self, tag: str):
        if self.cfg.save_checkpoints and self.cfg.save_init_checkpoint:
            epochs_root = os.path.join(self.run_dir, self.cfg.epochs_dir_name)
            os.makedirs(epochs_root, exist_ok=True)
            ep0_ckpt = os.path.join(epochs_root, "epoch_00.pt")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "epoch": 0,
                    "global_step": int(self.global_step),
                    "opt_steps": int(self._opt_step),
                    "primary_metric": self.primary_metric,
                },
                ep0_ckpt,
            )

        tinfo = (self.run_meta.get("loaders", {}) or {}).get("train", {}) if isinstance(self.run_meta, dict) else {}
        vinfo = (self.run_meta.get("loaders", {}) or {}).get("val", {}) if isinstance(self.run_meta, dict) else {}
        eff_bs = (self.run_meta.get("effective_batch_size", None) if isinstance(self.run_meta, dict) else None)

        print(
            f"[TRAIN] device={self.device} epochs={self.cfg.epochs} amp={self.scaler.is_enabled()} dtype={self._amp_dtype} "
            f"primary={self.primary_metric} (maximize={self.maximize}) fast_mode={self.fast_mode} "
            f"nonfinite_sync={self.cfg.nonfinite_sync} stop_on_nan={self.cfg.stop_on_nan}"
        )
        print(
            f"[META] train_samples={tinfo.get('num_samples')} train_batches={tinfo.get('num_batches')} "
            f"val_samples={vinfo.get('num_samples')} val_batches={vinfo.get('num_batches')} "
            f"batch_size={tinfo.get('batch_size')} grad_accum={self.cfg.grad_accum_steps} eff_batch={eff_bs} "
            f"num_workers={tinfo.get('num_workers')} pin_memory={tinfo.get('pin_memory')} persistent_workers={tinfo.get('persistent_workers')}"
        )
        tb = (self.run_meta.get("trainable_breakdown", {}) if isinstance(self.run_meta, dict) else {})
        byk = tb.get("by_kind_params", {}) if isinstance(tb, dict) else {}
        print(
            f"[PARAM] total={self.total_params} trainable={self.trainable_params} ({100.0*self.trainable_fraction:.3f}%) "
            f"lora={byk.get('lora')} paca={byk.get('paca')} head={byk.get('head')} other={byk.get('other')} "
            f"| trainable_module_paths={tb.get('num_trainable_module_paths')}"
        )
        print(
            f"[BEST] policy=cfg.best_metric='{self.cfg.best_metric}' -> select='{self.best_selection_name}' "
            f"(maximize={self.best_selection_maximize})"
        )

        self.model.train()
        self._stop_training = False

        for cb in self.callbacks:
            try:
                cb.on_fit_start(self)
            except Exception as e:
                print(f"[WARN] callback.on_fit_start failed: {e}")

        try:
            method_cfg = (
                dataclasses.asdict(self.method.cfg)
                if dataclasses.is_dataclass(getattr(self.method, "cfg", None))
                else getattr(self.method, "cfg", None)
            )
            meta = {
                "name": type(self.method).__name__,
                "class_path": f"{self.method.__class__.__module__}.{self.method.__class__.__name__}",
                "cfg": method_cfg,
            }
            with open(os.path.join(self.run_dir, "method_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        try:
            train_total_iters = len(self.train_loader)
        except TypeError:
            train_total_iters = None

        log_every_train = self._resolve_log_every(self.cfg.train_log_interval, train_total_iters)
        total_epochs = int(self.cfg.epochs)

        if self.fast_mode:
            k_raw = int(getattr(self.cfg, "fast_mode_eval_every", -1) or -1)
            if k_raw < 0:
                k = max(1, int(math.ceil(0.1 * total_epochs)))
                print(f"[VAL] fast_mode eval schedule: auto (~10%) -> every k={k} epochs (and always last epoch)")
            elif k_raw == 0:
                print("[VAL] fast_mode eval schedule: disabled (k=0) (may still eval on last epoch if needed)")
            else:
                print(f"[VAL] fast_mode eval schedule: every k={int(k_raw)} epochs (and always last epoch)")

        for epoch in range(1, total_epochs + 1):
            epoch_t0 = _now()
            self._nonfinite_reset()
            self._nonfinite_hit_this_epoch = False

            if self.data_module is not None:
                try:
                    self.data_module.on_epoch_start(epoch)
                except Exception as e:
                    print(f"[WARN] epoch hook failed: {e}")

            for cb in self.callbacks:
                try:
                    cb.on_epoch_start(self, epoch)
                except Exception:
                    pass

            # ------------------ TRAIN ------------------
            train_t0 = _now()

            epoch_tokens = 0
            epoch_opt0 = int(self._opt_step)

            running_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            interval_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            seen = 0
            interval_seen = 0

            t_last = _now()
            last_it = 0

            try:
                self.opt.zero_grad(set_to_none=True)
            except Exception:
                pass

            gas = max(1, int(self.cfg.grad_accum_steps))
            epoch_total_iters_eff = self._epoch_total_iters(train_total_iters, self.cfg.limit_train_batches)

            accum_in_cycle = 0
            cycle_target = max(1, gas)

            aborted_by_nonfinite = False
            nonfinite_exception_msg = None

            try:
                for it, batch in enumerate(self.train_loader, 1):
                    for cb in self.callbacks:
                        try:
                            cb.on_train_batch_start(self, it, self.global_step)
                        except Exception:
                            pass

                    if accum_in_cycle == 0:
                        cycle_target = self._next_cycle_target(
                            gas=gas, epoch_total_iters=epoch_total_iters_eff, it=it, accum_in_cycle=accum_in_cycle
                        )

                    loss = self._step_batch(batch, accum_div=cycle_target)

                    bs = int(batch["labels"].shape[0]) if isinstance(batch.get("labels", None), torch.Tensor) else 1
                    loss_f = loss.detach().float()
                    running_loss_sum += loss_f * bs
                    interval_loss_sum += loss_f * bs
                    seen += bs
                    interval_seen += bs

                    self.global_step += 1
                    accum_in_cycle += 1

                    if accum_in_cycle >= cycle_target:
                        self._optimizer_step()
                        accum_in_cycle = 0

                    for cb in self.callbacks:
                        try:
                            cb.on_train_batch_end(self, it, self.global_step, loss.detach())
                        except Exception:
                            pass

                    # ---- progress logging (safe in fast_mode) ----
                    want_print = False
                    if self.fast_mode and self.cfg.fast_mode_print_every:
                        want_print = (it % max(1, int(self.cfg.fast_mode_print_every))) == 0
                    elif log_every_train is not None:
                        want_print = (it % log_every_train) == 0

                    if want_print:
                        dt = _now() - t_last
                        steps_since = max(1, it - last_it)
                        avg_time = dt / steps_since
                        throughput = interval_seen / max(1e-6, dt)

                        avg_loss_t = interval_loss_sum / float(max(1, interval_seen))
                        self._train_loss_meter.push(avg_loss_t)
                        avg_loss = self._train_loss_meter.poll()

                        left = (train_total_iters - it) if (train_total_iters is not None) else None
                        eta_str = self._fmt_eta(left * avg_time) if left is not None else ""
                        hdr = f"[TRAIN] epoch {epoch}/{total_epochs} it {it}/{train_total_iters if train_total_iters is not None else '?'}"
                        msg = [
                            f"avg_time/batch={avg_time:.3f}s",
                            f"avg_loss={(f'{avg_loss:.4f}' if isinstance(avg_loss, (int, float)) else '...')}",
                            f"throughput={throughput:.1f} samples/s",
                        ]
                        if left is not None:
                            msg.append(f"left={left}")
                        print(f" {hdr}: " + ", ".join(msg) + eta_str)

                        interval_loss_sum.zero_()
                        interval_seen = 0
                        t_last = _now()
                        last_it = it

                    if self.cfg.limit_train_batches is not None and it >= int(self.cfg.limit_train_batches):
                        break
            except NonFiniteError as e:
                aborted_by_nonfinite = True
                nonfinite_exception_msg = str(e)
                self._nonfinite_hit_this_epoch = True
                # Discard any partial grads to avoid stepping contaminated state.
                try:
                    self.opt.zero_grad(set_to_none=True)
                except Exception:
                    pass
                accum_in_cycle = 0

            # Flush remainder: fix scaling if epoch ends mid-cycle unexpectedly.
            if (not aborted_by_nonfinite) and (accum_in_cycle > 0):
                if accum_in_cycle != cycle_target:
                    # grads currently represent (1/cycle_target)*sum grad_i; convert to (1/accum_in_cycle)*sum grad_i
                    self._mul_grads_(float(cycle_target) / float(accum_in_cycle))
                self._optimizer_step()
                accum_in_cycle = 0

            train_wall = float(_now() - train_t0)
            train_epoch_loss_t = running_loss_sum / float(max(1, seen))

            boundary_scalars: Dict[str, torch.Tensor] = {
                "train_epoch_loss": train_epoch_loss_t,
            }
            if self._last_grad_norm_t is not None:
                boundary_scalars["grad_norm"] = self._last_grad_norm_t

            nonfinite_hit = False
            mode = str(getattr(self.cfg, "nonfinite_sync", "epoch")).lower().strip()
            if aborted_by_nonfinite:
                nonfinite_hit = True
            elif self._nonfinite_enabled() and mode == "epoch":
                nonfinite_hit = self._nonfinite_sync_and_reset()
            elif self._nonfinite_enabled() and mode == "batch":
                nonfinite_hit = bool(self._nonfinite_hit_this_epoch)

            if nonfinite_hit:
                self._ever_nonfinite = True
                self._nonfinite_epochs.append(int(epoch))
                if nonfinite_exception_msg:
                    print(f"[NONFINITE] {nonfinite_exception_msg}")
                else:
                    print(f"[NONFINITE] detected during training epoch {epoch}.")

            synced = self._scalar_sync.read(boundary_scalars)
            self.last_train_epoch_loss = float(synced.get("train_epoch_loss", float("nan")))
            if "grad_norm" in synced:
                self._last_grad_norm = float(synced["grad_norm"])

            self._last_epoch_train_sec = float(train_wall)
            self._last_epoch_train_seen = int(seen)
            self._last_epoch_train_tokens = int(epoch_tokens)
            self._last_epoch_train_opt_steps = int(self._opt_step - epoch_opt0)

            self.epoch_train_seen.append(int(seen))
            self.epoch_train_tokens.append(int(epoch_tokens))
            self.epoch_train_opt_steps.append(int(self._opt_step - epoch_opt0))
            self.epoch_train_wall_sec.append(float(train_wall))
            self.epoch_train_samples_per_s.append(float(seen) / max(1e-6, float(train_wall)))
            self.epoch_train_tokens_per_s.append(float(epoch_tokens) / max(1e-6, float(train_wall)) if epoch_tokens else 0.0)

            for cb in self.callbacks:
                try:
                    cb.on_train_epoch_end(self, epoch)
                except Exception:
                    pass

            if self.last_train_epoch_loss is not None and math.isfinite(self.last_train_epoch_loss):
                print(f"[TRAIN] epoch={epoch} avg_loss={self.last_train_epoch_loss:.4f} train_wall={train_wall:.2f}s")
            else:
                print(f"[TRAIN] epoch={epoch} train_wall={train_wall:.2f}s")

            # ------------------ EVAL ------------------
            eval_t0 = _now()

            skip_eval = bool(nonfinite_hit) and bool(getattr(self.cfg, "nonfinite_skip_eval", True))
            if skip_eval:
                val_loss = float("nan")
                metrics = {"loss": float("nan")}
                score = float("nan")
                eval_wall = float(_now() - eval_t0)
                self._last_eval_sec = float(eval_wall)
                self._last_val_loss = float(val_loss)
                print(f"[VAL] epoch={epoch} SKIPPED (non-finite detected in train); eval_wall={eval_wall:.2f}s")
            else:
                do_eval = True
                skip_reason = None

                if self.fast_mode:
                    k_raw = int(getattr(self.cfg, "fast_mode_eval_every", -1) or -1)
                    k = max(1, int(math.ceil(0.1 * total_epochs))) if k_raw < 0 else max(0, int(k_raw))

                    if k == 0:
                        need_last = (
                            (epoch == total_epochs)
                            and bool(getattr(self.cfg, "fast_mode_eval_on_last_epoch_for_best", True))
                            and (
                                bool(getattr(self.cfg, "save_checkpoints", True))
                                or (self.test_loader is not None)
                                or bool(getattr(self.cfg, "save_best_to_disk", True))
                            )
                        )
                        do_eval = bool(need_last)
                        if not do_eval:
                            skip_reason = "fast_mode_eval_every=0"
                    else:
                        if (epoch % k != 0) and (epoch != total_epochs):
                            do_eval = False
                            skip_reason = f"fast_mode_schedule(k={k})"

                if not do_eval:
                    score = float("nan")
                    val_loss = float("nan")
                    metrics = {"loss": float("nan")}
                    eval_wall = float(_now() - eval_t0)
                    self._last_eval_sec = float(eval_wall)
                    self._last_val_loss = float("nan")
                    print(f"[VAL] epoch={epoch} SKIPPED ({skip_reason}); eval_wall={eval_wall:.2f}s")
                else:
                    best_key = str(getattr(self, "best_selection_name", "loss")).lower().strip()
                    primary_key = str(getattr(self, "primary_metric", "loss")).lower().strip()

                    best_cfg = str(getattr(self.cfg, "best_metric", "loss")).lower().strip()
                    want_acc = bool(getattr(self.cfg, "always_compute_acc", True)) or (best_cfg in {"acc", "accuracy"})

                    # 1A: ensure we compute metrics whenever selection/primary requires it
                    need_metrics = (best_key != "loss") or (primary_key != "loss") or want_acc

                    compute_metrics = True
                    if self.fast_mode and bool(getattr(self.cfg, "fast_mode_eval_loss_only", True)):
                        compute_metrics = bool(need_metrics)

                    requested_no_sync = bool(self.fast_mode and bool(getattr(self.cfg, "fast_mode_no_gpu_sync", False)))
                    no_gpu_sync = requested_no_sync

                    # If we need a meaningful selection metric, force a synced eval.
                    if (
                        self.fast_mode
                        and requested_no_sync
                        and bool(getattr(self.cfg, "fast_mode_force_sync_for_best", True))
                        and (
                            bool(getattr(self.cfg, "save_checkpoints", True))
                            or bool(getattr(self.cfg, "save_best_to_disk", True))
                            or (self.test_loader is not None)
                        )
                    ):
                        no_gpu_sync = False

                    val_loss, metrics, score = self._evaluate(
                        self.val_loader,
                        limit_batches=self.cfg.limit_eval_batches,
                        stage="val",
                        compute_metrics=compute_metrics,
                        no_gpu_sync=no_gpu_sync,
                    )

                    eval_wall = float(_now() - eval_t0)
                    self._last_eval_sec = float(eval_wall)
                    self._last_val_loss = float(val_loss)

                    try:
                        metrics_print = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (metrics or {}).items()}
                    except Exception:
                        metrics_print = metrics

                    if isinstance(val_loss, (int, float)) and math.isfinite(float(val_loss)):
                        print(f"[VAL] epoch={epoch} val_loss={float(val_loss):.4f} eval_wall={eval_wall:.2f}s metrics={metrics_print}")
                    else:
                        print(f"[VAL] epoch={epoch} val_loss=... eval_wall={eval_wall:.2f}s metrics={metrics_print}")

            # ------------------ BEST / SAVE ------------------
            save_t0 = _now()
            skip_save = bool(nonfinite_hit) and bool(getattr(self.cfg, "nonfinite_skip_save", True))

            sel_val = self._extract_selection_value(val_loss=float(val_loss), metrics=(metrics or {}), score=float(score))
            is_better = self._is_better(sel_val)
            is_best_for_epoch = bool(is_better)

            if is_better:
                self.best_selection_value = float(sel_val)  # sel_val is not None here
                self.best_score = float(self.best_selection_value)
                self.best_epoch = int(epoch)

                save_best_to_disk = bool(getattr(self.cfg, "save_best_to_disk", True))
                should_save_best = (bool(self.cfg.save_checkpoints) or save_best_to_disk) and (not skip_save)

                if should_save_best:
                    self._save_best()

                self.best_val_loss = float(val_loss) if (isinstance(val_loss, (int, float)) and math.isfinite(float(val_loss))) else None
                try:
                    self.best_val_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (metrics or {}).items()}
                except Exception:
                    self.best_val_metrics = metrics or {}

                if should_save_best:
                    print(f" -> new best by '{self.best_selection_name}'={self.best_selection_value:.6g}; saved: {self.best_path}")
                else:
                    print(f" -> new best by '{self.best_selection_name}'={self.best_selection_value:.6g}; (saving skipped/disabled)")

            want_epoch_save = bool(self.cfg.save_checkpoints and self.cfg.save_all_epochs and (not skip_save))
            if want_epoch_save and self.cfg.save_epoch_every:
                try:
                    k = int(self.cfg.save_epoch_every)
                    if k > 1 and (epoch % k != 0) and (epoch != total_epochs):
                        want_epoch_save = False
                except Exception:
                    pass

            if want_epoch_save:
                epochs_root = os.path.join(self.run_dir, self.cfg.epochs_dir_name)
                os.makedirs(epochs_root, exist_ok=True)
                ep_ckpt = os.path.join(epochs_root, f"epoch_{epoch:02d}.pt")

                payload = {
                    "model": self.model.state_dict(),
                    "epoch": int(epoch),
                    "global_step": int(self.global_step),
                    "opt_steps": int(self._opt_step),
                    "primary_metric": self.primary_metric,
                    "primary_score": (float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None),
                    "best_selection_name": str(self.best_selection_name),
                    "best_selection_value": (float(sel_val) if sel_val is not None else None),
                    "is_best": bool(is_best_for_epoch),
                }
                torch.save(payload, ep_ckpt)

                ep_json = {
                    "epoch": int(epoch),
                    "val_loss": float(val_loss) if isinstance(val_loss, (int, float)) else None,
                    "val_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (metrics or {}).items()},
                    "primary_metric": self.primary_metric,
                    "primary_score": (float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None),
                    "best_selection_name": str(self.best_selection_name),
                    "best_selection_value": (float(sel_val) if sel_val is not None else None),
                    "is_best": bool(is_best_for_epoch),
                    "checkpoint_path": ep_ckpt,
                }
                with open(os.path.join(epochs_root, f"epoch_{epoch:02d}.json"), "w") as _f:
                    json.dump(ep_json, _f, indent=2)
            elif skip_save:
                print(f"[SAVE] epoch={epoch} SKIPPED (non-finite detected in train).")

            self._last_save_sec = float(_now() - save_t0)

            for cb in self.callbacks:
                try:
                    cb.on_validation_end(self, epoch, metrics, float(score))
                except Exception:
                    pass
            for cb in self.callbacks:
                try:
                    cb.on_epoch_end(self, epoch)
                except Exception:
                    pass

            epoch_wall = float(_now() - epoch_t0)
            self._last_epoch_sec = float(epoch_wall)
            self.epoch_durations.append(float(epoch_wall))

            if nonfinite_hit and bool(getattr(self.cfg, "stop_on_nan", True)):
                self._stop_training = True
                print(f"[NONFINITE] aborting after epoch {epoch} (stop_on_nan=True).")

            if self.fast_mode and bool(getattr(self.cfg, "fast_mode_disable_early_stopping", True)):
                cb_stop = False
            else:
                cb_stop = any(cb.should_stop() for cb in self.callbacks)

            self._stop_training = bool(self._stop_training) or bool(cb_stop)

            if self._stop_training:
                print(
                    f"[EARLY-STOP] Stopping after epoch {epoch}. "
                    f"Best(by {self.best_selection_name})={self.best_selection_value:.6g} "
                    f"| best_ckpt={self.best_path}"
                )
                break

        print(f"[DONE] best(by {self.best_selection_name})={self.best_selection_value:.6g} | ckpt={self.best_path}")

        if self.fast_mode and bool(getattr(self.cfg, "fast_mode_final_sync_eval", False)):
            try:
                print("[VAL] final synced eval (fast_mode_final_sync_eval=True)")
                vloss, vmetrics, vscore = self._evaluate(
                    self.val_loader,
                    limit_batches=self.cfg.limit_eval_batches,
                    stage="val",
                    compute_metrics=True,
                    no_gpu_sync=False,
                )
                print(f"[VAL] final val_loss={vloss:.4f} score={vscore:.4f} metrics={vmetrics}")
            except Exception as e:
                print(f"[WARN] final synced eval failed: {e}")

        test_loss = None
        test_metrics = None
        if self.test_loader is not None:
            if self._ever_nonfinite and bool(getattr(self.cfg, "nonfinite_skip_test", True)):
                print("[TEST] skipped (non-finite detected during training).")
            elif (self.best_path is None) or (not os.path.isfile(self.best_path)):
                print("[TEST] skipped (no best checkpoint/state available).")
            else:
                self._load_best()
                test_loss, test_metrics, _ = self._evaluate(
                    self.test_loader,
                    limit_batches=self.cfg.limit_eval_batches,
                    stage="test",
                    compute_metrics=True,
                    no_gpu_sync=False,
                )
                self._test_loss = float(test_loss)
                self._test_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (test_metrics or {}).items()}
                try:
                    test_print = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (test_metrics or {}).items()}
                except Exception:
                    test_print = test_metrics
                print(f"[TEST] loss={test_loss:.4f} metrics={test_print}")

        gpu_info = {}
        if torch.cuda.is_available() and self._cuda_enabled:
            try:
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                    "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                }
            except Exception:
                pass

        export: Dict[str, Any] = {
            "tag": tag,
            "epochs": int(self.cfg.epochs),
            "epoch_durations_sec": [float(x) for x in self.epoch_durations],
            "epoch_train": {
                "seen": [int(x) for x in self.epoch_train_seen],
                "tokens": [int(x) for x in self.epoch_train_tokens],
                "opt_steps": [int(x) for x in self.epoch_train_opt_steps],
                "wall_sec": [float(x) for x in self.epoch_train_wall_sec],
                "samples_per_s": [float(x) for x in self.epoch_train_samples_per_s],
                "tokens_per_s": [float(x) for x in self.epoch_train_tokens_per_s],
            },
            "params": {
                "total": int(self.total_params),
                "trainable": int(self.trainable_params),
                "trainable_fraction": float(self.trainable_fraction),
                "reduction_factor_vs_full": float(self.reduction_factor_vs_full) if self.reduction_factor_vs_full else None,
            },
            "primary_metric": self.primary_metric,
            "maximize": bool(self.maximize),
            "best_selection": {
                "best_metric_cfg": str(getattr(self.cfg, "best_metric", "loss")),
                "resolved_name": str(self.best_selection_name),
                "maximize": bool(self.best_selection_maximize),
                "value": float(self.best_selection_value) if math.isfinite(float(self.best_selection_value)) else None,
            },
            "best": {
                "metric_name": str(self.best_selection_name),
                "score": float(self.best_selection_value) if math.isfinite(float(self.best_selection_value)) else None,
                "checkpoint": self.best_path,
                "epoch": int(self.best_epoch) if self.best_epoch is not None else None,
                "val_loss": float(self.best_val_loss) if self.best_val_loss is not None else None,
                "val_metrics": self.best_val_metrics or {},
                "primary_metric": self.primary_metric,
                "primary_score": (self.best_val_metrics or {}).get(self.primary_metric),
            },
            "test": {
                "loss": float(test_loss) if test_loss is not None else None,
                "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (test_metrics or {}).items()},
                "score": (float((test_metrics or {}).get(self.primary_metric)) if (test_metrics and self.primary_metric in test_metrics) else None),
            },
            "nonfinite": {
                "detected": bool(self._ever_nonfinite),
                "epochs": [int(e) for e in self._nonfinite_epochs],
                "policy": {
                    "detect_nonfinite": bool(getattr(self.cfg, "detect_nonfinite", True)),
                    "nonfinite_sync": str(getattr(self.cfg, "nonfinite_sync", "epoch")),
                    "stop_on_nan": bool(getattr(self.cfg, "stop_on_nan", True)),
                    "skip_eval": bool(getattr(self.cfg, "nonfinite_skip_eval", True)),
                    "skip_save": bool(getattr(self.cfg, "nonfinite_skip_save", True)),
                    "skip_test": bool(getattr(self.cfg, "nonfinite_skip_test", True)),
                },
            },
            "device": str(self.device),
            "method": type(self.method).__name__,
            "git_commit": self.git_commit,
            "stopped_early": bool(self._stop_training),
            "global_step": int(self.global_step),
            "optimizer_steps": int(self._opt_step),
            "save_dir": self.cfg.save_dir,
            "save_name_prefix": self.cfg.save_name_prefix,
            "run_dir": self.run_dir,
            "per_epoch_saves": {
                "enabled": bool(self.cfg.save_all_epochs),
                "folder": self.cfg.epochs_dir_name,
                "every": int(self.cfg.save_epoch_every) if self.cfg.save_epoch_every else None,
            },
            "run_meta": self.run_meta,
            "env": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": getattr(__import__("transformers"), "__version__", None),
            },
            "gpu": gpu_info,
            "scheduler": self._sched_meta,
        }

        export["train_config"] = dataclasses.asdict(self.cfg) if dataclasses.is_dataclass(self.cfg) else {}

        for cb in self.callbacks:
            try:
                if cb.__class__.__name__ != "WandbCallback":
                    cb.update_summary(export)
            except Exception:
                pass
        for cb in self.callbacks:
            try:
                if cb.__class__.__name__ == "WandbCallback":
                    cb.update_summary(export)
            except Exception:
                pass

        json_path = os.path.join(self.run_dir, f"{self.cfg.save_name_prefix}-metrics.json")
        with open(json_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"[WRITE] metrics -> {json_path}")

        for cb in self.callbacks:
            try:
                cb.on_fit_end(self)
            except Exception:
                pass

        return self.best_path, self.best_score