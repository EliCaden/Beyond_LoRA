# trainers/callbacks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import os

import torch
import torch.nn as nn


# ---- Optional adapter imports (used only for instance checks / stats) ----
# Keep these defensive imports: some downstream callbacks/scripts may use them.
try:
    from models.lora_layers import LoRALinearAdapter
except Exception:
    LoRALinearAdapter = None

try:
    from models.paca_layers import PaCALinearAdapter
except Exception:
    PaCALinearAdapter = None


# ------------------------------
# Sync-safe async scalar meter
# ------------------------------

class _AsyncScalarD2H:
    """
    Deferred GPU->CPU scalar meter (best-effort, sync-minimized).

    - push(x): enqueue a non_blocking D2H copy of a 0-dim CUDA tensor into pinned CPU memory
              and record a CUDA event.
    - poll(): if the event has completed, return the latest copied value; otherwise return
              the last completed value (or None if none have completed yet).

    Notes:
      - This avoids *forcing* a GPU sync from the logging site.
      - If pushes happen faster than copies complete, poll() can lag. We track only the
        most recent pending slot (cheap + sufficient for progress logging).
    """
    def __init__(self, *, enabled: bool, nbuf: int = 4, dtype: torch.dtype = torch.float32):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.dtype = dtype
        self._latest: Optional[float] = None

        self._nbuf = max(2, int(nbuf))
        self._buf: List[torch.Tensor] = []
        self._ev: List[torch.cuda.Event] = []
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

        # CPU path (no async needed)
        if (not self.enabled) or x.device.type != "cuda":
            if x.device.type == "cpu":
                try:
                    self._latest = float(x.detach().to(dtype=self.dtype).item())
                except Exception:
                    self._latest = None
            else:
                # non-cuda accelerator: best-effort sync read
                try:
                    self._latest = float(x.detach().to("cpu", dtype=self.dtype).item())
                except Exception:
                    self._latest = None
            return

        i = self._idx
        self._idx = (self._idx + 1) % self._nbuf

        try:
            # cast on GPU if needed, then async copy to pinned CPU scalar
            self._buf[i].copy_(x.detach().to(dtype=self.dtype), non_blocking=True)
            self._ev[i].record(torch.cuda.current_stream())
            self._pending_idx = i
            self._has_pending = True
        except Exception:
            pass

    def poll(self) -> Optional[float]:
        if not self.enabled:
            return self._latest

        if self._has_pending and self._pending_idx is not None:
            i = self._pending_idx
            try:
                if self._ev[i].query():
                    self._latest = float(self._buf[i].item())
                    self._has_pending = False
                    self._pending_idx = None
            except Exception:
                pass
        return self._latest


# ------------------------------
# Callback base
# ------------------------------

class Callback:
    """
    Minimal callback surface used by GenericTrainer.

    Sync-safety guidelines:
      - Do NOT call .item() / .cpu() / .numpy() on CUDA tensors inside per-step hooks.
        That forces a GPU->CPU sync and can dominate runtime.
      - For per-step logging, use _AsyncScalarD2H (best-effort) or log only CPU scalars.
      - Prefer epoch/validation hooks: GenericTrainer already coalesces/synchronizes scalar
        metrics at epoch/eval boundaries and passes Python floats in `metrics`.
      - In fast mode, GenericTrainer may reduce evaluation frequency and/or reduce the
        metric set; callbacks should tolerate missing keys gracefully.
    """
    def on_fit_start(self, trainer: "GenericTrainer"): ...
    def on_fit_end(self, trainer: "GenericTrainer"): ...

    def on_epoch_start(self, trainer: "GenericTrainer", epoch: int): ...
    def on_epoch_end(self, trainer: "GenericTrainer", epoch: int): ...

    def on_train_batch_start(self, trainer: "GenericTrainer", batch_idx: int, global_step: int): ...
    def on_train_batch_end(self, trainer: "GenericTrainer", batch_idx: int, global_step: int, loss: torch.Tensor): ...

    def on_train_epoch_end(self, trainer: "GenericTrainer", epoch: int): ...
    def on_optimizer_step_end(self, trainer: "GenericTrainer", global_step: int, opt_step: int): ...

    def on_validation_end(self, trainer: "GenericTrainer", epoch: int, metrics: Dict[str, Any], score: float): ...

    def should_stop(self) -> bool:
        return False

    def update_summary(self, export: Dict[str, Any]) -> None:
        # Called at end of fit; safe place to write extra fields to export JSON/W&B summary.
        return


# ------------------------------
# Early stopping
# ------------------------------

def _to_finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v == v):  # NaN
        return None
    if v == float("inf") or v == -float("inf"):
        return None
    return v


@dataclass
class EarlyStoppingCallback(Callback):
    """
    Early stopping based on validation results.

    This callback is "pure" (no GPU sync): it consumes the Python floats passed in
    `metrics`/`score` by the trainer at validation boundary.

    Monitor options (monitor=...):
      - "best"    (default): monitor the *same* metric the trainer uses to select best checkpoint
                             (trainer.best_selection_name / best_selection_maximize), if present.
      - "score"   : monitor the score passed by trainer (typically primary metric; fallback to loss)
      - "loss"    : monitor metrics["loss"] (minimize)
      - "acc"     : monitor metrics["acc"] (maximize)
      - "primary" : monitor trainer.primary_metric (direction per trainer.maximize); fallback to score
      - any other string: monitor metrics[that_key] (direction default: maximize=True)

    Note:
      - In fast mode, GenericTrainer may choose to ignore callback-driven early stopping entirely
        (via config). In that case, this callback may still set _stop=True, but the trainer will
        not honor it.
      - Missing metrics / NaN/Inf -> no update (and thus no early stop).
    """
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "best"

    def __post_init__(self):
        self._reset()

    def _reset(self):
        self._best: Optional[float] = None
        self._bad_epochs: int = 0
        self._stop: bool = False

    def on_fit_start(self, trainer: "GenericTrainer"):
        self._reset()

    def _resolve_value_and_direction(
        self,
        trainer: "GenericTrainer",
        metrics: Dict[str, Any],
        score: float,
    ) -> Tuple[Optional[float], bool]:
        mon = (self.monitor or "best").strip().lower()

        if mon == "best":
            key = str(getattr(trainer, "best_selection_name", "score")).strip().lower()
            maximize = bool(getattr(trainer, "best_selection_maximize", True))

            if isinstance(metrics, dict):
                if key in metrics:
                    v = _to_finite_float(metrics.get(key))
                    if v is not None:
                        return v, maximize
                if key in {"val_loss", "loss"}:
                    v = _to_finite_float(metrics.get("loss"))
                    if v is not None:
                        return v, False

            return _to_finite_float(score), maximize

        if mon == "score":
            return _to_finite_float(score), bool(getattr(trainer, "maximize", True))

        if mon in {"loss", "val_loss"}:
            v = _to_finite_float(metrics.get("loss") if isinstance(metrics, dict) else None)
            return v, False

        if mon in {"acc", "accuracy"}:
            v = _to_finite_float(metrics.get("acc") if isinstance(metrics, dict) else None)
            return v, True

        if mon == "primary":
            pm = str(getattr(trainer, "primary_metric", "score"))
            v = _to_finite_float(metrics.get(pm) if isinstance(metrics, dict) else None)
            if v is None:
                v = _to_finite_float(score)
            return v, bool(getattr(trainer, "maximize", True))

        # custom metric key
        v = _to_finite_float(metrics.get(mon) if isinstance(metrics, dict) else None)
        maximize = True
        try:
            if mon == str(getattr(trainer, "primary_metric", "")).strip().lower():
                maximize = bool(getattr(trainer, "maximize", True))
        except Exception:
            pass
        return v, maximize

    def on_validation_end(self, trainer: "GenericTrainer", epoch: int, metrics: Dict[str, Any], score: float):
        v, maximize = self._resolve_value_and_direction(trainer, metrics or {}, score)
        if v is None:
            return

        if self._best is None:
            self._best = float(v)
            self._bad_epochs = 0
            return

        if maximize:
            improved = float(v) > float(self._best) + float(self.min_delta)
        else:
            improved = float(v) < float(self._best) - float(self.min_delta)

        if improved:
            self._best = float(v)
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs >= int(self.patience):
                self._stop = True

    def should_stop(self) -> bool:
        return bool(self._stop)


# ------------------------------
# Chain reset (optimizer rebuild aware)
# ------------------------------

@dataclass
class ChainResetCallback(Callback):
    """
    Applies a method-specific chain reset recipe periodically and rebuilds the optimizer
    afterward if trainable parameters changed.

    Design goals:
      - Safe no-ops if the method doesn't implement chain reset entrypoints.
      - No checkpoint/resume expectations: optimizer rebuild is done in-memory.
      - Cache invalidation hooks are defensive (only if model exposes flags).
    """
    recipe: str
    every_n_opt_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    keep_group_lrs: bool = True
    disable_scheduler: bool = True

    def __post_init__(self):
        self._next_opt_trigger = int(self.every_n_opt_steps) if self.every_n_opt_steps else None

    def _invalidate_adapter_caches(self, trainer: "GenericTrainer") -> None:
        model = getattr(trainer, "model", None)
        if model is None:
            return
        # Some models in this repo use these flags; safe no-ops otherwise.
        for flag in ("_lora_cache_valid", "_paca_cache_valid"):
            if hasattr(model, flag):
                try:
                    setattr(model, flag, False)
                except Exception:
                    pass

    def _apply_chain_reset(self, trainer: "GenericTrainer"):
        m = getattr(trainer, "method", None)
        if m is None:
            return

        candidates = [
            "chain_reset",
            "reset_chain",
            "apply_chain_reset",
            "reset_adapters",
            "apply_recipe",
        ]
        for name in candidates:
            fn = getattr(m, name, None)
            if callable(fn):
                try:
                    fn(trainer.model, recipe=self.recipe)
                except TypeError:
                    try:
                        fn(trainer.model, self.recipe)
                    except TypeError:
                        try:
                            fn(recipe=self.recipe)
                        except TypeError:
                            try:
                                fn(self.recipe)
                            except Exception:
                                pass
                except Exception:
                    pass
                break

        self._invalidate_adapter_caches(trainer)

        # Rebuild optimizer param groups (if method changed trainable params).
        try:
            trainer.rebuild_optimizer_after_chain(
                keep_group_lrs=bool(self.keep_group_lrs),
                disable_scheduler=bool(self.disable_scheduler),
            )
        except Exception:
            pass

    def on_optimizer_step_end(self, trainer: "GenericTrainer", global_step: int, opt_step: int):
        if self._next_opt_trigger is None:
            return
        if opt_step >= self._next_opt_trigger:
            self._apply_chain_reset(trainer)
            self._next_opt_trigger = opt_step + int(self.every_n_opt_steps)

    def on_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        if self.every_n_epochs is None:
            return
        k = int(self.every_n_epochs)
        if k > 0 and (epoch % k == 0):
            self._apply_chain_reset(trainer)


# ------------------------------
# FLOPs counter (op-level, cached by input shape)
# ------------------------------

try:
    # PyTorch 2.x
    from torch.utils.flop_counter import FlopCounterMode
except Exception:
    FlopCounterMode = None


class FlopsCounterCallback(Callback):
    """
    FLOPs counter:

    - Eval (val/test): counts forward FLOPs via forward hooks + FlopCounterMode, cached by input signature.
    - Train: counts forward+backward FLOPs by keeping FlopCounterMode active across the entire microbatch
      (forward -> loss -> backward). This requires GenericTrainer to call:
          on_flops_train_step_start(trainer, inputs)
          on_flops_train_step_after_forward(trainer)
          on_flops_train_step_end(trainer)
      The measurement is cached by input signature, so overhead is paid only once per unique shape/dtype.

    Notes:
      - This counts "useful FLOPs" known to PyTorch's flop formulas (matmul/conv/sdpa/etc).
      - It excludes optimizer/update FLOPs because the bracket ends immediately after backward.
      - If FlopCounterMode is unavailable, falls back to the old Linear/Conv2d forward-hook estimate
        and uses backward_factor as a crude multiplier (train_total_est).
    """

    def __init__(
        self,
        backward_factor: float = 2.0,
        *,
        count_eval: bool = True,
        depth: int = 2,
        cache_by_shape: bool = True,
    ):
        self.backward_factor = float(backward_factor)  # fallback only
        self.count_eval = bool(count_eval)
        self.depth = int(depth)
        self.cache_by_shape = bool(cache_by_shape)

        self._hooks = []
        self._trainer = None

        # ---- Eval forward cache (signature -> fwd_flops) ----
        self._cache: Dict[Any, int] = {}

        # ---- Train-step cache (signature -> (fwd_flops, bwd_flops)) ----
        self._train_cache: Dict[Any, Tuple[int, int]] = {}

        # ---- Totals (forward buckets) ----
        self._fwd_total = 0
        self._fwd_train = 0
        self._fwd_val = 0
        self._fwd_test = 0

        # ---- Train backward total ----
        self._bwd_train = 0

        # per-epoch bookkeeping
        self._epoch_start = (0, 0, 0, 0, 0)  # (fwd_total, fwd_train, fwd_val, fwd_test, bwd_train)
        self._per_epoch = {
            "fwd_train": [],
            "bwd_train": [],
            "train_total": [],
            "fwd_val": [],
            "fwd_test": [],
            "fwd_total": [],
        }

        # ---- Active eval-forward state (hook-based) ----
        self._active_sig = None
        self._active_stage = None
        self._active_mode = None
        self._active_measuring = False

        # ---- Active train-step state (trainer-bracketed) ----
        self._train_active_sig = None
        self._train_active_mode = None
        self._train_active_measuring = False
        self._train_active_fwd_flops = 0

        # fallback (old behavior) if FlopCounterMode is unavailable
        self._fallback_hooks = []
        self._fallback_by_module: Dict[str, int] = {}

    # --------------------------
    # Common helpers
    # --------------------------
    def _add_fwd(self, stage: str, flops: int) -> None:
        if flops <= 0:
            return
        self._fwd_total += int(flops)
        st = str(stage)
        if st == "train":
            self._fwd_train += int(flops)
        elif st == "val":
            self._fwd_val += int(flops)
        elif st == "test":
            self._fwd_test += int(flops)

    def _add_train_bwd(self, flops: int) -> None:
        if flops <= 0:
            return
        self._bwd_train += int(flops)

    def _stage(self, module: nn.Module) -> str:
        tr = self._trainer
        if tr is not None:
            s = getattr(tr, "_flops_stage", None)
            if isinstance(s, str) and s:
                # If we're in a train-step bracket measurement, ignore hook-based train counting
                if s == "train" and bool(getattr(tr, "_flops_in_train_step", False)):
                    return "ignore"
                return s

        if bool(module.training) and torch.is_grad_enabled():
            return "train"
        if (not bool(module.training)) and self.count_eval:
            return "val"
        return "ignore"

    def _sig_from_kwargs(self, kwargs: Dict[str, Any]) -> Any:
        items = []

        def rec(prefix: str, obj: Any):
            if isinstance(obj, torch.Tensor):
                items.append((prefix, tuple(obj.shape), str(obj.dtype)))
                return
            if isinstance(obj, dict):
                for k in sorted(obj.keys(), key=lambda x: str(x)):
                    rec(prefix + "." + str(k), obj[k])
                return
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    rec(prefix + f"[{i}]", v)
                return

        for k in sorted(kwargs.keys(), key=lambda x: str(x)):
            rec(str(k), kwargs[k])

        return tuple(items)

    # --------------------------
    # Fallback (Linear/Conv2d) forward-only counting
    # --------------------------
    def _fallback_add(self, key: str, v: int):
        if v <= 0:
            return
        self._fallback_by_module[key] = self._fallback_by_module.get(key, 0) + int(v)

    def _linear_flops(self, x: torch.Tensor, w: torch.Tensor) -> int:
        try:
            in_f = int(w.shape[1])
            out_f = int(w.shape[0])
            elems = int(x.numel() // max(1, in_f))
            return 2 * elems * in_f * out_f
        except Exception:
            return 0

    def _conv2d_flops(self, w: torch.Tensor, y: torch.Tensor, groups: int) -> int:
        try:
            out_c = int(w.shape[0])
            kH = int(w.shape[2])
            kW = int(w.shape[3])
            in_c_per_g = int(w.shape[1])
            bs = int(y.shape[0])
            out_h = int(y.shape[2])
            out_w = int(y.shape[3])
            per_out = 2 * in_c_per_g * kH * kW
            return bs * out_c * out_h * out_w * per_out
        except Exception:
            return 0

    def _hook_linear_fallback(self, module: nn.Module, inputs, output):
        if not module.training or not torch.is_grad_enabled():
            return
        try:
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            w = getattr(module, "weight", None)
            if not isinstance(w, torch.Tensor):
                return
            self._fallback_add("Linear", self._linear_flops(x, w))
        except Exception:
            pass

    def _hook_conv2d_fallback(self, module: nn.Module, inputs, output):
        if not module.training or not torch.is_grad_enabled():
            return
        try:
            y = output
            w = getattr(module, "weight", None)
            if not (isinstance(y, torch.Tensor) and isinstance(w, torch.Tensor)):
                return
            groups = int(getattr(module, "groups", 1))
            self._fallback_add("Conv2d", self._conv2d_flops(w, y, groups))
        except Exception:
            pass

    # --------------------------
    # Eval forward counting (hook-based, cached)
    # --------------------------
    def _pre_hook(self, module: nn.Module, args, kwargs):
        stage = self._stage(module)
        self._active_stage = stage
        self._active_sig = None
        self._active_mode = None
        self._active_measuring = False

        if stage == "ignore":
            return
        if stage == "train":
            # train forward is counted by train-step bracket when enabled
            return

        if FlopCounterMode is None:
            return

        if not self.cache_by_shape:
            self._active_sig = ("no_cache",)
            self._active_mode = FlopCounterMode(mods=module, depth=self.depth, display=False)
            self._active_mode.__enter__()
            self._active_measuring = True
            return

        sig = self._sig_from_kwargs(kwargs or {})
        self._active_sig = sig
        if sig in self._cache:
            return

        self._active_mode = FlopCounterMode(mods=module, depth=self.depth, display=False)
        self._active_mode.__enter__()
        self._active_measuring = True

    def _post_hook(self, module: nn.Module, args, kwargs, output):
        stage = self._active_stage or self._stage(module)
        if stage == "ignore":
            return
        if stage == "train":
            return

        if FlopCounterMode is None:
            return

        sig = self._active_sig if self.cache_by_shape else ("no_cache",)

        total = 0
        if self._active_measuring and (self._active_mode is not None):
            try:
                total = int(self._active_mode.get_total_flops())
            except Exception:
                total = 0
            try:
                self._active_mode.__exit__(None, None, None)
            except Exception:
                pass

            if self.cache_by_shape:
                self._cache[sig] = int(total)

        fl = int(self._cache.get(sig, 0)) if self.cache_by_shape else int(total)
        self._add_fwd(stage, fl)

        self._active_sig = None
        self._active_stage = None
        self._active_mode = None
        self._active_measuring = False

    # --------------------------
    # Train-step counting (trainer-bracketed, cached)
    # --------------------------
    def on_flops_train_step_start(self, trainer: "GenericTrainer", inputs: Dict[str, Any]) -> None:
        """
        Called BEFORE forward() in a training microbatch.
        Keeps FlopCounterMode active until on_flops_train_step_end() so backward is included.
        """
        if FlopCounterMode is None:
            return
        self._trainer = trainer
        try:
            setattr(trainer, "_flops_in_train_step", True)
        except Exception:
            pass

        sig = self._sig_from_kwargs(inputs or {})
        self._train_active_sig = sig
        self._train_active_fwd_flops = 0
        self._train_active_mode = None
        self._train_active_measuring = False

        if self.cache_by_shape and sig in self._train_cache:
            fwd, bwd = self._train_cache[sig]
            self._add_fwd("train", int(fwd))
            self._add_train_bwd(int(bwd))
            return

        # Measure once for this signature
        try:
            self._train_active_mode = FlopCounterMode(mods=trainer.model, depth=self.depth, display=False)
            self._train_active_mode.__enter__()
            self._train_active_measuring = True
        except Exception:
            self._train_active_mode = None
            self._train_active_measuring = False

    def on_flops_train_step_after_forward(self, trainer: "GenericTrainer") -> None:
        """
        Called AFTER forward+loss are computed, but BEFORE backward().
        We snapshot forward-only FLOPs so backward FLOPs can be computed by difference.
        """
        if not self._train_active_measuring or self._train_active_mode is None:
            return
        try:
            self._train_active_fwd_flops = int(self._train_active_mode.get_total_flops())
        except Exception:
            self._train_active_fwd_flops = 0

    def on_flops_train_step_end(self, trainer: "GenericTrainer") -> None:
        """
        Called AFTER backward() finishes (or in finally).
        Finalizes the measurement and caches (fwd,bwd) FLOPs for this signature.
        """
        try:
            setattr(trainer, "_flops_in_train_step", False)
        except Exception:
            pass

        if not self._train_active_measuring or self._train_active_mode is None:
            self._train_active_sig = None
            self._train_active_mode = None
            self._train_active_measuring = False
            self._train_active_fwd_flops = 0
            return

        sig = self._train_active_sig if self._train_active_sig is not None else ("no_sig",)

        try:
            tot = int(self._train_active_mode.get_total_flops())
        except Exception:
            tot = 0

        fwd = int(self._train_active_fwd_flops) if int(self._train_active_fwd_flops) > 0 else int(tot)
        bwd = max(0, int(tot) - int(fwd))

        try:
            self._train_active_mode.__exit__(None, None, None)
        except Exception:
            pass

        if self.cache_by_shape:
            self._train_cache[sig] = (int(fwd), int(bwd))

        self._add_fwd("train", int(fwd))
        self._add_train_bwd(int(bwd))

        self._train_active_sig = None
        self._train_active_mode = None
        self._train_active_measuring = False
        self._train_active_fwd_flops = 0

    # --------------------------
    # Callback lifecycle
    # --------------------------
    def on_fit_start(self, trainer: "GenericTrainer"):
        self._trainer = trainer

        self._cache = {}
        self._train_cache = {}

        self._fwd_total = self._fwd_train = self._fwd_val = self._fwd_test = 0
        self._bwd_train = 0

        self._per_epoch = {
            "fwd_train": [],
            "bwd_train": [],
            "train_total": [],
            "fwd_val": [],
            "fwd_test": [],
            "fwd_total": [],
        }

        # Hook ROOT module for eval forward counting (train is ignored when bracketed).
        if FlopCounterMode is not None:
            try:
                self._hooks.append(trainer.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
                self._hooks.append(trainer.model.register_forward_hook(self._post_hook, with_kwargs=True))
            except TypeError:
                self.cache_by_shape = False
                self._hooks.append(trainer.model.register_forward_pre_hook(lambda m, a: self._pre_hook(m, a, {})))
                self._hooks.append(trainer.model.register_forward_hook(lambda m, a, o: self._post_hook(m, a, {}, o)))
            return

        # Fallback: Linear/Conv2d forward-only (train only)
        self._fallback_hooks = []
        self._fallback_by_module = {}
        for m in trainer.model.modules():
            if isinstance(m, nn.Linear):
                self._fallback_hooks.append(m.register_forward_hook(self._hook_linear_fallback))
            elif isinstance(m, nn.Conv2d):
                self._fallback_hooks.append(m.register_forward_hook(self._hook_conv2d_fallback))

    def on_fit_end(self, trainer: "GenericTrainer"):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

        for h in self._fallback_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._fallback_hooks = []

        self._trainer = None

    def on_epoch_start(self, trainer: "GenericTrainer", epoch: int):
        self._epoch_start = (self._fwd_total, self._fwd_train, self._fwd_val, self._fwd_test, self._bwd_train)

    def on_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        t0, tr0, v0, te0, b0 = self._epoch_start
        df_total = int(self._fwd_total - t0)
        df_train = int(self._fwd_train - tr0)
        df_val = int(self._fwd_val - v0)
        df_test = int(self._fwd_test - te0)
        db_train = int(self._bwd_train - b0)
        dtrain_total = int(df_train + db_train)

        self._per_epoch["fwd_total"].append(df_total)
        self._per_epoch["fwd_train"].append(df_train)
        self._per_epoch["fwd_val"].append(df_val)
        self._per_epoch["fwd_test"].append(df_test)
        self._per_epoch["bwd_train"].append(db_train)
        self._per_epoch["train_total"].append(dtrain_total)

        # Expose last-epoch + totals to trainer for W&B logging (no extra sync)
        try:
            trainer._last_epoch_flops_fwd_total = int(df_total)
            trainer._last_epoch_flops_fwd_train = int(df_train)
            trainer._last_epoch_flops_fwd_val = int(df_val)
            trainer._last_epoch_flops_fwd_test = int(df_test)

            # New: train backward + exact train total (forward+backward)
            trainer._last_epoch_flops_bwd_train = int(db_train)

            trainer._flops_fwd_total = int(self._fwd_total)
            trainer._flops_fwd_train_total = int(self._fwd_train)

            # Empirical backward factor (fallback to configured factor if not measurable)
            if df_train > 0 and db_train >= 0:
                bf = float(db_train) / float(df_train)
            else:
                bf = float(self.backward_factor)

            trainer._flops_backward_factor = float(bf)

            # Exact cumulative train total if we measured backward, else estimate
            if self._bwd_train > 0:
                trainer._flops_train_total_est = int(self._fwd_train + self._bwd_train)
            else:
                trainer._flops_train_total_est = int(self._fwd_train * (1.0 + float(self.backward_factor)))
        except Exception:
            pass

    def update_summary(self, export: Dict[str, Any]) -> None:
        export.setdefault("flops", {})

        if FlopCounterMode is not None:
            export["flops"]["method"] = "torch.utils.flop_counter.FlopCounterMode"
            export["flops"]["cache_by_shape"] = bool(self.cache_by_shape)
            export["flops"]["num_cached_signatures_eval_fwd"] = int(len(self._cache))
            export["flops"]["num_cached_signatures_train_step"] = int(len(self._train_cache))

            export["flops"]["forward_total"] = int(self._fwd_total)
            export["flops"]["forward_train_total"] = int(self._fwd_train)
            export["flops"]["forward_val_total"] = int(self._fwd_val)
            export["flops"]["forward_test_total"] = int(self._fwd_test)

            export["flops"]["backward_train_total"] = int(self._bwd_train)
            export["flops"]["train_total_exact"] = int(self._fwd_train + self._bwd_train)

            bf = None
            if self._fwd_train > 0:
                bf = float(self._bwd_train) / float(self._fwd_train)
            export["flops"]["backward_factor_empirical"] = bf
            export["flops"]["backward_factor_fallback"] = float(self.backward_factor)

            # Keep legacy key too (now exact when backward measured)
            export["flops"]["train_total_est"] = int(self._fwd_train + self._bwd_train) if self._bwd_train > 0 else int(
                self._fwd_train * (1.0 + float(self.backward_factor))
            )

            export["flops"]["per_epoch"] = {
                "fwd_total": [int(x) for x in self._per_epoch["fwd_total"]],
                "fwd_train": [int(x) for x in self._per_epoch["fwd_train"]],
                "bwd_train": [int(x) for x in self._per_epoch["bwd_train"]],
                "train_total": [int(x) for x in self._per_epoch["train_total"]],
                "fwd_val": [int(x) for x in self._per_epoch["fwd_val"]],
                "fwd_test": [int(x) for x in self._per_epoch["fwd_test"]],
            }
        else:
            export["flops"]["method"] = "fallback_linear_conv_hooks"
            export["flops"]["forward_by_module"] = {k: int(v) for k, v in self._fallback_by_module.items()}
            export["flops"]["forward_total"] = int(sum(self._fallback_by_module.values()))
            export["flops"]["train_total_est"] = int(export["flops"]["forward_total"] * (1.0 + float(self.backward_factor)))
            export["flops"]["backward_factor_fallback"] = float(self.backward_factor)
        
# ------------------------------
# GPU peak memory (per-epoch, no extra sync)
# ------------------------------

class CudaPeakMemoryCallback(Callback):
    """
    Records CUDA peak memory per epoch.

    - reset_peak_memory_stats() at epoch start
    - capture max_memory_{allocated,reserved} after *training* epoch
      (so validation/test allocations don't contaminate the measurement)

    Exposes:
      trainer._last_epoch_peak_alloc_bytes
      trainer._last_epoch_peak_reserved_bytes

    And exports:
      export["gpu"]["peak_allocated_bytes_per_epoch"]
      export["gpu"]["peak_reserved_bytes_per_epoch"]
    """
    def __init__(self, *, enabled: bool = True, capture_after: str = "train"):
        self.enabled = bool(enabled)
        self.capture_after = str(capture_after).strip().lower()  # "train" or "epoch"
        self._alloc_per_epoch: List[int] = []
        self._rsvd_per_epoch: List[int] = []

    def _cuda_ok(self, trainer: "GenericTrainer") -> bool:
        return bool(self.enabled and getattr(trainer, "_cuda_enabled", False) and torch.cuda.is_available())

    def on_epoch_start(self, trainer: "GenericTrainer", epoch: int):
        if not self._cuda_ok(trainer):
            return
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _capture(self, trainer: "GenericTrainer", epoch: int):
        if not self._cuda_ok(trainer):
            return
        try:
            alloc = int(torch.cuda.max_memory_allocated())
            rsvd = int(torch.cuda.max_memory_reserved())
        except Exception:
            return

        self._alloc_per_epoch.append(alloc)
        self._rsvd_per_epoch.append(rsvd)

        # Make available to W&B epoch logger (no change needed in trainer)
        trainer._last_epoch_peak_alloc_bytes = alloc
        trainer._last_epoch_peak_reserved_bytes = rsvd

    def on_train_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        if self.capture_after == "train":
            self._capture(trainer, epoch)

    def on_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        if self.capture_after == "epoch":
            self._capture(trainer, epoch)

    def update_summary(self, export: Dict[str, Any]) -> None:
        try:
            export.setdefault("gpu", {})
            export["gpu"]["peak_allocated_bytes_per_epoch"] = [int(x) for x in self._alloc_per_epoch]
            export["gpu"]["peak_reserved_bytes_per_epoch"] = [int(x) for x in self._rsvd_per_epoch]
        except Exception:
            pass


# ------------------------------
# W&B callback (sync-minimized)
# ------------------------------

class WandbCallback(Callback):
    """
    W&B logger designed to minimize GPU syncs.

    Guarantees / intentions:
      - Never calls .item() on CUDA tensors inside per-step hooks.
      - Uses _AsyncScalarD2H for best-effort train loss (no forced sync).
      - Logs epoch/validation metrics (Python floats already produced by the trainer).
      - Logs run_meta / train_config to wandb.config (one-time) with size trimming.
      - Optional artifact upload of JSON files at end.

    Fast mode notes:
      - Train loss logged from per-step hook is best-effort (may lag slightly).
      - Validation may be less frequent and may include fewer metrics depending on trainer config.
    """
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: Optional[str] = None,  # "online" | "offline" | "disabled"
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        log_every_n_steps: int = 50,
        watch_model: bool = False,
        upload_artifacts: bool = False,
        out_dir: Optional[str] = None,
        log_run_meta_to_config: bool = True,
        log_train_config_to_config: bool = True,
        log_epoch_throughput: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.mode = (mode or "").strip().lower() if mode is not None else None
        self.run_name = run_name
        self.tags = tags
        self.config = config or {}
        self.log_every_n_steps = int(log_every_n_steps) if log_every_n_steps else 50
        self.watch_model = bool(watch_model)
        self.upload_artifacts = bool(upload_artifacts)
        self.out_dir = out_dir

        self.log_run_meta_to_config = bool(log_run_meta_to_config)
        self.log_train_config_to_config = bool(log_train_config_to_config)
        self.log_epoch_throughput = bool(log_epoch_throughput)

        self._enabled = False
        self._wandb = None
        self._run = None
        self._loss_meter: Optional[_AsyncScalarD2H] = None

    def _safe_small_meta(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        small = dict(d)

        tb = small.get("trainable_breakdown", None)
        if isinstance(tb, dict):
            tb2 = dict(tb)
            if "trainable_module_paths" in tb2:
                v = tb2.get("trainable_module_paths")
                if isinstance(v, list):
                    tb2["trainable_module_paths"] = f"<omitted:{len(v)}>"
            small["trainable_breakdown"] = tb2

        for k in ["trainable_module_paths", "state_dict_keys", "all_param_names"]:
            if k in small:
                small[k] = "<omitted>"

        return small

    def on_fit_start(self, trainer: "GenericTrainer"):
        if self.mode == "disabled":
            self._enabled = False
            return
        try:
            import wandb
            self._wandb = wandb
        except Exception:
            self._enabled = False
            return

        init_kwargs: Dict[str, Any] = dict(
            project=self.project,
            entity=self.entity,
            group=self.group,
            name=self.run_name,
            tags=self.tags,
            config=self.config,
        )
        if self.mode in {"offline", "online"}:
            init_kwargs["mode"] = self.mode

        if self.out_dir:
            try:
                Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                init_kwargs["dir"] = str(Path(self.out_dir).resolve())
            except Exception:
                pass

        try:
            self._run = self._wandb.init(**init_kwargs)
            self._enabled = True
        except Exception:
            self._enabled = False
            self._run = None
            return

        cuda_enabled = bool(getattr(trainer, "_cuda_enabled", False))
        self._loss_meter = _AsyncScalarD2H(enabled=cuda_enabled)

        if self.watch_model and self._enabled:
            try:
                self._wandb.watch(trainer.model, log="none")
            except Exception:
                pass

        # One-time config push (keep it small)
        try:
            cfg_update: Dict[str, Any] = {}

            if self.log_run_meta_to_config:
                meta = getattr(trainer, "run_meta", None)
                if isinstance(meta, dict):
                    cfg_update["run_meta"] = self._safe_small_meta(meta)

            if self.log_train_config_to_config:
                tc = getattr(trainer, "cfg", None)
                if tc is not None:
                    try:
                        import dataclasses
                        if dataclasses.is_dataclass(tc):
                            cfg_update["train_config"] = dataclasses.asdict(tc)
                        elif isinstance(tc, dict):
                            cfg_update["train_config"] = dict(tc)
                    except Exception:
                        pass

            try:
                cfg_update["git_commit"] = getattr(trainer, "git_commit", None)
                cfg_update["method_name"] = type(getattr(trainer, "method", None)).__name__
                cfg_update["primary_metric"] = getattr(trainer, "primary_metric", None)
                cfg_update["best_metric_cfg"] = getattr(getattr(trainer, "cfg", None), "best_metric", None)
                cfg_update["best_selection_name"] = getattr(trainer, "best_selection_name", None)
            except Exception:
                pass

            if cfg_update and self._run is not None:
                self._run.config.update(cfg_update, allow_val_change=True)
        except Exception:
            pass

    def on_train_batch_end(self, trainer: "GenericTrainer", batch_idx: int, global_step: int, loss: torch.Tensor):
        if not self._enabled:
            return

        if self._loss_meter is not None and isinstance(loss, torch.Tensor) and loss.numel() == 1:
            self._loss_meter.push(loss.detach())

        if self.log_every_n_steps <= 0:
            return
        if (global_step % self.log_every_n_steps) != 0:
            return

        log: Dict[str, Any] = {}

        if self._loss_meter is not None:
            v = self._loss_meter.poll()
            if isinstance(v, (int, float)):
                log["train/loss"] = float(v)

        try:
            if getattr(trainer, "opt", None) is not None:
                log["train/lr"] = float(trainer.opt.param_groups[0].get("lr", 0.0))
        except Exception:
            pass

        # These are expected to be Python floats cached by the trainer (no sync here).
        try:
            if getattr(trainer, "_last_grad_norm", None) is not None:
                log["train/grad_norm"] = float(trainer._last_grad_norm)
        except Exception:
            pass
        try:
            if getattr(trainer, "_last_amp_scale", None) is not None:
                log["train/amp_scale"] = float(trainer._last_amp_scale)
        except Exception:
            pass

        if log:
            try:
                if self._run is not None:
                    self._run.log(log, step=int(global_step))
                else:
                    self._wandb.log(log, step=int(global_step))
            except Exception:
                pass

    def on_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        if (not self._enabled) or (not self.log_epoch_throughput):
            return

        log: Dict[str, Any] = {"epoch": int(epoch)}

        try:
            tr_sec = getattr(trainer, "_last_epoch_train_sec", None)
            ep_sec = getattr(trainer, "_last_epoch_sec", None)
            seen = getattr(trainer, "_last_epoch_train_seen", None)
            toks = getattr(trainer, "_last_epoch_train_tokens", None)
            opt_steps = getattr(trainer, "_last_epoch_train_opt_steps", None)

            if isinstance(tr_sec, (int, float)):
                log["time/train_epoch_sec"] = float(tr_sec)
            if isinstance(ep_sec, (int, float)):
                log["time/epoch_total_sec"] = float(ep_sec)
            if isinstance(seen, int):
                log["train/seen"] = int(seen)
            if isinstance(toks, int):
                log["train/tokens"] = int(toks)
            if isinstance(opt_steps, int):
                log["train/opt_steps"] = int(opt_steps)

            if isinstance(tr_sec, (int, float)) and tr_sec > 0 and isinstance(seen, int):
                log["train/samples_per_s"] = float(seen) / float(tr_sec)
            if isinstance(tr_sec, (int, float)) and tr_sec > 0 and isinstance(toks, int) and toks > 0:
                log["train/tokens_per_s"] = float(toks) / float(tr_sec)

            bv = getattr(trainer, "best_selection_value", None)
            bn = getattr(trainer, "best_selection_name", None)
            be = getattr(trainer, "best_epoch", None)
            if bn is not None:
                log["best/selection_name"] = str(bn)
            if isinstance(bv, (int, float)):
                log["best/selection_value"] = float(bv)
            if isinstance(be, int):
                log["best/epoch"] = int(be)
                
            # GPU peak mem (if available)
            try:
                pa = getattr(trainer, "_last_epoch_peak_alloc_bytes", None)
                pr = getattr(trainer, "_last_epoch_peak_reserved_bytes", None)
                if isinstance(pa, int):
                    log["gpu/peak_alloc_bytes"] = int(pa)
                if isinstance(pr, int):
                    log["gpu/peak_reserved_bytes"] = int(pr)
            except Exception:
                pass
            
            # FLOPs (if available)
            try:
                f_tr = getattr(trainer, "_last_epoch_flops_fwd_train", None)
                f_tot = getattr(trainer, "_last_epoch_flops_fwd_total", None)
                bf = getattr(trainer, "_flops_backward_factor", None)

                if isinstance(f_tr, int):
                    log["flops/fwd_train_epoch"] = int(f_tr)
                    if isinstance(bf, (int, float)):
                        log["flops/train_total_est_epoch"] = int(f_tr * (1.0 + float(bf)))

                if isinstance(f_tot, int):
                    log["flops/fwd_total_epoch"] = int(f_tot)

                # optional: cumulative
                c_tr = getattr(trainer, "_flops_fwd_train_total", None)
                c_est = getattr(trainer, "_flops_train_total_est", None)
                if isinstance(c_tr, int):
                    log["flops/fwd_train_cum"] = int(c_tr)
                if isinstance(c_est, int):
                    log["flops/train_total_est_cum"] = int(c_est)
            except Exception:
                pass

        except Exception:
            pass

        try:
            step = int(getattr(trainer, "global_step", epoch))
            if self._run is not None:
                self._run.log(log, step=step)
            else:
                self._wandb.log(log, step=step)
        except Exception:
            pass

    def on_validation_end(self, trainer: "GenericTrainer", epoch: int, metrics: Dict[str, Any], score: float):
        if not self._enabled:
            return

        # Trainer provides python floats here (coalesced sync already happened).
        log: Dict[str, Any] = {"epoch": int(epoch)}

        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)):
                log[f"val/{k}"] = float(v)
        try:
            log["val/score"] = float(score)
        except Exception:
            pass

        try:
            if getattr(trainer, "best_selection_name", None) is not None:
                log["best/selection_name"] = str(trainer.best_selection_name)
            if getattr(trainer, "best_selection_value", None) is not None:
                bv = trainer.best_selection_value
                if isinstance(bv, (int, float)):
                    log["best/selection_value"] = float(bv)
            if getattr(trainer, "best_epoch", None) is not None:
                log["best/epoch"] = int(trainer.best_epoch)
        except Exception:
            pass

        step = int(getattr(trainer, "global_step", epoch))
        try:
            if self._run is not None:
                self._run.log(log, step=step)
            else:
                self._wandb.log(log, step=step)
        except Exception:
            pass

    def update_summary(self, export: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        # Keep W&B summary SMALL + FLAT (W&B summary has practical limits).
        def _get(d, *ks, default=None):
            cur = d
            for k in ks:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        summary: Dict[str, Any] = {}
        try:
            summary["git_commit"] = export.get("git_commit")
            summary["device"] = export.get("device")
            summary["epochs"] = export.get("epochs")

            summary["params/total"] = _get(export, "params", "total")
            summary["params/trainable"] = _get(export, "params", "trainable")
            summary["params/trainable_fraction"] = _get(export, "params", "trainable_fraction")

            summary["best/metric_name"] = _get(export, "best", "metric_name")
            summary["best/value"] = _get(export, "best_selection", "value")
            summary["best/epoch"] = _get(export, "best", "epoch")
            summary["best/checkpoint"] = _get(export, "best", "checkpoint")
            summary["best/val_loss"] = _get(export, "best", "val_loss")

            bvm = _get(export, "best", "val_metrics", default={})
            if isinstance(bvm, dict):
                for k in ("acc", "f1", "mcc", "pearson"):
                    if k in bvm and isinstance(bvm[k], (int, float)):
                        summary[f"best/{k}"] = float(bvm[k])

            sps = _get(export, "epoch_train", "samples_per_s")
            tps = _get(export, "epoch_train", "tokens_per_s")
            if isinstance(sps, list) and sps:
                summary["train/samples_per_s_last"] = float(sps[-1])
            if isinstance(tps, list) and tps:
                summary["train/tokens_per_s_last"] = float(tps[-1])

            summary["test/loss"] = _get(export, "test", "loss")
            tmet = _get(export, "test", "metrics", default={})
            if isinstance(tmet, dict):
                for k, v in tmet.items():
                    if isinstance(v, (int, float)):
                        summary[f"test/{k}"] = float(v)
                        
            # FLOPs summary (from export)
            summary["flops/fwd_total"] = _get(export, "flops", "forward_total")
            summary["flops/fwd_train_total"] = _get(export, "flops", "forward_train_total")
            summary["flops/train_total_est"] = _get(export, "flops", "train_total_est")
            summary["flops/num_cached_signatures"] = _get(export, "flops", "num_cached_signatures")
            summary["flops/method"] = _get(export, "flops", "method")

            summary = {k: v for k, v in summary.items() if v is not None}

            if self._run is not None:
                self._run.summary.update(summary)
            else:
                self._wandb.summary.update(summary)
        except Exception:
            pass

    def on_fit_end(self, trainer: "GenericTrainer"):
        if not self._enabled:
            return

        # --- Log final TEST metrics to W&B history (charts), if available ---
        try:
            tm = getattr(trainer, "_test_metrics", None)
            tl = getattr(trainer, "_test_loss", None)

            if isinstance(tm, dict) and (tm or isinstance(tl, (int, float))):
                log: Dict[str, Any] = {}

                # Prefer explicit trainer cached loss, but also support metrics["loss"]
                if isinstance(tl, (int, float)):
                    log["test/loss"] = float(tl)

                for k, v in tm.items():
                    if isinstance(v, (int, float)):
                        log[f"test/{k}"] = float(v)

                # Convenience: test/score = primary metric if present
                pm = getattr(trainer, "primary_metric", None)
                if isinstance(pm, str) and pm in tm and isinstance(tm[pm], (int, float)):
                    log["test/score"] = float(tm[pm])

                # Put it at a fresh step so it doesn't collide with last val log step
                step = int(getattr(trainer, "global_step", 0)) + 1

                if log:
                    if self._run is not None:
                        self._run.log(log, step=step)
                    else:
                        self._wandb.log(log, step=step)
        except Exception:
            pass


        if self.upload_artifacts and self._run is not None:
            try:
                art = self._wandb.Artifact(
                    name=f"{self._run.id}_artifacts",
                    type="run_outputs",
                )

                run_dir = getattr(trainer, "run_dir", None)
                if isinstance(run_dir, str) and os.path.isdir(run_dir):
                    cand = [
                        os.path.join(run_dir, "run_meta.json"),
                        os.path.join(run_dir, "method_meta.json"),
                        os.path.join(run_dir, f"{getattr(getattr(trainer, 'cfg', None), 'save_name_prefix', 'model')}-metrics.json"),
                    ]
                    for p in cand:
                        if os.path.isfile(p):
                            art.add_file(p)

                self._run.log_artifact(art)
            except Exception:
                pass

        try:
            if self._run is not None:
                self._run.finish()
            else:
                self._wandb.finish()
        except Exception:
            pass


# ------------------------------
# BA-sparse callbacks (defensive)
# ------------------------------

class BASparsityLoggerCallback(Callback):
    """
    Logs BA mask keep fraction / sparsity once per epoch without forcing GPU sync.
    """
    def __init__(self, numeric_eps: float = 1e-6):
        self.numeric_eps = float(numeric_eps)
        self._last: Dict[str, Any] = {}
        self._keep_meter: Optional[_AsyncScalarD2H] = None

    def on_fit_start(self, trainer: "GenericTrainer"):
        self._keep_meter = _AsyncScalarD2H(enabled=bool(getattr(trainer, "_cuda_enabled", False)))

    def on_train_epoch_end(self, trainer: "GenericTrainer", epoch: int):
        m = getattr(trainer, "method", None)
        if m is None:
            return
        mask = getattr(m, "mask", None) or getattr(m, "ba_mask", None) or getattr(m, "sparse_mask", None)
        if mask is None or not isinstance(mask, torch.Tensor):
            return

        try:
            t = mask.detach()
            keep_t = (t.abs() > self.numeric_eps).float().mean()
            if self._keep_meter is not None:
                self._keep_meter.push(keep_t)
                keep = self._keep_meter.poll()
            else:
                keep = _to_finite_float(keep_t)
            if isinstance(keep, (int, float)):
                keep = float(keep)
                self._last = {"ba_keep_frac": keep, "ba_sparsity": float(1.0 - keep), "epoch": int(epoch)}
        except Exception:
            pass

    def update_summary(self, export: Dict[str, Any]) -> None:
        if self._last:
            export.setdefault("ba", {})
            export["ba"].update(self._last)


class BASparseFinalCallback(Callback):
    """
    For 'ba_sparse_final' style recipes: finalize/freeze mask once (best-effort).
    """
    def __init__(self, eval_masked_test: bool = True, respect_save_json_only: bool = True):
        self.eval_masked_test = bool(eval_masked_test)
        self.respect_save_json_only = bool(respect_save_json_only)

    def on_fit_start(self, trainer: "GenericTrainer"):
        m = getattr(trainer, "method", None)
        if m is None:
            return
        for name in ["finalize_mask", "apply_final_mask", "freeze_mask"]:
            fn = getattr(m, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break


# ------------------------------
# Extra eval callbacks (kept minimal)
# ------------------------------

class ZeroShotEvalCallback(Callback):
    """
    Run an additional evaluation pass at end of training (synced eval).
    """
    def __init__(
        self,
        split: str = "test",
        stage_name: str = "zero_shot",
        log_to_wandb: bool = False,
        eval_after: bool = False,
        limit_batches=None,
    ):
        self.split = split
        self.stage_name = stage_name
        self.log_to_wandb = bool(log_to_wandb)
        self.eval_after = bool(eval_after)
        self.limit_batches = limit_batches

    def on_fit_end(self, trainer: "GenericTrainer"):
        if not self.eval_after:
            return
        loader = getattr(trainer, "test_loader", None) if self.split == "test" else getattr(trainer, "val_loader", None)
        if loader is None:
            return
        try:
            vloss, vmetrics, vscore = trainer._evaluate(
                loader,
                limit_batches=self.limit_batches,
                stage=self.split,
                compute_metrics=True,
                no_gpu_sync=False,
            )
            print(f"[{self.stage_name.upper()}] loss={vloss:.4f} score={vscore:.4f} metrics={vmetrics}")
        except Exception as e:
            print(f"[WARN] {self.stage_name} eval failed: {e}")


class PretrainEvalCallback(Callback):
    """
    Evaluate a provided pretrain/heldout loader at end of training (synced eval).
    """
    def __init__(
        self,
        pretrain_loader,
        limit_batches=None,
        stage_name: str = "pretrain",
        log_to_wandb: bool = False,
    ):
        self.pretrain_loader = pretrain_loader
        self.limit_batches = limit_batches
        self.stage_name = stage_name
        self.log_to_wandb = bool(log_to_wandb)
        self._last: Dict[str, Any] = {}

    def on_fit_end(self, trainer: "GenericTrainer"):
        if self.pretrain_loader is None:
            return
        try:
            vloss, vmetrics, vscore = trainer._evaluate(
                self.pretrain_loader,
                limit_batches=self.limit_batches,
                stage="val",
                compute_metrics=True,
                no_gpu_sync=False,
            )
            self._last = {"loss": float(vloss), "score": float(vscore), "metrics": vmetrics}
            print(f"[{self.stage_name.upper()}] loss={vloss:.4f} score={vscore:.4f} metrics={vmetrics}")
        except Exception as e:
            print(f"[WARN] {self.stage_name} eval failed: {e}")

    def update_summary(self, export: Dict[str, Any]) -> None:
        if self._last:
            export.setdefault("pretrain_eval", {})
            export["pretrain_eval"].update(self._last)