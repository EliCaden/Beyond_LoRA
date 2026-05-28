# models/lora_layers.py
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearAdapter(nn.Module):
    """
    Wrap nn.Linear with LoRA:
      y = base(x) + delta(x)
      delta(x) = (alpha/r) * B(A(x))

    Shapes:
      A: (r, in_features)
      B: (out_features, r)

    Notes:
      - Base linear is frozen by default.
      - detach_base=True (default) prevents gradients into base even if someone unfreezes it later.
        Set detach_base=False to fine-tune base weights while wrapped.
      - Supports optional Sparse-A fast path (index_select) with A rewritten to a consistent row-sparse matrix.
      - Supports optional BA masking (T_alpha on |B@A|) with cached masks.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        *,
        freeze_base: bool = True,
        detach_base: bool = True,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.dim() != 2:
            raise TypeError("LoRALinearAdapter expects nn.Linear with 2D weight.")
        if r < 1:
            raise ValueError(f"r={r} must be >= 1")
        if r > base.in_features:
            raise ValueError(f"r={r} > in_features={base.in_features}")

        self.base = base
        self.r = int(r)

        self.detach_base = bool(detach_base)

        self.alpha = int(alpha)
        if self.alpha == 0:
            # Preserve prior behavior.
            print("[LoRA] alpha=0 -> using 2*r for scaling")
            self.alpha = 2 * self.r

        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        dev = base.weight.device
        dt = base.weight.dtype

        # Persist alpha so checkpoints can round-trip; MUST keep scaling synced on load.
        self.register_buffer("lora_alpha", torch.tensor(float(self.alpha), device=dev, dtype=torch.float32), persistent=True)

        # scaling is a python float used in hot paths; keep it in sync with lora_alpha.
        self.scaling: float = float(self.alpha) / max(1, self.r)

        # Trainable LoRA params
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r, device=dev, dtype=dt))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Cheap/Shuffle state
        self._cheap_offset = 0
        self._shuffle_stream = None
        self._shuffle_ptr = 0

        # ----- Sparse-A optimization (non-persistent) -----
        self._sparse_A_active = False
        self.register_buffer("_sparse_idx", torch.empty(0, dtype=torch.long, device=dev), persistent=False)
        self.register_buffer("_sparse_scale", torch.empty(0, dtype=dt, device=dev), persistent=False)

        # ----- BA mask state -----
        self._ba_mask_enabled = False
        self._ba_fixmask_sparse = False
        self._ba_use_external_step = False

        self._ba_mask_alpha: float = 0.5
        self._ba_mask_ties_keep: bool = True
        self._ba_mask_rounding: str = "ceil"
        self._ba_mask_every: int = 1
        self._ba_mask_step: int = 0
        self._ba_last_refresh_step: int = -1

        self.register_buffer("_ba_cached_mask", torch.empty(0, 0, dtype=torch.bool, device=dev), persistent=False)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}, "
            f"scaling={self.scaling:.6g}, detach_base={self.detach_base}"
        )

    def _sync_alpha_scaling_from_buffer(self) -> None:
        # Called after state_dict load to ensure scaling matches checkpoint alpha.
        try:
            a = float(self.lora_alpha.detach().item())
        except Exception:
            a = float(self.alpha)
        a_int = int(round(a))
        if a_int <= 0:
            a_int = max(1, int(self.alpha))
        self.alpha = a_int
        self.scaling = float(self.alpha) / max(1, self.r)
        # Keep buffer consistent with integer alpha.
        try:
            self.lora_alpha.fill_(float(self.alpha))
        except Exception:
            pass

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # If checkpoint provided lora_alpha, ensure scaling reflects it.
        key = prefix + "lora_alpha"
        if key in state_dict:
            self._sync_alpha_scaling_from_buffer()

    # ---------------------------------------------------------------------
    # Base path (optional defensive detach)
    # ---------------------------------------------------------------------

    def _base_linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        Base path:
        - If detach_base=True, prevent gradients into base params even if someone unfreezes them.
        - If detach_base=False, behave like normal nn.Linear (grad into base if requires_grad).
        """
        if not self.detach_base:
            return self.base(x)

        w = self.base.weight
        b = self.base.bias
        if w.requires_grad:
            w = w.detach()
        if b is not None and b.requires_grad:
            b = b.detach()
        return F.linear(x, w, b)

    # ---------------------------------------------------------------------
    # LoRA delta path (exposed for fused-qkv wrappers)
    # ---------------------------------------------------------------------

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return *scaled* delta(x) only (no base path).
        """
        # ---- Standard LoRA ----
        if not self._ba_mask_enabled:
            if self._sparse_A_active and self._sparse_idx.numel() == self.r:
                xA = x.index_select(-1, self._sparse_idx)  # (..., r)
                if self._sparse_scale.numel() == self.r:
                    xA = xA * self._sparse_scale.view(*([1] * (xA.dim() - 1)), -1)
            else:
                xA = F.linear(x, self.A)  # (..., r)

            xAB = F.linear(xA, self.B)  # (..., out)
            return self.scaling * xAB

        # ---- BA-masked path ----
        if self._ba_fixmask_sparse:
            return self.scaling * self._forward_ba_sparse_fix_mask(x)

        C = self.B @ self.A  # (out, in)
        if self._need_refresh_mask(C.shape):
            self._refresh_mask_from_C_abs(C.abs())

        if not self._ba_use_external_step:
            self._ba_mask_step += 1

        C_masked = C.masked_fill(~self._ba_cached_mask, 0)
        return self.scaling * F.linear(x, C_masked)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base_linear(x) + self.forward_delta(x)

    # ---------------------------------------------------------------------
    # Mask refresh helpers
    # ---------------------------------------------------------------------

    def _need_refresh_mask(self, shape: Tuple[int, int]) -> bool:
        if self._ba_cached_mask.numel() == 0 or tuple(self._ba_cached_mask.shape) != tuple(shape):
            return True
        if (self._ba_mask_step % self._ba_mask_every) != 0:
            return False
        if self._ba_use_external_step and (self._ba_last_refresh_step == self._ba_mask_step):
            return False
        return True

    @torch.no_grad()
    def _refresh_mask_from_C_abs(self, C_abs: torch.Tensor) -> None:
        M = self._talpha_mask_paper(
            C_abs,
            self._ba_mask_alpha,
            ties_keep=self._ba_mask_ties_keep,
            rounding=self._ba_mask_rounding,
        )
        self._ba_cached_mask.resize_(M.shape)
        self._ba_cached_mask.copy_(M)
        self._ba_last_refresh_step = int(self._ba_mask_step)

    # ---------------------------------------------------------------------
    # Merge / init helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """
        Merge current (possibly masked) delta into base.weight and zero A/B.
        """
        if self._ba_mask_enabled:
            C = self.B @ self.A
            if self._ba_cached_mask.numel() and self._ba_cached_mask.shape == C.shape:
                M = self._ba_cached_mask
            else:
                M = self._talpha_mask_paper(
                    C.abs(),
                    self._ba_mask_alpha,
                    ties_keep=self._ba_mask_ties_keep,
                    rounding=self._ba_mask_rounding,
                )
            deltaW = self.scaling * (C * M.to(C.dtype))
        else:
            deltaW = self.scaling * (self.B @ self.A)

        self.base.weight.data.add_(deltaW.to(self.base.weight.dtype))
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)
        self.disable_sparse_A()

    def freeze_A(self, flag: bool = True) -> None:
        self.A.requires_grad_(not flag)

    def freeze_B(self, flag: bool = True) -> None:
        self.B.requires_grad_(not flag)

    @torch.no_grad()
    def set_A_gaussian(self, std: float = 0.02) -> None:
        nn.init.normal_(self.A, std=std)

    @torch.no_grad()
    def set_A_zero(self) -> None:
        nn.init.zeros_(self.A)

    @torch.no_grad()
    def set_B_gaussian(self, std: float = 0.02) -> None:
        nn.init.normal_(self.B, std=std)

    @torch.no_grad()
    def set_B_zero(self) -> None:
        nn.init.zeros_(self.B)

    @torch.no_grad()
    def set_A_identity_window(self, offset: int = 0) -> None:
        if offset < 0 or offset + self.r > self.in_features:
            raise ValueError(f"offset {offset} with r={self.r} out of range for in_features={self.in_features}")
        self.A.zero_()
        dev = self.A.device
        rows = torch.arange(self.r, device=dev)
        cols = torch.arange(offset, offset + self.r, device=dev)
        self.A[rows, cols] = 1.0
        self._cheap_offset = int(offset)
        self.disable_sparse_A()

    # ---------------------------------------------------------------------
    # Sparse-A utilities
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def enable_sparse_A_select(self, idx, scale=None) -> None:
        """
        Enable sparse-A fast path and rewrite A to an equivalent row-sparse matrix:
          A[row, idx[row]] = scale[row] (or 1.0)
        """
        dev = self.A.device
        dt = self.A.dtype

        idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev).view(-1)
        if idx_t.numel() != self.r:
            raise ValueError(f"sparse-A: idx must have shape (r,), got {tuple(idx_t.shape)} with r={self.r}")
        # Validation; may sync once if on CUDA, but this is not per-step.
        if int(idx_t.min().item()) < 0 or int(idx_t.max().item()) >= self.in_features:
            raise ValueError("sparse-A: idx contains out-of-range column indices")

        if scale is None:
            scale_t = torch.ones(self.r, dtype=dt, device=dev)
        else:
            scale_t = torch.as_tensor(scale, dtype=dt, device=dev).view(-1)
            if scale_t.numel() != self.r:
                raise ValueError(f"sparse-A: scale must have shape (r,), got {tuple(scale_t.shape)} with r={self.r}")

        # Rewrite A for correctness.
        self.A.zero_()
        rows = torch.arange(self.r, device=dev)
        self.A[rows, idx_t] = scale_t

        # Fast-path buffers
        self._sparse_idx.resize_(idx_t.shape)
        self._sparse_idx.copy_(idx_t)
        self._sparse_scale.resize_(scale_t.shape)
        self._sparse_scale.copy_(scale_t)

        self._sparse_A_active = True
        self.A.requires_grad_(False)

    @torch.no_grad()
    def enable_sparse_A_from_current(self) -> None:
        vals, idx = self.A.detach().abs().max(dim=1)
        signs = torch.sign(self.A.detach()[torch.arange(self.r, device=self.A.device), idx])
        scale = (vals * signs).to(self.A.dtype)
        self.enable_sparse_A_select(idx=idx, scale=scale)

    def disable_sparse_A(self) -> None:
        self._sparse_A_active = False
        self._sparse_idx.resize_(0)
        self._sparse_scale.resize_(0)
        self.A.requires_grad_(True)

    # ---------------------------------------------------------------------
    # ShuffleLoRA utilities
    # ---------------------------------------------------------------------

    def build_shuffle_stream(self) -> None:
        max_start = max(0, self.in_features - self.r)
        windows = list(range(0, max_start + 1, self.r)) or [0]
        self._shuffle_stream = windows
        self._shuffle_ptr = 0

    def shuffle_stream_once(self, g: torch.Generator | None = None) -> None:
        if self._shuffle_stream is None:
            self.build_shuffle_stream()
        idx = torch.randperm(len(self._shuffle_stream), generator=g).tolist()
        self._shuffle_stream = [self._shuffle_stream[i] for i in idx]
        self._shuffle_ptr = 0

    @torch.no_grad()
    def set_A_next_shuffle_window(self) -> None:
        if self._shuffle_stream is None:
            self.build_shuffle_stream()
        off = self._shuffle_stream[self._shuffle_ptr]
        self.set_A_identity_window(off)
        self._shuffle_ptr = (self._shuffle_ptr + 1) % len(self._shuffle_stream)

    # ---------------------------------------------------------------------
    # BA masking
    # ---------------------------------------------------------------------

    @staticmethod
    def _talpha_mask_paper(
        absS: torch.Tensor,
        alpha: float,
        *,
        ties_keep: bool = True,
        rounding: str = "ceil",
    ) -> torch.Tensor:
        m, n = absS.shape
        if rounding == "ceil":
            k_row = max(1, math.ceil(alpha * n))
            k_col = max(1, math.ceil(alpha * m))
        elif rounding == "floor":
            k_row = max(1, math.floor(alpha * n))
            k_col = max(1, math.floor(alpha * m))
        else:
            raise ValueError("rounding must be 'ceil' or 'floor'.")

        row_cut = torch.topk(absS, k_row, dim=1, largest=True, sorted=True).values[:, -1]
        col_cut = torch.topk(absS, k_col, dim=0, largest=True, sorted=True).values[-1, :]

        if ties_keep:
            row_ok = absS >= row_cut.unsqueeze(1)
            col_ok = absS >= col_cut.unsqueeze(0)
        else:
            row_ok = absS > row_cut.unsqueeze(1)
            col_ok = absS > col_cut.unsqueeze(0)

        return row_ok & col_ok

    def enable_ba_mask(
        self,
        alpha: float,
        *,
        ties_keep: bool = True,
        rounding: str = "ceil",
        recompute_every: int = 1,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1].")
        if rounding not in {"ceil", "floor"}:
            raise ValueError("rounding must be 'ceil' or 'floor'")
        self._ba_mask_enabled = True
        self._ba_fixmask_sparse = False
        self._ba_mask_alpha = float(alpha)
        self._ba_mask_ties_keep = bool(ties_keep)
        self._ba_mask_rounding = rounding
        self._ba_mask_every = max(1, int(recompute_every))
        self._ba_mask_step = 0
        self._ba_last_refresh_step = -1
        self._ba_cached_mask.resize_(0, 0)

    def disable_ba_mask(self) -> None:
        self._ba_mask_enabled = False
        self._ba_fixmask_sparse = False
        self._ba_cached_mask.resize_(0, 0)
        self._ba_last_refresh_step = -1

    @torch.no_grad()
    def enable_ba_sparse_fix_mask(
        self,
        alpha: float,
        *,
        ties_keep: bool = True,
        rounding: str = "ceil",
        recompute_every: int = 1,
    ) -> None:
        self.enable_ba_mask(alpha, ties_keep=ties_keep, rounding=rounding, recompute_every=recompute_every)
        self._ba_fixmask_sparse = True

    def _forward_ba_sparse_fix_mask(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x2d = x.view(-1, x_shape[-1])
        device = x2d.device
        dtype = x2d.dtype
        out_f, in_f = self.out_features, self.in_features

        if self._need_refresh_mask((out_f, in_f)):
            with torch.no_grad():
                C_abs = (self.B @ self.A).abs()
                self._refresh_mask_from_C_abs(C_abs)

        if not self._ba_use_external_step:
            self._ba_mask_step += 1

        I, J = self._ba_cached_mask.nonzero(as_tuple=True)
        if I.numel() == 0:
            delta = x2d.new_zeros((x2d.size(0), out_f))
            return delta.view(*x_shape[:-1], out_f)

        BI = self.B[I, :]
        AJ = self.A[:, J].t()
        vals = (BI * AJ).sum(dim=1).to(dtype=dtype, device=device)

        idx = torch.stack([I.to(device), J.to(device)], dim=0)
        C_sparse = torch.sparse_coo_tensor(idx, vals, size=(out_f, in_f)).coalesce()

        y = torch.sparse.mm(C_sparse, x2d.t())
        delta2d = y.t()
        return delta2d.view(*x_shape[:-1], out_f)

    def notify_optimizer_step(self) -> None:
        """
        Call after each optimizer.step() for mask refresh cadence based on optimizer steps.
        """
        self._ba_use_external_step = True
        self._ba_mask_step += 1

        # Eager refresh so forward doesn't pay the cost.
        if self._ba_mask_enabled and (not self._ba_fixmask_sparse):
            shape = (self.out_features, self.in_features)
            if self._need_refresh_mask(shape):
                with torch.no_grad():
                    C_abs = (self.B @ self.A).abs()
                    self._refresh_mask_from_C_abs(C_abs)