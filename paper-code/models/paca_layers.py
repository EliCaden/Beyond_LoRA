# models/paca_layers.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# AMP decorators moved in newer torch; support both.
try:
    from torch.amp import custom_fwd, custom_bwd  # type: ignore
    _AMP_KW = {"device_type": "cuda"}
except Exception:
    from torch.cuda.amp import custom_fwd, custom_bwd  # type: ignore
    _AMP_KW = {}


def _prune_lastdim(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x.index_select(dim=-1, index=cols)


class _PaCALinearFn(torch.autograd.Function):
    """
    PaCA linear:
      y = x @ W^T + b

    where W is the *already patched* base weight containing:
      W[:, paca_cols] = scaling * paca_cols_weight

    Backward:
      - returns grad only for paca_cols_weight (and optional bias)
      - (optionally) returns grad_x if requested
    """

    @staticmethod
    @custom_fwd(**_AMP_KW)
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        paca_cols_weight: torch.Tensor,  # unscaled param (out, k)
        bias: torch.Tensor | None,
        paca_cols: torch.Tensor,         # (k,)
        save_full_x: bool,
        scaling: float,
    ) -> torch.Tensor:
        # Forward uses patched weight directly.
        y = F.linear(x, weight, bias)

        # IMPORTANT: do NOT save `weight` via save_for_backward.
        # Saving it triggers version-counter errors if weight is patched in-place elsewhere.
        ctx.weight_ref = weight
        ctx.cols_ref = paca_cols

        # Save either full x or pruned x for grad_paca computation.
        if save_full_x:
            ctx.save_for_backward(x)
            ctx._saved_is_pruned = False
        else:
            x_sel = _prune_lastdim(x, paca_cols)
            ctx.save_for_backward(x_sel)
            ctx._saved_is_pruned = True

        ctx.scaling = float(scaling)
        ctx.x_dtype = x.dtype
        ctx.paca_dtype = paca_cols_weight.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = (bias.dtype if bias is not None else None)
        return y

    @staticmethod
    @custom_bwd(**_AMP_KW)
    def backward(ctx, grad_out: torch.Tensor):
        (x_or_xsel,) = ctx.saved_tensors
        weight = ctx.weight_ref
        paca_cols = ctx.cols_ref

        grad_x = None
        grad_paca = None
        grad_bias = None

        # ---- grad_x = grad_out @ W ----
        if ctx.needs_input_grad[0]:
            go2d = grad_out.reshape(-1, grad_out.shape[-1])           # (N, out)
            gx2d = go2d.to(dtype=weight.dtype).matmul(weight)         # (N, in)
            grad_x = gx2d.reshape(*grad_out.shape[:-1], weight.shape[1])
            if grad_x.dtype != ctx.x_dtype:
                grad_x = grad_x.to(dtype=ctx.x_dtype)

        # ---- grad_paca ----
        # W[:, cols] = scaling * P  => dL/dP = scaling * dL/dW_cols
        if ctx.needs_input_grad[2]:
            if ctx._saved_is_pruned:
                x_sel = x_or_xsel
            else:
                x_sel = _prune_lastdim(x_or_xsel, paca_cols)

            go2d = grad_out.reshape(-1, grad_out.shape[-1])           # (N, out)
            xs2d = x_sel.reshape(-1, x_sel.shape[-1])                 # (N, k)

            # fp32 accumulation for stability, then cast
            grad_paca_f32 = go2d.to(torch.float32).transpose(0, 1).matmul(xs2d.to(torch.float32))  # (out, k)
            grad_paca_f32.mul_(ctx.scaling)
            grad_paca = grad_paca_f32.to(dtype=ctx.paca_dtype)

        # ---- grad_bias ----
        if ctx.has_bias and ctx.needs_input_grad[3]:
            dims = list(range(grad_out.dim() - 1))
            grad_bias = grad_out.sum(dims)
            if ctx.bias_dtype is not None and grad_bias.dtype != ctx.bias_dtype:
                grad_bias = grad_bias.to(dtype=ctx.bias_dtype)

        # (x, weight, paca_cols_weight, bias, paca_cols, save_full_x, scaling)
        return grad_x, None, grad_paca, grad_bias, None, None, None


class PaCALinearAdapter(nn.Module):
    """
    PaCA adapter for nn.Linear.

    Maintains learnable P = paca_cols_weight of shape (out_features, k_cols)
    and patches base.weight[:, paca_cols] = scaling * P.

    Correctness:
      - During training, patching must be enabled (patching=True and patch_mode != 'never'),
        otherwise P updates won't affect forward.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: int = 16,
        k_per_row: int | None = None,
        seed: int = 0,
        init: str = "from_base",
        train_bias: bool = False,
        patching: bool = True,
        save_full_x_for_grad: bool = False,
        patch_mode: str = "dirty",          # "dirty" | "always" | "never"
        patch_on_eval_if_dirty: bool = True,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.dim() != 2:
            raise TypeError("PaCALinearAdapter expects nn.Linear with 2D weight.")

        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        k_cols = int(r) if (k_per_row is None) else int(k_per_row)
        k_cols = max(1, min(k_cols, self.in_features))
        self.k_cols = int(k_cols)

        self.alpha = int(alpha) if int(alpha) != 0 else int(2 * self.k_cols)
        self.scaling = float(self.alpha) / float(self.k_cols) if self.k_cols > 0 else 1.0

        # Persist alpha for state_dict correctness; keep strict-load backward compatible.
        self.register_buffer("paca_alpha", torch.tensor(int(self.alpha), dtype=torch.int64), persistent=True)

        self.patching = bool(patching)
        self.save_full_x_for_grad = bool(save_full_x_for_grad)

        self.patch_mode = str(patch_mode).lower().strip()
        if self.patch_mode not in {"dirty", "always", "never"}:
            raise ValueError("patch_mode must be one of {'dirty','always','never'}")
        self.patch_on_eval_if_dirty = bool(patch_on_eval_if_dirty)

        # freeze base; optionally train bias
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

        # choose fixed columns (stored persistently)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        cols_cpu = torch.randperm(self.in_features, generator=g)[: self.k_cols].long()
        cols = cols_cpu.to(device=self.base.weight.device)
        self.register_buffer("paca_cols", cols, persistent=True)

        # init paca param from base columns (or zero/normal)
        base_cols = self.base.weight.detach().index_select(dim=1, index=self.paca_cols)
        init_l = str(init).strip().lower()
        if init_l == "from_base":
            p = base_cols / float(self.scaling)
        elif init_l == "zero":
            p = torch.zeros_like(base_cols)
        elif init_l == "normal":
            p = torch.empty_like(base_cols)
            nn.init.normal_(p, std=0.02)
        else:
            raise ValueError("init must be 'from_base', 'zero', or 'normal'")

        self.paca_cols_weight = nn.Parameter(p.to(device=self.base.weight.device, dtype=self.base.weight.dtype))

        # reusable buffer for patched (scaled) columns
        self.register_buffer(
            "_paca_cols_buf",
            torch.empty_like(self.paca_cols_weight, device=self.base.weight.device, dtype=self.base.weight.dtype),
            persistent=False,
        )

        # dirty tracking
        self._needs_patch: bool = True
        self._last_paca_version: int = -1
        self._last_cols_version: int = -1

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, k={self.k_cols}, alpha={self.alpha}, "
            f"scaling={self.scaling:.6g}, patch_mode={self.patch_mode}, patching={self.patching}"
        )

    def _sync_alpha_scaling_from_buffer(self) -> None:
        try:
            a = int(self.paca_alpha.detach().item())
        except Exception:
            a = int(self.alpha)
        if a <= 0:
            a = max(1, int(self.alpha))
        self.alpha = a
        self.scaling = float(self.alpha) / float(self.k_cols) if self.k_cols > 0 else 1.0
        try:
            self.paca_alpha.fill_(int(self.alpha))
        except Exception:
            pass

    def _infer_scaling_from_loaded_weights(self) -> None:
        # Best-effort backward compat if old checkpoints lacked paca_alpha.
        # If base weight columns match scaling * paca_cols_weight, infer scaling = <Wcols,P>/<P,P>.
        try:
            cols = self.paca_cols.to(device=self.base.weight.device)
            Wcols = self.base.weight.detach().index_select(dim=1, index=cols).to(torch.float32)
            P = self.paca_cols_weight.detach().to(torch.float32)
            denom = (P * P).sum()
            if float(denom.item()) <= 1e-12:
                return
            scale = (Wcols * P).sum() / denom
            s = float(scale.item())
            if not (s > 0.0):
                return
            self.scaling = s
            self.alpha = int(round(self.scaling * float(self.k_cols)))
            self.paca_alpha.fill_(int(self.alpha))
        except Exception:
            return

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

        key = prefix + "paca_alpha"
        if key in state_dict:
            self._sync_alpha_scaling_from_buffer()
        else:
            # Backward compatible strict-load: don't error on missing paca_alpha.
            if key in missing_keys:
                missing_keys.remove(key)
            # Try to infer scaling once (helps old checkpoints even if user constructed with wrong alpha).
            self._infer_scaling_from_loaded_weights()

        # After load, patched base weights might already match; still mark dirty so next forward can re-patch if needed.
        self._needs_patch = True

    def mark_dirty(self) -> None:
        self._needs_patch = True

    def _is_dirty(self) -> bool:
        if not self.patching or self.patch_mode == "never":
            return False
        if self._needs_patch:
            return True
        pv = int(getattr(self.paca_cols_weight, "_version", 0))
        cv = int(getattr(self.paca_cols, "_version", 0))
        return (pv != self._last_paca_version) or (cv != self._last_cols_version)

    @torch.no_grad()
    def paca_update(self) -> None:
        """
        Patch base.weight columns using current paca_cols_weight.
        """
        if not self.patching or self.patch_mode == "never":
            return

        self._paca_cols_buf.copy_(self.paca_cols_weight)
        self._paca_cols_buf.mul_(float(self.scaling))
        self.base.weight.data.index_copy_(1, self.paca_cols, self._paca_cols_buf)

        self._needs_patch = False
        self._last_paca_version = int(getattr(self.paca_cols_weight, "_version", 0))
        self._last_cols_version = int(getattr(self.paca_cols, "_version", 0))

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """
        Optional hook: call after optimizer.step() to patch immediately (avoids paying in forward).
        """
        if self.patch_mode in {"dirty", "always"} and self._is_dirty():
            self.paca_update()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- correctness guards ----
        if self.training:
            if not self.patching:
                raise RuntimeError(
                    "PaCALinearAdapter is in training mode with patching=False. "
                    "paca_cols_weight will update but will NOT affect forward. "
                    "Set patching=True or call merge_into_base() and freeze this module."
                )
            if self.patch_mode == "never":
                raise RuntimeError(
                    "PaCALinearAdapter is in training mode with patch_mode='never'. "
                    "This disables patching, so training is incorrect. Use patch_mode='dirty' or 'always'."
                )

        # ---- patch if needed ----
        if self.patching and self.patch_mode != "never":
            if self.patch_mode == "always":
                self.paca_update()
            else:
                dirty = self._is_dirty()
                if dirty and (self.training or self.patch_on_eval_if_dirty):
                    self.paca_update()

        return _PaCALinearFn.apply(
            x,
            self.base.weight,
            self.paca_cols_weight,
            self.base.bias,
            self.paca_cols,
            bool(self.save_full_x_for_grad),
            float(self.scaling),
        )

    @torch.no_grad()
    def merge_into_base(self, *, zero_adapter: bool = False, disable_patching: bool = True) -> None:
        """
        Ensure base weight reflects current adapter params.
        If disable_patching=True, we also freeze adapter params to prevent “train-but-no-effect”.
        """
        self.paca_update()

        if zero_adapter:
            self.paca_cols_weight.zero_()
            self.mark_dirty()

        if disable_patching:
            self.patching = False
            self.paca_cols_weight.requires_grad_(False)


@torch.no_grad()
def cpaca_merge_and_replace_(
    model: nn.Module,
    *,
    seed: int,
    init: str = "from_base",
    seed_stride: int = 1,
) -> int:
    """
    Merge existing PaCA adapters into their base weights, then replace them with fresh adapters.
    NOTE: safer to collect targets first, then replace, to avoid mutating while iterating.
    """
    targets: list[tuple[nn.Module, str, PaCALinearAdapter]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, PaCALinearAdapter):
                targets.append((parent, name, child))

    replaced = 0
    for parent, name, child in targets:
        child.merge_into_base(zero_adapter=False, disable_patching=True)

        new_seed = int(seed + replaced * int(seed_stride))
        new_adapter = PaCALinearAdapter(
            child.base,
            r=child.k_cols,
            alpha=child.alpha,
            k_per_row=child.k_cols,
            seed=new_seed,
            init=init,
            train_bias=(child.base.bias is not None and bool(child.base.bias.requires_grad)),
            patching=True,
            save_full_x_for_grad=getattr(child, "save_full_x_for_grad", False),
            patch_mode=getattr(child, "patch_mode", "dirty"),
            patch_on_eval_if_dirty=getattr(child, "patch_on_eval_if_dirty", True),
        )
        new_adapter.to(child.base.weight.device)
        setattr(parent, name, new_adapter)
        replaced += 1

    return replaced