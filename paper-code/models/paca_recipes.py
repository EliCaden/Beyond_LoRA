# models/paca_recipes.py
from __future__ import annotations

from typing import Iterable, List, Union, Optional
import torch
import torch.nn as nn

try:
    from models.paca_layers import PaCALinearAdapter
except ImportError:
    from .paca_layers import PaCALinearAdapter


def _norm_recipe(name: str) -> str:
    t = name.strip().lower().replace("-", "_")
    if t in {"c_paca", "chain_paca"}:
        return "cpaca"
    if t in {"d_paca", "det_paca"}:
        return "dpaca"
    if t in {"dc_paca", "det_chain_paca"}:
        return "dcpaca"
    return t


def is_paca_recipe(name: Optional[str]) -> bool:
    if not name:
        return False
    t = _norm_recipe(name)
    return t in {"dpaca", "cpaca", "dcpaca"}


def find_paca_adapters(root: Union[nn.Module, Iterable[nn.Module]]) -> List[PaCALinearAdapter]:
    """
    Speedup: if the wrapper exposes paca_modules(), prefer that (cached)
    instead of scanning model.modules() every time.
    """
    if isinstance(root, PaCALinearAdapter):
        return [root]

    if isinstance(root, nn.Module):
        pm = getattr(root, "paca_modules", None)
        if callable(pm):
            out = pm()
            return [m for m in out if isinstance(m, PaCALinearAdapter)]
        return [m for m in root.modules() if isinstance(m, PaCALinearAdapter)]

    out: List[PaCALinearAdapter] = []
    for item in root:
        if isinstance(item, PaCALinearAdapter):
            out.append(item)
        elif isinstance(item, nn.Module):
            pm = getattr(item, "paca_modules", None)
            if callable(pm):
                out.extend([m for m in pm() if isinstance(m, PaCALinearAdapter)])
            else:
                out.extend([m for m in item.modules() if isinstance(m, PaCALinearAdapter)])
    return out


def has_paca_adapters(root: Union[nn.Module, Iterable[nn.Module]]) -> bool:
    return len(find_paca_adapters(root)) > 0


@torch.no_grad()
def merge_all_paca(root: Union[nn.Module, Iterable[nn.Module]], *, disable_patching: bool = False) -> int:
    n = 0
    for ada in find_paca_adapters(root):
        ada.merge_into_base(disable_patching=disable_patching)
        n += 1
    return n


@torch.no_grad()
def _set_cols(ada: PaCALinearAdapter, cols_1d: torch.Tensor) -> None:
    cols = cols_1d.to(dtype=torch.long, device=ada.paca_cols.device)
    if cols.numel() != ada.k_cols:
        raise ValueError(f"PaCA: expected {ada.k_cols} cols, got {cols.numel()}")
    ada.paca_cols.copy_(cols)

    md = getattr(ada, "mark_dirty", None)
    if callable(md):
        md()


@torch.no_grad()
def _reinit_cols_weight(ada: PaCALinearAdapter, *, init_mode: str, init_seed: int) -> None:
    init_mode = init_mode.strip().lower()

    W = ada.base.weight.data
    cols = ada.paca_cols
    dev = ada.paca_cols_weight.device
    dt = ada.paca_cols_weight.dtype

    if init_mode == "from_base":
        base_cols = W.index_select(dim=1, index=cols.to(device=W.device)).to(device=dev, dtype=dt)
        ada.paca_cols_weight.data.copy_(base_cols / float(ada.scaling))

    elif init_mode == "zero":
        ada.paca_cols_weight.zero_()

    elif init_mode == "normal":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(init_seed))
        tmp = torch.empty(ada.paca_cols_weight.shape, device="cpu", dtype=torch.float32)
        tmp.normal_(mean=0.0, std=0.02, generator=g)
        ada.paca_cols_weight.data.copy_(tmp.to(device=dev, dtype=dt))

    else:
        raise ValueError("PaCA: init_mode must be 'from_base', 'zero', or 'normal'.")

    # ensure adapter is trainable again (merge_into_base(disable_patching=True) freezes it)
    ada.paca_cols_weight.requires_grad_(True)

    md = getattr(ada, "mark_dirty", None)
    if callable(md):
        md()


def _dcpaca_windows(in_feats: int, k: int) -> List[int]:
    max_start = max(0, in_feats - k)
    return list(range(0, max_start + 1, k)) or [0]


@torch.no_grad()
def apply_paca_recipe(
    root: Union[nn.Module, Iterable[nn.Module]],
    recipe: str,
    *,
    seed: int = 0,
    offset_step: int = 0,
    init: str = "from_base",
) -> None:
    recipe = _norm_recipe(recipe)
    if recipe not in {"dpaca", "cpaca", "dcpaca"}:
        raise ValueError(f"Unknown PaCA recipe: {recipe}")

    adapters = find_paca_adapters(root)
    for layer_i, ada in enumerate(adapters):
        ada.patching = True
        ada.paca_cols_weight.requires_grad_(True)

        k = int(ada.k_cols)
        in_feats = int(ada.in_features)

        ada._init_recipe = recipe
        ada._paca_seed = int(seed)
        ada._paca_init_mode = str(init)
        ada._paca_layer_i = int(layer_i)

        if recipe == "dpaca":
            cols = torch.arange(0, k, dtype=torch.long, device="cpu")
            _set_cols(ada, cols)
            ada._paca_step = 0
            ada._paca_window_ptr = 0
            _reinit_cols_weight(ada, init_mode=ada._paca_init_mode, init_seed=int(seed) + 1000003 * layer_i)

        elif recipe == "cpaca":
            step = int(offset_step)
            ada._paca_step = step
            ada._paca_window_ptr = 0
            layer_seed = int(seed) + 1000003 * layer_i
            g = torch.Generator(device="cpu")
            g.manual_seed(layer_seed + step)
            cols = torch.randperm(in_feats, generator=g)[:k].long()
            _set_cols(ada, cols)
            _reinit_cols_weight(ada, init_mode=ada._paca_init_mode, init_seed=layer_seed + step)

        elif recipe == "dcpaca":
            windows = _dcpaca_windows(in_feats, k)
            ptr = int(offset_step) % len(windows)
            ada._paca_window_ptr = ptr
            ada._paca_step = 0
            off = windows[ptr]
            cols = torch.arange(off, off + k, dtype=torch.long, device="cpu")
            _set_cols(ada, cols)
            _reinit_cols_weight(ada, init_mode=ada._paca_init_mode, init_seed=int(seed) + 1000003 * layer_i + ptr)


@torch.no_grad()
def paca_chain_step(root: Union[nn.Module, Iterable[nn.Module]], recipe: str) -> None:
    recipe = _norm_recipe(recipe)
    allowed = {"cpaca", "dcpaca"}
    if recipe not in allowed:
        raise ValueError(f"Unknown PaCA chain recipe: {recipe}. Allowed: {sorted(allowed)}")

    adapters = find_paca_adapters(root)
    for ada in adapters:
        ada.patching = True
        ada.paca_cols_weight.requires_grad_(True)
        init_rec = getattr(ada, "_init_recipe", None)
        if init_rec != recipe:
            raise ValueError(
                f"PaCA chain recipe '{recipe}' requires prior init with SAME recipe; "
                f"adapter was initialized with '{init_rec}'."
            )

    # Merge current effect into base weights (keeps adapters active for continued training)
    merge_all_paca(root, disable_patching=False)

    for ada in adapters:
        k = int(ada.k_cols)
        in_feats = int(ada.in_features)

        base_seed = int(getattr(ada, "_paca_seed", 0))
        init_mode = str(getattr(ada, "_paca_init_mode", "from_base"))
        layer_i = int(getattr(ada, "_paca_layer_i", 0))
        layer_seed = base_seed + 1000003 * layer_i

        if recipe == "cpaca":
            step = int(getattr(ada, "_paca_step", 0)) + 1
            ada._paca_step = step
            g = torch.Generator(device="cpu")
            g.manual_seed(layer_seed + step)
            cols = torch.randperm(in_feats, generator=g)[:k].long()
            _set_cols(ada, cols)
            _reinit_cols_weight(ada, init_mode=init_mode, init_seed=layer_seed + step)

        elif recipe == "dcpaca":
            windows = _dcpaca_windows(in_feats, k)
            ptr = int(getattr(ada, "_paca_window_ptr", 0))
            ptr = (ptr + 1) % len(windows)
            ada._paca_window_ptr = ptr
            off = windows[ptr]
            cols = torch.arange(off, off + k, dtype=torch.long, device="cpu")
            _set_cols(ada, cols)
            _reinit_cols_weight(ada, init_mode=init_mode, init_seed=layer_seed + ptr)


@torch.no_grad()
def after_optimizer_step_all(root: Union[nn.Module, Iterable[nn.Module]]) -> None:
    """
    Speed/correctness hook: call after optimizer.step() to patch base weights immediately.
    (Still safe if you don't call; next forward will patch if dirty.)
    """
    for ada in find_paca_adapters(root):
        fn = getattr(ada, "after_optimizer_step", None)
        if callable(fn):
            fn()