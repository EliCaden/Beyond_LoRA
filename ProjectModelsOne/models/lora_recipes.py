# models/lora_recipes.py
from __future__ import annotations

from typing import Iterable, List, Union
import torch
import torch.nn as nn

try:
    from models.lora_layers import LoRALinearAdapter
except ImportError:
    from .lora_layers import LoRALinearAdapter


def _norm_recipe(name: str) -> str:
    t = name.strip().lower().replace("-", "_")
    if t in {"rac", "rac_a"}:
        return "rac_a"
    if t in {"racb", "rac_b"}:
        return "rac_b"
    if t in {"c3la", "c^3la", "c3-lora"}:
        return "c3la"
    return t


def find_lora_adapters(root: Union[nn.Module, Iterable[nn.Module]]) -> List[LoRALinearAdapter]:
    """
    Speedup: if the wrapper exposes lora_modules(), prefer that (cached)
    instead of scanning model.modules() every time.
    """
    if isinstance(root, LoRALinearAdapter):
        return [root]

    if isinstance(root, nn.Module):
        lm = getattr(root, "lora_modules", None)
        if callable(lm):
            out = lm()
            return [m for m in out if isinstance(m, LoRALinearAdapter)]
        return [m for m in root.modules() if isinstance(m, LoRALinearAdapter)]

    out: List[LoRALinearAdapter] = []
    for item in root:
        if isinstance(item, LoRALinearAdapter):
            out.append(item)
        elif isinstance(item, nn.Module):
            lm = getattr(item, "lora_modules", None)
            if callable(lm):
                out.extend([m for m in lm() if isinstance(m, LoRALinearAdapter)])
            else:
                out.extend([m for m in item.modules() if isinstance(m, LoRALinearAdapter)])
    return out


@torch.no_grad()
def merge_all_lora(root: Union[nn.Module, Iterable[nn.Module]]) -> int:
    n = 0
    for ada in find_lora_adapters(root):
        ada.merge_into_base()
        n += 1
    return n


@torch.no_grad()
def strip_all_lora(root: Union[nn.Module, Iterable[nn.Module]]) -> int:
    def _strip(module: nn.Module) -> int:
        cnt = 0
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinearAdapter):
                setattr(module, name, child.base)
                cnt += 1
            else:
                cnt += _strip(child)
        return cnt

    if isinstance(root, nn.Module):
        return _strip(root)

    total = 0
    for m in root:
        if isinstance(m, nn.Module):
            total += _strip(m)
    return total


@torch.no_grad()
def merge_and_strip_all_lora(root: Union[nn.Module, Iterable[nn.Module]]) -> int:
    n = merge_all_lora(root)
    strip_all_lora(root)
    return n


def apply_recipe(
    root: Union[nn.Module, Iterable[nn.Module]],
    recipe: str,
    *,
    offset_step: int | None = None,
    shuffle_seed: int | None = 0,
    mask_alpha: float | None = None,
    mask_ties_keep: bool = True,
    mask_rounding: str = "ceil",
    mask_recompute_every: int = 1,
) -> None:
    recipe = _norm_recipe(recipe)

    # Sparse wrappers first
    if recipe in {"sparse_cheap", "sparse_shuffle", "sparse_c3la"}:
        base_map = {
            "sparse_cheap": "cheap",
            "sparse_shuffle": "shuffle",
            "sparse_c3la": "c3la",
        }
        base_rec = base_map[recipe]
        apply_recipe(root, base_rec, offset_step=offset_step, shuffle_seed=shuffle_seed)
        for a in find_lora_adapters(root):
            a.enable_sparse_A_from_current()
            a._init_recipe = recipe
        return

    g = None
    if shuffle_seed is not None:
        g = torch.Generator()
        g.manual_seed(int(shuffle_seed))

    for ada in find_lora_adapters(root):
        if recipe == "base":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
            ada._init_recipe = "base"

        elif recipe == "asym_a":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._init_recipe = "asym_a"

        elif recipe == "asym_b":
            ada.set_A_zero()
            ada.set_B_gaussian(0.02)
            ada.freeze_A(False)
            ada.freeze_B(True)
            ada._init_recipe = "asym_b"

        elif recipe == "cheap":
            ada.set_A_identity_window(offset=0)
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._cheap_offset = 0
            ada._init_recipe = "cheap"

        elif recipe == "random_cheap":
            in_feats = ada.in_features
            r = ada.r
            max_start = max(0, in_feats - r)
            if max_start == 0:
                off = 0
            else:
                if isinstance(g, torch.Generator):
                    off = int(torch.randint(0, max_start + 1, (1,), generator=g).item())
                else:
                    off = int(torch.randint(0, max_start + 1, (1,)).item())
            ada.set_A_identity_window(offset=off)
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._cheap_offset = off
            ada._init_recipe = "random_cheap"

        elif recipe == "shuffle":
            ada.build_shuffle_stream()
            ada.shuffle_stream_once(g if isinstance(g, torch.Generator) else None)
            ada.set_A_next_shuffle_window()
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._init_recipe = "shuffle"

        elif recipe == "c3la":
            ada.build_shuffle_stream()
            ada.set_A_next_shuffle_window()
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._init_recipe = "c3la"

        elif recipe == "cola":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
            ada._init_recipe = "cola"

        elif recipe == "rac_a":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada._init_recipe = "rac_a"

        elif recipe == "rac_b":
            ada.set_A_zero()
            ada.set_B_gaussian(0.02)
            ada.freeze_A(False)
            ada.freeze_B(True)
            ada._init_recipe = "rac_b"

        elif recipe == "ba_sparse_lora":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
            alpha = 0.5 if (mask_alpha is None) else float(mask_alpha)
            ada.enable_ba_mask(
                alpha=alpha,
                ties_keep=mask_ties_keep,
                rounding=mask_rounding,
                recompute_every=mask_recompute_every,
            )
            ada._init_recipe = "ba_sparse_lora"

        elif recipe == "ba_sparse_final":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
            alpha = 0.5 if (mask_alpha is None) else float(mask_alpha)
            ada._final_ba_mask = {
                "alpha": float(alpha),
                "ties_keep": bool(mask_ties_keep),
                "rounding": str(mask_rounding),
            }
            ada._init_recipe = "ba_sparse_final"

        elif recipe == "ba_sparse_fix_mask":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
            alpha = 0.5 if (mask_alpha is None) else float(mask_alpha)
            ada.enable_ba_sparse_fix_mask(
                alpha=alpha,
                ties_keep=mask_ties_keep,
                rounding=mask_rounding,
                recompute_every=mask_recompute_every,
            )
            ada._init_recipe = "ba_sparse_fix_mask"

        else:
            raise ValueError(f"Unknown recipe: {recipe}")


@torch.no_grad()
def chain_step(root: Union[nn.Module, Iterable[nn.Module]], recipe: str) -> None:
    recipe = _norm_recipe(recipe)
    allowed = {
        "cola", "rac_a", "rac_b", "c3la", "shuffle",
        "sparse_shuffle", "sparse_c3la",
    }
    if recipe not in allowed:
        raise ValueError(f"Unknown chain recipe: {recipe}. Allowed: {sorted(allowed)}")

    for ada in find_lora_adapters(root):
        init_rec = getattr(ada, "_init_recipe", None)
        if init_rec != recipe:
            raise ValueError(
                f"Chain recipe '{recipe}' requires prior initialization with the SAME recipe; "
                f"adapter was initialized with '{init_rec}'."
            )

    merge_all_lora(root)

    if recipe in {"sparse_shuffle", "sparse_c3la"}:
        for ada in find_lora_adapters(root):
            ada.set_A_next_shuffle_window()
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
            ada.enable_sparse_A_from_current()
        return

    for ada in find_lora_adapters(root):
        if recipe == "cola":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(False)
            ada.freeze_B(False)
        elif recipe == "rac_a":
            ada.set_A_gaussian(0.02)
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
        elif recipe == "rac_b":
            ada.set_A_zero()
            ada.set_B_gaussian(0.02)
            ada.freeze_A(False)
            ada.freeze_B(True)
        elif recipe in {"shuffle", "c3la"}:
            ada.set_A_next_shuffle_window()
            ada.set_B_zero()
            ada.freeze_A(True)
            ada.freeze_B(False)
        else:
            raise ValueError(f"Unknown chain recipe: {recipe}")


@torch.no_grad()
def finalize_ba_sparse_all(
    root: Union[nn.Module, Iterable[nn.Module]],
    *,
    strip: bool = True,
    default_alpha: float = 0.5,
    default_ties_keep: bool = True,
    default_rounding: str = "ceil",
) -> int:
    adapters = find_lora_adapters(root)
    n = 0
    for ada in adapters:
        spec = getattr(ada, "_final_ba_mask", None)
        if spec is None:
            continue
        alpha = float(spec.get("alpha", default_alpha))
        ties = bool(spec.get("ties_keep", default_ties_keep))
        rounding = str(spec.get("rounding", default_rounding))
        ada.enable_ba_mask(alpha=alpha, ties_keep=ties, rounding=rounding, recompute_every=1)
        n += 1

    if n > 0:
        merge_all_lora(root)
        if strip:
            strip_all_lora(root)
    return n


@torch.no_grad()
def notify_optimizer_step_all(root: Union[nn.Module, Iterable[nn.Module]]) -> None:
    """
    Speed/behavior hook: call after optimizer.step() so adapters can refresh
    BA masks on optimizer-step cadence (instead of per-forward), when enabled.
    """
    for ada in find_lora_adapters(root):
        fn = getattr(ada, "notify_optimizer_step", None)
        if callable(fn):
            fn()
