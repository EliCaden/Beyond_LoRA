# utils/injectors.py
from __future__ import annotations

import os
from typing import List, Sequence, Type, Optional

import torch
import torch.nn as nn


# -----------------------------
# Shared helpers
# -----------------------------

def _inject_debug_enabled() -> bool:
    # Keep simple; env reads are cheap and this is not per-step.
    return os.environ.get("INJECT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def mark_adapters_dirty(model: nn.Module, *, kind: str) -> None:
    """
    Best-effort adapter invalidation hooks.
    IMPORTANT: do NOT swallow exceptions if the hook exists but is broken.
    """
    if kind == "lora":
        fn = getattr(model, "mark_lora_dirty", None)
        if callable(fn):
            fn()
            return

    if kind == "paca":
        fn = getattr(model, "mark_paca_dirty", None)
        if callable(fn):
            fn()
            return

    fn = getattr(model, "mark_adapters_dirty", None)
    if callable(fn):
        fn()


def _resolve_maybe_callable_module(x) -> Optional[nn.Module]:
    if isinstance(x, nn.Module):
        return x
    if callable(x):
        try:
            y = x()
        except Exception:
            return None
        return y if isinstance(y, nn.Module) else None
    return None


def unfreeze_head_modules(model: nn.Module) -> int:
    """
    Robust head unfreeze without accidentally catching attention proj layers named 'proj' / 'dense'.
    Tries common top-level head attributes first; falls back to name substring.
    """
    net = getattr(model, "net", model)
    candidates: List[nn.Module] = []

    for attr in ("classifier", "head", "lm_head"):
        m = _resolve_maybe_callable_module(getattr(net, attr, None))
        if m is not None:
            candidates.append(m)

    if candidates:
        count = 0
        for m in candidates:
            for p in m.parameters():
                p.requires_grad = True
                count += 1
        return count

    # Last resort: parameter-name heuristic
    count = 0
    for n, p in model.named_parameters():
        if ("classifier" in n) or (".head" in n) or ("lm_head" in n):
            p.requires_grad = True
            count += 1
    return count


def _wrap_linear_lora(
    parent: nn.Module,
    attr: str,
    AdapterCls: Type[nn.Module],
    *,
    r: int,
    alpha: int,
) -> int:
    m = getattr(parent, attr, None)
    if isinstance(m, nn.Linear):
        setattr(parent, attr, AdapterCls(m, r=int(r), alpha=int(alpha)))
        return 1
    return 0


def _wrap_linear_paca(
    parent: nn.Module,
    attr: str,
    AdapterCls: Type[nn.Module],
    *,
    r: int,
    alpha: int,
    seed: int,
    k_per_row: int | None,
) -> int:
    m = getattr(parent, attr, None)
    if isinstance(m, nn.Linear):
        setattr(parent, attr, AdapterCls(m, r=int(r), alpha=int(alpha), seed=int(seed), k_per_row=k_per_row))
        return 1
    return 0


# -----------------------------
# Model structure iterators
# -----------------------------

def _iter_roberta_layers(model: nn.Module):
    net = getattr(model, "net", model)
    layers = net.roberta.encoder.layer  # may raise AttributeError
    for layer in layers:
        yield layer


def _iter_hf_vit_layers(model: nn.Module):
    net = getattr(model, "net", model)
    vit = getattr(net, "vit", None) or getattr(net, "vision_model", None) or getattr(net, "model", None)
    if vit is None:
        vit = getattr(model, "vit", None) or getattr(model, "vision_model", None)
    if vit is None:
        raise AttributeError("No vit/vision_model found")

    enc = getattr(vit, "encoder", None)
    if enc is None or not hasattr(enc, "layer"):
        raise AttributeError("No vit.encoder.layer found")

    for layer in enc.layer:
        yield layer


def _iter_timm_vit_blocks(model: nn.Module):
    net = getattr(model, "net", model)
    vit = getattr(net, "vit", None) or getattr(model, "vit", None) or net
    blocks = getattr(vit, "blocks", None) or getattr(net, "blocks", None) or getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("No timm-like blocks found")
    for b in blocks:
        yield b


# -----------------------------
# Architecture probes (fail fast before mutating)
# -----------------------------

def _probe_roberta(model: nn.Module) -> None:
    layer = next(_iter_roberta_layers(model))  # may raise StopIteration/AttributeError
    sa = layer.attention.self
    _ = sa.query
    _ = sa.key
    _ = sa.value
    ao = layer.attention.output
    _ = ao.dense
    f1 = layer.intermediate
    _ = f1.dense
    f2 = layer.output
    _ = f2.dense


def _probe_hf_vit(model: nn.Module) -> None:
    layer = next(_iter_hf_vit_layers(model))
    attn = layer.attention
    sa = attn.attention
    _ = sa.query
    _ = sa.key
    _ = sa.value
    ao = attn.output
    _ = ao.dense
    f1 = layer.intermediate
    _ = f1.dense
    f2 = layer.output
    _ = f2.dense


def _probe_timm_vit(model: nn.Module) -> None:
    block = next(_iter_timm_vit_blocks(model))
    attn = getattr(block, "attn", None)
    if attn is None:
        raise AttributeError("timm-like block has no .attn")
    # common fields; some timm models may omit qkv if split projections
    _ = getattr(attn, "qkv", None) or getattr(attn, "q", None) or getattr(attn, "query", None)


# -----------------------------
# timm fused-qkv selective wrapper (LoRA only)
# -----------------------------

class _SelectiveQKVLoRA(nn.Module):
    """
    Wrap a fused timm attn.qkv (Linear: dim -> 3*dim) and apply LoRA deltas
    only to selected thirds (q/k/v) while keeping the fused base GEMM.
    """

    def __init__(self, qkv: nn.Linear, AdapterCls: Type[nn.Module], *, r: int, alpha: int, targets: Sequence[str]):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("SelectiveQKVLoRA expects an nn.Linear qkv.")
        if qkv.out_features % 3 != 0:
            raise ValueError(f"qkv.out_features={qkv.out_features} not divisible by 3.")

        self.base = qkv
        self.in_features = int(qkv.in_features)
        self.out_features = int(qkv.out_features)
        self.dim = self.out_features // 3

        want = {t for t in targets if t in {"q", "k", "v"}}
        self.want_q = "q" in want
        self.want_k = "k" in want
        self.want_v = "v" in want

        dev = qkv.weight.device
        dt = qkv.weight.dtype

        def _make_stub(out_dim: int) -> nn.Linear:
            stub = nn.Linear(self.in_features, out_dim, bias=False, device=dev, dtype=dt)
            with torch.no_grad():
                stub.weight.zero_()
            return stub

        self.lora_q = AdapterCls(_make_stub(self.dim), r=int(r), alpha=int(alpha)) if self.want_q else None
        self.lora_k = AdapterCls(_make_stub(self.dim), r=int(r), alpha=int(alpha)) if self.want_k else None
        self.lora_v = AdapterCls(_make_stub(self.dim), r=int(r), alpha=int(alpha)) if self.want_v else None

        for m in (self.lora_q, self.lora_k, self.lora_v):
            if m is not None and not hasattr(m, "forward_delta"):
                raise TypeError("SelectiveQKVLoRA requires AdapterCls to implement forward_delta(x).")

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_q is None and self.lora_k is None and self.lora_v is None:
            return y

        # Avoid in-place edits on a tensor that autograd may reuse.
        y = y.clone()
        d = self.dim
        if self.lora_q is not None:
            y[..., 0 * d : 1 * d] = y[..., 0 * d : 1 * d] + self.lora_q.forward_delta(x)
        if self.lora_k is not None:
            y[..., 1 * d : 2 * d] = y[..., 1 * d : 2 * d] + self.lora_k.forward_delta(x)
        if self.lora_v is not None:
            y[..., 2 * d : 3 * d] = y[..., 2 * d : 3 * d] + self.lora_v.forward_delta(x)
        return y

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """
        Merge selected deltas into fused qkv weight slices (best-effort).
        Prefer calling adapter.merge_into_base if present; otherwise use (B@A) heuristic.
        """
        d = self.dim

        def _merge_one(adapter: nn.Module | None, slc: slice):
            if adapter is None:
                return

            # Prefer adapter API if available
            fn = getattr(adapter, "merge_into_base", None)
            if callable(fn):
                # If the adapter supports merging into its own base stub, do it,
                # then copy stub weight into the fused slice.
                fn()
                if hasattr(adapter, "base") and isinstance(adapter.base, nn.Linear):
                    self.base.weight.data[slc, :].add_(adapter.base.weight.data.to(self.base.weight.dtype))
                    adapter.base.weight.zero_()
                return

            # Fallback: (B @ A) pattern (assumes LoRA-style fields)
            if not (hasattr(adapter, "A") and hasattr(adapter, "B") and hasattr(adapter, "scaling")):
                return
            deltaW = adapter.scaling * (adapter.B @ adapter.A)
            self.base.weight.data[slc, :].add_(deltaW.to(self.base.weight.dtype))
            nn.init.zeros_(adapter.A)
            nn.init.zeros_(adapter.B)

        _merge_one(self.lora_q, slice(0 * d, 1 * d))
        _merge_one(self.lora_k, slice(1 * d, 2 * d))
        _merge_one(self.lora_v, slice(2 * d, 3 * d))


# -----------------------------
# LoRA injectors (fallbacks)
# -----------------------------

def inject_lora_roberta(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
    targets: Sequence[str],
) -> int:
    _probe_roberta(model)

    replaced = 0
    for layer in _iter_roberta_layers(model):
        sa = layer.attention.self
        ao = layer.attention.output
        f1 = layer.intermediate
        f2 = layer.output

        if "q" in targets:
            replaced += _wrap_linear_lora(sa, "query", AdapterCls, r=r, alpha=alpha)
        if "k" in targets:
            replaced += _wrap_linear_lora(sa, "key", AdapterCls, r=r, alpha=alpha)
        if "v" in targets:
            replaced += _wrap_linear_lora(sa, "value", AdapterCls, r=r, alpha=alpha)
        if "o" in targets:
            replaced += _wrap_linear_lora(ao, "dense", AdapterCls, r=r, alpha=alpha)
        if "ffn1" in targets:
            replaced += _wrap_linear_lora(f1, "dense", AdapterCls, r=r, alpha=alpha)
        if "ffn2" in targets:
            replaced += _wrap_linear_lora(f2, "dense", AdapterCls, r=r, alpha=alpha)

    if replaced > 0:
        mark_adapters_dirty(model, kind="lora")
    return replaced


def inject_lora_hf_vit(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
    targets: Sequence[str],
) -> int:
    _probe_hf_vit(model)

    replaced = 0
    for layer in _iter_hf_vit_layers(model):
        attn = layer.attention
        sa = attn.attention
        ao = attn.output
        f1 = layer.intermediate
        f2 = layer.output

        if "q" in targets:
            replaced += _wrap_linear_lora(sa, "query", AdapterCls, r=r, alpha=alpha)
        if "k" in targets:
            replaced += _wrap_linear_lora(sa, "key", AdapterCls, r=r, alpha=alpha)
        if "v" in targets:
            replaced += _wrap_linear_lora(sa, "value", AdapterCls, r=r, alpha=alpha)
        if "o" in targets:
            replaced += _wrap_linear_lora(ao, "dense", AdapterCls, r=r, alpha=alpha)
        if "ffn1" in targets:
            replaced += _wrap_linear_lora(f1, "dense", AdapterCls, r=r, alpha=alpha)
        if "ffn2" in targets:
            replaced += _wrap_linear_lora(f2, "dense", AdapterCls, r=r, alpha=alpha)

    if replaced > 0:
        mark_adapters_dirty(model, kind="lora")
    return replaced


def inject_lora_timm_vit(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
    targets: Sequence[str],
) -> int:
    _probe_timm_vit(model)

    replaced = 0
    for block in _iter_timm_vit_blocks(model):
        attn = getattr(block, "attn", None)
        mlp = getattr(block, "mlp", None)
        if attn is None:
            continue

        # q/k/v
        if any(t in targets for t in ("q", "k", "v")):
            qkv = getattr(attn, "qkv", None)
            if isinstance(qkv, nn.Linear):
                want = {t for t in ("q", "k", "v") if t in targets}
                if want == {"q", "k", "v"}:
                    replaced += _wrap_linear_lora(attn, "qkv", AdapterCls, r=r, alpha=alpha)
                else:
                    if not isinstance(qkv, _SelectiveQKVLoRA):
                        attn.qkv = _SelectiveQKVLoRA(qkv, AdapterCls, r=r, alpha=alpha, targets=tuple(sorted(want)))
                        replaced += 1
            else:
                # Non-fused projections
                mapping = [
                    ("q", ("q", "query")),
                    ("k", ("k", "key")),
                    ("v", ("v", "value")),
                ]
                for key, attrs in mapping:
                    if key not in targets:
                        continue
                    for a in attrs:
                        if hasattr(attn, a):
                            replaced += _wrap_linear_lora(attn, a, AdapterCls, r=r, alpha=alpha)
                            break

        # o
        if "o" in targets:
            if hasattr(attn, "proj"):
                replaced += _wrap_linear_lora(attn, "proj", AdapterCls, r=r, alpha=alpha)
            elif hasattr(attn, "dense"):
                replaced += _wrap_linear_lora(attn, "dense", AdapterCls, r=r, alpha=alpha)

        # ffn
        if mlp is not None:
            if "ffn1" in targets and hasattr(mlp, "fc1"):
                replaced += _wrap_linear_lora(mlp, "fc1", AdapterCls, r=r, alpha=alpha)
            if "ffn2" in targets and hasattr(mlp, "fc2"):
                replaced += _wrap_linear_lora(mlp, "fc2", AdapterCls, r=r, alpha=alpha)

    if replaced > 0:
        mark_adapters_dirty(model, kind="lora")
    return replaced


def inject_lora_fallback(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
    targets: Sequence[str],
) -> int:
    """
    Try known architectures in order.
    Only swallow AttributeError (structure mismatch). Do NOT hide real bugs.
    """
    dbg = _inject_debug_enabled()
    for fn in (inject_lora_roberta, inject_lora_hf_vit, inject_lora_timm_vit):
        try:
            n = fn(model, AdapterCls=AdapterCls, r=r, alpha=alpha, targets=targets)
        except AttributeError as e:
            if dbg:
                print(f"[inject_lora_fallback] {fn.__name__} mismatch: {e}")
            continue
        if n <= 0 and dbg:
            print(f"[inject_lora_fallback] {fn.__name__} matched but replaced=0")
        if n > 0:
            return n
        # If matched but replaced=0, keep trying others (safe; others should mismatch quickly).
    return 0


def inject_lora_roberta_head(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
) -> int:
    net = getattr(model, "net", model)
    head = _resolve_maybe_callable_module(getattr(net, "classifier", None))
    if head is None:
        return 0

    replaced = 0
    for attr in ("dense", "out_proj", "classifier", "proj"):
        if hasattr(head, attr):
            replaced += _wrap_linear_lora(head, attr, AdapterCls, r=r, alpha=alpha)

    if replaced > 0:
        mark_adapters_dirty(model, kind="lora")
    return replaced


def inject_lora_generic_head(
    model: nn.Module,
    *,
    AdapterCls: Type[nn.Module],
    r: int,
    alpha: int,
) -> int:
    net = getattr(model, "net", model)
    replaced = 0

    for attr in ("classifier", "head", "fc", "lm_head"):
        m = _resolve_maybe_callable_module(getattr(net, attr, None))
        if m is None:
            continue
        if isinstance(m, nn.Linear):
            replaced += _wrap_linear_lora(net, attr, AdapterCls, r=r, alpha=alpha)
        else:
            for sub in ("dense", "out_proj", "proj", "classifier", "head", "fc"):
                if hasattr(m, sub):
                    replaced += _wrap_linear_lora(m, sub, AdapterCls, r=r, alpha=alpha)

    for holder_attr in ("vit", "vision_model", "model"):
        holder = _resolve_maybe_callable_module(getattr(net, holder_attr, None))
        if holder is None:
            continue
        for attr in ("classifier", "head", "fc"):
            if hasattr(holder, attr):
                replaced += _wrap_linear_lora(holder, attr, AdapterCls, r=r, alpha=alpha)

    if replaced > 0:
        mark_adapters_dirty(model, kind="lora")
    return replaced


# -----------------------------
# PaCA injectors (fallbacks)
# -----------------------------

def inject_paca_roberta(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    seed: int,
    k_per_row: int | None,
    targets: Sequence[str],
) -> int:
    _probe_roberta(model)
    from models.paca_layers import PaCALinearAdapter

    replaced = 0
    offsets = {"q": 1, "k": 2, "v": 3, "o": 4, "ffn1": 5, "ffn2": 6}

    for i, layer in enumerate(_iter_roberta_layers(model)):
        base = int(seed + 1000 * i)
        sa = layer.attention.self
        ao = layer.attention.output
        f1 = layer.intermediate
        f2 = layer.output

        if "q" in targets:
            replaced += _wrap_linear_paca(sa, "query", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["q"], k_per_row=k_per_row)
        if "k" in targets:
            replaced += _wrap_linear_paca(sa, "key", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["k"], k_per_row=k_per_row)
        if "v" in targets:
            replaced += _wrap_linear_paca(sa, "value", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["v"], k_per_row=k_per_row)
        if "o" in targets:
            replaced += _wrap_linear_paca(ao, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["o"], k_per_row=k_per_row)
        if "ffn1" in targets:
            replaced += _wrap_linear_paca(f1, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["ffn1"], k_per_row=k_per_row)
        if "ffn2" in targets:
            replaced += _wrap_linear_paca(f2, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["ffn2"], k_per_row=k_per_row)

    if replaced > 0:
        mark_adapters_dirty(model, kind="paca")
    return replaced


def inject_paca_hf_vit(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    seed: int,
    k_per_row: int | None,
    targets: Sequence[str],
) -> int:
    _probe_hf_vit(model)
    from models.paca_layers import PaCALinearAdapter

    replaced = 0
    offsets = {"q": 1, "k": 2, "v": 3, "o": 4, "ffn1": 5, "ffn2": 6}

    for i, layer in enumerate(_iter_hf_vit_layers(model)):
        base = int(seed + 1000 * i)
        attn = layer.attention
        sa = attn.attention
        ao = attn.output
        f1 = layer.intermediate
        f2 = layer.output

        if "q" in targets:
            replaced += _wrap_linear_paca(sa, "query", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["q"], k_per_row=k_per_row)
        if "k" in targets:
            replaced += _wrap_linear_paca(sa, "key", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["k"], k_per_row=k_per_row)
        if "v" in targets:
            replaced += _wrap_linear_paca(sa, "value", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["v"], k_per_row=k_per_row)
        if "o" in targets:
            replaced += _wrap_linear_paca(ao, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["o"], k_per_row=k_per_row)
        if "ffn1" in targets:
            replaced += _wrap_linear_paca(f1, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["ffn1"], k_per_row=k_per_row)
        if "ffn2" in targets:
            replaced += _wrap_linear_paca(f2, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base + offsets["ffn2"], k_per_row=k_per_row)

    if replaced > 0:
        mark_adapters_dirty(model, kind="paca")
    return replaced


def inject_paca_timm_vit(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    seed: int,
    k_per_row: int | None,
    targets: Sequence[str],
) -> int:
    _probe_timm_vit(model)
    from models.paca_layers import PaCALinearAdapter

    replaced = 0
    offsets = {"q": 1, "k": 2, "v": 3, "o": 4, "ffn1": 5, "ffn2": 6}

    for i, block in enumerate(_iter_timm_vit_blocks(model)):
        base_seed = int(seed + 1000 * i)
        attn = getattr(block, "attn", None)
        mlp = getattr(block, "mlp", None)
        if attn is None:
            continue

        # q/k/v
        if any(t in targets for t in ("q", "k", "v")):
            qkv = getattr(attn, "qkv", None)
            if isinstance(qkv, nn.Linear):
                replaced += _wrap_linear_paca(
                    attn, "qkv", PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets["q"], k_per_row=k_per_row
                )
            else:
                mapping = [
                    ("q", ("q", "query")),
                    ("k", ("k", "key")),
                    ("v", ("v", "value")),
                ]
                for key, attrs in mapping:
                    if key not in targets:
                        continue
                    for a in attrs:
                        if hasattr(attn, a):
                            replaced += _wrap_linear_paca(
                                attn, a, PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets[key], k_per_row=k_per_row
                            )
                            break

        # o
        if "o" in targets:
            if hasattr(attn, "proj"):
                replaced += _wrap_linear_paca(
                    attn, "proj", PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets["o"], k_per_row=k_per_row
                )
            elif hasattr(attn, "dense"):
                replaced += _wrap_linear_paca(
                    attn, "dense", PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets["o"], k_per_row=k_per_row
                )

        # ffn
        if mlp is not None:
            if "ffn1" in targets and hasattr(mlp, "fc1"):
                replaced += _wrap_linear_paca(
                    mlp, "fc1", PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets["ffn1"], k_per_row=k_per_row
                )
            if "ffn2" in targets and hasattr(mlp, "fc2"):
                replaced += _wrap_linear_paca(
                    mlp, "fc2", PaCALinearAdapter, r=r, alpha=alpha, seed=base_seed + offsets["ffn2"], k_per_row=k_per_row
                )

    if replaced > 0:
        mark_adapters_dirty(model, kind="paca")
    return replaced


def inject_paca_fallback(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    seed: int,
    k_per_row: int | None,
    targets: Sequence[str],
) -> int:
    """
    Try known architectures in order.
    Only swallow AttributeError (structure mismatch). Do NOT hide real bugs.
    """
    dbg = _inject_debug_enabled()
    for fn in (inject_paca_roberta, inject_paca_hf_vit, inject_paca_timm_vit):
        try:
            n = fn(model, r=r, alpha=alpha, seed=seed, k_per_row=k_per_row, targets=targets)
        except AttributeError as e:
            if dbg:
                print(f"[inject_paca_fallback] {fn.__name__} mismatch: {e}")
            continue
        if n <= 0 and dbg:
            print(f"[inject_paca_fallback] {fn.__name__} matched but replaced=0")
        if n > 0:
            return n
    return 0