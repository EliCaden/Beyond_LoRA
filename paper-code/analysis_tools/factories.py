# analysis_tools/factories.py — PAPER-FRIENDLY tweaks
# Keeps your modular builders; ensures eval = VALIDATION loader, model frozen.
from typing import Tuple
import os, re, json, time, warnings, torch
from torch.utils.data import DataLoader

# quiet HF/datasets logs (optional)
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    from datasets.utils import logging as ds_logging
    ds_logging.set_verbosity_error()
except Exception:
    pass

from data import get_data
from models.vit_classifier import ViTImageModel

_NUM_RE = re.compile(r"(\d+)")

# --------- Text (RoBERTa on GLUE) builder: paper-style eval ---------
def build_roberta_glue_eval(
    run_dir: str,
    device: str = "cuda",
    *,
    task: str = "cola",
    max_length: int = 128,
    batch_size: int = 32,
    seed: int = 0,
    num_workers: int | None = None,
):
    """
    Paper-style evaluation builder for RoBERTa on GLUE:
      - loads the *best* checkpoint found under run_dir (by epoch/metrics.json),
      - maps suffix keys so we tolerate wrappers,
      - merges LoRA adapters into base if present,
      - freezes params and returns (model, VALIDATION loader).
    """
    import warnings, re
    from transformers import AutoTokenizer
    from models.roberta import RobertaGLUEModel
    from data import get_data

    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    print(f"[FACTORY:ROBERTA] Preparing model/val loader from run_dir={run_dir} task={task}")

    # tokenizer + data module
    tok = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    dm = get_data(
        "glue",
        task_name=task,
        tokenizer=tok,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        num_workers=(num_workers if num_workers is not None else max(1, min(4, (os.cpu_count() or 4) // 2))),
        use_length_bucketing=True,
        pad_to_multiple_of=8,
        drop_last=False,
    )
    dm.setup()

    # model
    model = RobertaGLUEModel(task=task, variant="base").to(device)
    model.eval()

    # choose checkpoint + best epoch hint
    best_ckpt = _choose_best_ckpt(run_dir)
    metrics = _find_metrics_path(run_dir)
    best_epoch = _maybe_best_from_metrics(metrics) if metrics else None

    # load base (suffix-map + *.base.* fallback)
    raw_sd = None
    if best_ckpt:
        try:
            raw_sd = _load_state_dict_candidates(best_ckpt)
            model_keys = set(model.state_dict().keys())
            mapped = _remap_suffix_keys(raw_sd, model_keys)
            # fill from *.base.weight/bias if present
            for k in list(model_keys):
                if k in mapped: continue
                if k.endswith(".weight"):
                    alt = k[:-7] + ".base.weight"
                    if alt in raw_sd: mapped[k] = raw_sd[alt]
                elif k.endswith(".bias"):
                    alt = k[:-5] + ".base.bias"
                    if alt in raw_sd: mapped[k] = raw_sd[alt]
            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if missing:   print(f"[FACTORY:ROBERTA] Missing keys (example): {list(missing)[:5]}")
            if unexpected:print(f"[FACTORY:ROBERTA] Unexpected keys (example): {list(unexpected)[:5]}")
            print(f"[FACTORY:ROBERTA] Loaded checkpoint: {os.path.basename(best_ckpt)}")
        except Exception as e:
            warnings.warn(f"[FACTORY:ROBERTA] Checkpoint load failed; using HF init. ({e})")
    else:
        print(f"[FACTORY:ROBERTA] No checkpoint found under: {run_dir}")

    # merge LoRA deltas if present (supports .A/.B, .lora_A/.lora_B, .lora.down/.lora.up)
    try:
        if raw_sd is not None:
            hint = (f"{os.path.basename(run_dir)}_{os.path.basename(best_ckpt)}"
                    if best_ckpt else os.path.basename(run_dir))
            n_merged = _merge_lora_into_model(model, raw_sd, hint_path=hint)
            print(f"[FACTORY:ROBERTA] LoRA merge applied to {n_merged} layer(s).")
    except Exception as me:
        warnings.warn(f"[FACTORY:ROBERTA] LoRA merge skipped due to error: {me}")

    # freeze & return VALIDATION loader
    for p in model.parameters(): p.requires_grad = False
    model.eval()

    loader = dm.val_dataloader()
    model._loaded_ckpt = os.path.basename(best_ckpt) if best_ckpt else None
    model._loaded_epoch = int(best_epoch) if best_epoch is not None else None
    print(f"[FACTORY:ROBERTA] Ready. val_batches≈{len(loader)} build_time={time.time()-t0:.1f}s")
    return model, loader


# --- Merge LoRA adapters (A/B) into base Linear weights in-place ---
def _merge_lora_into_model(model, sd, *, hint_path: str = "") -> int:
    """
    Looks for either:
      <prefix>.lora_A.weight + <prefix>.lora_B.weight
    or  <prefix>.lora.down.weight + <prefix>.lora.up.weight
    and applies:  W <- W + (B @ A) * (alpha / r)
    """
    import torch, re, os
    pmap = dict(model.named_parameters())
    merged = 0

    def _alpha_for(prefix: str, r: int) -> float:
        for k in (f"{prefix}.lora_alpha", f"{prefix}.alpha", f"{prefix}.scaling"):
            v = sd.get(k, None)
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                return float(v.item())
        m = re.search(r"[_\-]r(\d+)[_\-]a(\d+)\b", os.path.basename(hint_path or ""))
        if m:
            return float(int(m.group(2)))
        return float(r)

    keys = list(sd.keys())

    # Pattern set 1:  ...<prefix>.lora_A.weight / ...<prefix>.lora_B.weight
    for a_key in [k for k in keys if k.endswith(".lora_A.weight")]:
        prefix = a_key[:-len(".lora_A.weight")]
        b_key  = prefix + ".lora_B.weight"
        base_w = prefix + ".weight"
        if b_key not in sd or base_w not in pmap:
            continue
        A, B = sd[a_key], sd[b_key]  # A: (r,in), B: (out,r)
        if not (getattr(A, "ndim", 0) == 2 and getattr(B, "ndim", 0) == 2):
            continue
        r = int(A.shape[0]); alpha = _alpha_for(prefix, r); scale = alpha / max(1, r)
        with torch.no_grad():
            delta = (B @ A) * scale
            w = pmap[base_w].data
            if delta.shape != w.shape:
                continue
            w.add_(delta.to(dtype=w.dtype, device=w.device))
        merged += 1

    # Pattern set 2:  ...<prefix>.lora.down.weight / ...<prefix>.lora.up.weight
    for down_key in [k for k in keys if k.endswith(".lora.down.weight")]:
        prefix = down_key[:-len(".lora.down.weight")]
        up_key = prefix + ".lora.up.weight"
        base_w = prefix + ".weight"
        if up_key not in sd or base_w not in pmap:
            continue
        A, B = sd[down_key], sd[up_key]  # A: (r,in), B: (out,r)
        if not (getattr(A, "ndim", 0) == 2 and getattr(B, "ndim", 0) == 2):
            continue
        r = int(A.shape[0]); alpha = _alpha_for(prefix, r); scale = alpha / max(1, r)
        with torch.no_grad():
            delta = (B @ A) * scale
            w = pmap[base_w].data
            if delta.shape != w.shape:
                continue
            w.add_(delta.to(dtype=w.dtype, device=w.device))
        merged += 1

    # Pattern set 3:  ...<prefix>.A / ...<prefix>.B  (no .weight)  👈 ADD THIS BLOCK
    for a_key in [k for k in keys if k.endswith(".A")]:
        prefix = a_key[:-len(".A")]
        b_key  = prefix + ".B"
        base_w = prefix + ".weight"
        if b_key not in sd or base_w not in pmap:
            continue
        A, B = sd[a_key], sd[b_key]  # A: (r,in), B: (out,r)
        if not (getattr(A, "ndim", 0) == 2 and getattr(B, "ndim", 0) == 2):
            continue
        r = int(A.shape[0]); alpha = _alpha_for(prefix, r); scale = alpha / max(1, r)
        with torch.no_grad():
            delta = (B @ A) * scale
            w = pmap[base_w].data
            if delta.shape != w.shape:
                continue
            w.add_(delta.to(dtype=w.dtype, device=w.device))
        merged += 1

    return merged



def _find_metrics_path(run_dir: str):
    p = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(p): return p
    for f in os.listdir(run_dir):
        if f.endswith("-metrics.json"): return os.path.join(run_dir, f)
    return None

# --- in analysis_tools/factories.py (top-level) ---
def _maybe_enable_lora(model, run_dir: str):
    return False


def _maybe_best_from_metrics(metrics_path: str):
    try:
        with open(metrics_path,"r") as f: m = json.load(f)
        if "best_epoch" in m: return int(m["best_epoch"])
        if isinstance(m.get("best"), dict) and "epoch" in m["best"]: return int(m["best"]["epoch"])
        if "best_ckpt" in m:
            nums = _NUM_RE.findall(str(m["best_ckpt"]))
            if nums: return int(nums[-1])
    except Exception:
        pass
    return None

def _parse_epoch_from_name(path: str) -> int|None:
    name = os.path.basename(path).lower()
    if "best" in name: return None
    pats = [
        r'(?:^|[_\-])(?:epoch|ep|e)[=_\-]?(\d+)(?:[^0-9]|$)',
        r'(?:^|[_\-])(?:checkpoint|ckpt)[=_\-]?(\d+)(?:[^0-9]|$)',
        r'(\d+)(?:\.pt|\.pth|\.bin)$',
    ]
    for pat in pats:
        m = re.search(pat, name)
        if m:
            try: return int(m.group(1))
            except Exception: return None
    return None

def _find_ckpts(run_dir: str):
    out = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            fn = f.lower()
            if fn.endswith((".pt",".pth",".bin")) and "optim" not in fn and "optimizer" not in fn:
                out.append(os.path.join(root,f))
    def _key(p):
        parts = re.split(r"(\d+)", os.path.basename(p))
        return [int(t) if t.isdigit() else t for t in parts]
    return sorted(set(out), key=_key)

def _load_state_dict_candidates(obj):
    if isinstance(obj, str):
        try:
            sd = torch.load(obj, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(obj, map_location="cpu")
    else:
        sd = obj
    if isinstance(sd, dict):
        for k in ["state_dict","model_state_dict","module","model","weights"]:
            if k in sd and isinstance(sd[k], dict): return sd[k]
        if all(isinstance(v, torch.Tensor) for v in sd.values()): return sd
    raise RuntimeError("Could not find a state_dict mapping in checkpoint.")

def _remap_suffix_keys(sd: dict, model_keys: set):
    mapped = {}
    for k, v in sd.items():
        parts = k.split(".")
        for i in range(len(parts)):
            suf = ".".join(parts[i:])
            if suf in model_keys:
                mapped[suf] = v; break
    return mapped

def _choose_best_ckpt(run_dir: str):
    items = []
    metrics = _find_metrics_path(run_dir)
    best_epoch = _maybe_best_from_metrics(metrics) if metrics else None
    for p in _find_ckpts(run_dir):
        e = _parse_epoch_from_name(p)
        try: mt = os.path.getmtime(p)
        except Exception: mt = 0.0
        items.append((e,p,mt))
    if not items: return None
    if all(e is None for e,_,_ in items):
        return max(items, key=lambda t: t[2])[1]
    usable = [t for t in items if t[0] is not None]
    # Allow override from env var
    force = os.environ.get("FORCE_BEST_EPOCH") or os.environ.get("CENTER_EPOCH")
    if force:
        try:
            fe = int(force)
            cands = [t for t in usable if t[0] == fe]
            if cands:
                return max(cands, key=lambda t: t[2])[1]
        except Exception:
            pass

    if not usable: return max(items, key=lambda t: t[2])[1]
    if best_epoch is None:
        best_epoch = max(t[0] for t in usable)
    cands = [t for t in usable if t[0] == best_epoch]
    if cands:
        return max(cands, key=lambda t: t[2])[1]
    return max(items, key=lambda t: t[2])[1]

# -----------------------
# Vision (ViT) builders (paper-style eval)
# -----------------------
def _build_vit_vision(
    run_dir: str,
    device: str = "cuda",
    *,
    batch_size: int = 64,
    dataset: str | None = None,
    dataset_kwargs: dict | None = None,
    vit_variant: str | None = None,
    pretrained: bool = True,
):
    """
    Paper-style evaluation builder for ViT that ALSO supports LoRA runs:
      - Loads best checkpoint under run_dir (by metrics/epoch naming).
      - If adapters are present, first loads base weights (handles *.base.*),
        then merges LoRA deltas into Linear weights.
      - Freezes params and returns (model, VALIDATION loader).
    """
    import re, time, warnings
    from pathlib import Path

    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    run_name = os.path.basename(run_dir).lower()
    print(f"[FACTORY:VIT] Preparing model/val loader from run_dir={run_dir}")

    # ---- infer dataset & variant from run name / envs ----
    if dataset is None:
        for cand in ("cifar10", "cifar100", "officehome", "terraincognita"):
            if run_name.startswith(cand) or (f"{cand}_" in run_name):
                dataset = cand
                break
    dataset = dataset or os.environ.get("DATASET", "cifar10")

    if vit_variant is None:
        m = re.search(r"vit-([a-z0-9]+)", run_name)
        vit_variant = (m.group(1) if m else None) or os.environ.get("VIT_VARIANT", "tiny")

    # ---- choose checkpoint + best epoch hint ----
    best_ckpt = _choose_best_ckpt(run_dir)
    metrics = _find_metrics_path(run_dir)
    best_epoch = _maybe_best_from_metrics(metrics) if metrics else None

    # ---- dataset defaults ----
    dataset_kwargs = dict(dataset_kwargs or {})
    repo_root = Path(__file__).resolve().parents[1]
    default_root = str((repo_root / "datasets").resolve())
    data_root = os.path.join(default_root) if not os.environ.get("DATA_ROOT") else os.environ["DATA_ROOT"]
    dataset_kwargs.setdefault("data_root", data_root)
    dataset_kwargs.setdefault("img_size", int(os.environ.get("IMG_SIZE", 224)))
    dataset_kwargs.setdefault("seed", int(os.environ.get("SEED", 0)))
    dataset_kwargs.setdefault("num_workers", int(os.environ.get("NUM_WORKERS", 4)))
    dataset_kwargs.setdefault("batch_size", batch_size)

    print(f"[FACTORY:VIT] inferred dataset={dataset}, vit_variant={vit_variant}")
    print(
        f"[FACTORY:VIT] data_root={dataset_kwargs['data_root']} img_size={dataset_kwargs['img_size']} "
        f"seed={dataset_kwargs['seed']} workers={dataset_kwargs['num_workers']} batch_size={dataset_kwargs['batch_size']}"
    )

    dm = get_data(dataset, **dataset_kwargs)
    dm.setup()
    num_labels = getattr(dm, "num_classes", 100)

    model = ViTImageModel(variant=vit_variant, num_classes=num_labels, pretrained=pretrained).to(device)
    model.eval()

    # ---- load checkpoint (map base weights, then merge LoRA if present) ----
    raw_sd = None
    if best_ckpt:
        try:
            raw_sd = _load_state_dict_candidates(best_ckpt)

            # 1) Try normal suffix mapping
            model_keys = set(model.state_dict().keys())
            mapped = _remap_suffix_keys(raw_sd, model_keys)

            # 2) Fill missing from custom "*.base.weight/bias" into "*.weight/bias"
            #    e.g. "...query.base.weight" -> "...query.weight"
            for k in list(model_keys):
                if k not in mapped:
                    if k.endswith(".weight"):
                        alt = k[:-len(".weight")] + ".base.weight"
                        if alt in raw_sd:
                            mapped[k] = raw_sd[alt]
                    elif k.endswith(".bias"):
                        alt = k[:-len(".bias")] + ".base.bias"
                        if alt in raw_sd:
                            mapped[k] = raw_sd[alt]

            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if missing:
                print(f"[FACTORY:VIT] Missing keys (example): {list(missing)[:5]}")
            if unexpected:
                print(f"[FACTORY:VIT] Unexpected keys (example): {list(unexpected)[:5]}")
            print(f"[FACTORY:VIT] Loaded checkpoint: {os.path.basename(best_ckpt)}")
        except Exception as e:
            warnings.warn(f"[FACTORY:VIT] Checkpoint load failed; using random init. ({e})")
            raw_sd = None
    else:
        print(f"[FACTORY:VIT] No checkpoint found under: {run_dir}")

    # ---- LoRA merge helper (supports .A/.B, .lora_A/.lora_B, .lora.down/.lora.up) ----
    def _merge_lora_into_model(_model, sd, *, hint_path: str = "") -> int:
        import torch, re, os
        pmap = dict(_model.named_parameters())
        keys = list(sd.keys())
        merged = 0

        def _alpha_from_sd_or_hint(prefix: str, r: int) -> float:
            # (1) Prefer buffers saved in the checkpoint
            for k in (f"{prefix}.lora_alpha", f"{prefix}.alpha", f"{prefix}.scaling"):
                v = sd.get(k, None)
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    return float(v.item())
            # (2) Fall back to parsing from name/path "..._r8_a16..."
            m = re.search(r"[_\-]r(\d+)[_\-]a(\d+)\b", os.path.basename(hint_path or ""))
            if m:
                return float(int(m.group(2)))
            # (3) Your historical adapter behavior (alpha==0 -> 2*r) — keep previous default
            return float(2 * r)

        # Style 1: <prefix>.lora_A.weight / <prefix>.lora_B.weight
        for a_key in [k for k in keys if k.endswith(".lora_A.weight")]:
            prefix = a_key[:-len(".lora_A.weight")]
            b_key  = prefix + ".lora_B.weight"
            base_w = prefix + ".weight"
            if (b_key not in sd) or (base_w not in pmap):
                continue
            A, B = sd[a_key], sd[b_key]
            if getattr(A, "ndim", 0) != 2 or getattr(B, "ndim", 0) != 2:
                continue
            r = int(A.shape[0]); scale = _alpha_from_sd_or_hint(prefix, r) / max(1, r)
            with torch.no_grad():
                delta = (B @ A) * scale
                w = pmap[base_w].data
                if delta.shape == w.shape:
                    w.add_(delta.to(dtype=w.dtype, device=w.device)); merged += 1

        # Style 2: <prefix>.lora.down.weight / <prefix>.lora.up.weight
        for down_key in [k for k in keys if k.endswith(".lora.down.weight")]:
            prefix = down_key[:-len(".lora.down.weight")]
            up_key = prefix + ".lora.up.weight"
            base_w = prefix + ".weight"
            if (up_key not in sd) or (base_w not in pmap):
                continue
            A, B = sd[down_key], sd[up_key]
            if getattr(A, "ndim", 0) != 2 or getattr(B, "ndim", 0) != 2:
                continue
            r = int(A.shape[0]); scale = _alpha_from_sd_or_hint(prefix, r) / max(1, r)
            with torch.no_grad():
                delta = (B @ A) * scale
                w = pmap[base_w].data
                if delta.shape == w.shape:
                    w.add_(delta.to(dtype=w.dtype, device=w.device)); merged += 1

        # Style 3 (yours): <prefix>.A / <prefix>.B
        for a_key in [k for k in keys if k.endswith(".A")]:
            prefix = a_key[:-len(".A")]
            b_key  = prefix + ".B"
            base_w = prefix + ".weight"
            if (b_key not in sd) or (base_w not in pmap):
                continue
            A, B = sd[a_key], sd[b_key]
            if getattr(A, "ndim", 0) != 2 or getattr(B, "ndim", 0) != 2:
                continue
            r = int(A.shape[0]); scale = _alpha_from_sd_or_hint(prefix, r) / max(1, r)
            with torch.no_grad():
                delta = (B @ A) * scale
                w = pmap[base_w].data
                if delta.shape == w.shape:
                    w.add_(delta.to(dtype=w.dtype, device=w.device)); merged += 1

        return merged


    # ---- merge LoRA deltas (if present) ----
    try:
        if raw_sd is not None:
            n_merged = _merge_lora_into_model(model, raw_sd, hint_path=(best_ckpt or run_dir))
            if n_merged > 0:
                print(f"[FACTORY:VIT] LoRA merge applied to {n_merged} layer(s).")
            else:
                print(f"[FACTORY:VIT] No LoRA adapters found to merge.")
    except Exception as me:
        warnings.warn(f"[FACTORY:VIT] LoRA merge skipped due to error: {me}")

    # ---- freeze & return VALIDATION loader (paper mode) ----
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    loader = dm.val_dataloader()

    model._loaded_ckpt = os.path.basename(best_ckpt) if best_ckpt else None
    model._loaded_epoch = int(best_epoch) if best_epoch is not None else None

    print(
        f"[FACTORY:VIT] Ready. dataset={dataset} classes={num_labels} "
        f"val_batches≈{len(loader)} build_time={time.time()-t0:.1f}s"
    )
    return model, loader

# Public aliases (unchanged signatures)
def build_vit_vision_auto(run_dir: str, device: str = "cuda", *, batch_size: int = 64,
                          dataset: str | None = None, dataset_kwargs: dict | None = None,
                          vit_variant: str = "tiny"):
    return _build_vit_vision(run_dir, device, dataset=dataset, dataset_kwargs=dataset_kwargs,
                             vit_variant=vit_variant, batch_size=batch_size)

def build_vit_vision_full(run_dir: str, device: str = "cuda", *,
                          batch_size: int = 64, dataset: str | None = None,
                          dataset_kwargs: dict | None = None, vit_variant: str | None = None,
                          pretrained: bool = True):
    return _build_vit_vision(run_dir, device, dataset=dataset, dataset_kwargs=dataset_kwargs,
                             vit_variant=vit_variant, batch_size=batch_size, pretrained=pretrained)

def build_vit_vision_full_cifar10(run_dir: str, device: str = "cuda", *,
                                  batch_size: int = 64, dataset_kwargs: dict | None = None,
                                  vit_variant: str = "tiny"):
    return _build_vit_vision(run_dir, device, dataset="cifar10",
                             dataset_kwargs=dataset_kwargs, vit_variant=vit_variant, batch_size=batch_size)

# -----------------------
# ConViT builders (paper-style eval)
# -----------------------
def _resolve_convit_name(vit_variant: str | None) -> str:
    """
    Map your --vit-variant flag to a timm ConViT model name.
    Known variants in timm: convit_tiny, convit_small, convit_base
    """
    vv = (vit_variant or os.environ.get("VIT_VARIANT") or "tiny").lower()
    # allow short aliases
    alias = {"ti": "tiny", "t": "tiny", "s": "small", "sm": "small",
             "b": "base", "base": "base", "tiny": "tiny", "small": "small"}
    vv = alias.get(vv, vv)
    return f"convit_{vv}"

def _build_convit_vision(
    run_dir: str,
    device: str = "cuda",
    *,
    dataset: str | None = None,
    dataset_kwargs: dict | None = None,
    vit_variant: str | None = None,
    batch_size: int = 64,
    pretrained: bool = True,
):
    """
    Paper-style evaluation builder for ConViT.
    - Loads the *best* checkpoint found under run_dir (by epoch/metrics.json).
    - Freezes params and returns (model, validation_loader).
    - Uses the same get_data() pipeline as your ViT factory to ensure
      val-time transforms (no mixup/cutmix/etc.).
    """
    import re
    import time
    from pathlib import Path

    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    run_name = os.path.basename(run_dir).lower()
    print(f"[FACTORY:CONVIT] Preparing model/val loader from run_dir={run_dir}")

    # Infer dataset if not provided
    if dataset is None:
        for cand in ("cifar10", "cifar100", "officehome", "terraincognita"):
            if run_name.startswith(cand) or (f"{cand}_" in run_name):
                dataset = cand
                break
    dataset = dataset or os.environ.get("DATASET", "cifar10")

    # Resolve variant and timm model name
    if vit_variant is None:
        m = re.search(r"vit-([a-z0-9]+)", run_name)  # your run names contain vit-<variant>
        guessed = (m.group(1) if m else None) or os.environ.get("VIT_VARIANT", "tiny")
        vit_variant = guessed
    convit_name = _resolve_convit_name(vit_variant)
    print(f"[FACTORY:CONVIT] inferred dataset={dataset}, convit={convit_name}")

    # Data args (mirror ViT factory defaults)
    dataset_kwargs = dict(dataset_kwargs or {})
    repo_root = Path(__file__).resolve().parents[1]
    default_root = str((repo_root / "datasets").resolve())
    data_root = os.path.join(default_root) if not os.environ.get("DATA_ROOT") else os.environ["DATA_ROOT"]
    dataset_kwargs.setdefault("data_root", data_root)
    dataset_kwargs.setdefault("img_size", int(os.environ.get("IMG_SIZE", 224)))
    dataset_kwargs.setdefault("seed", int(os.environ.get("SEED", 0)))
    dataset_kwargs.setdefault("num_workers", int(os.environ.get("NUM_WORKERS", 4)))
    dataset_kwargs.setdefault("batch_size", batch_size)

    print(f"[FACTORY:CONVIT] data_root={dataset_kwargs['data_root']} img_size={dataset_kwargs['img_size']} "
          f"seed={dataset_kwargs['seed']} workers={dataset_kwargs['num_workers']} batch_size={dataset_kwargs['batch_size']}")

    dm = get_data(dataset, **dataset_kwargs)
    dm.setup()
    num_labels = getattr(dm, "num_classes", 100)

    # Build ConViT
    try:
        from timm import create_model
    except Exception as e:
        raise RuntimeError(
            "timm is required for ConViT. Install with `pip install timm`."
        ) from e

    print(f"[FACTORY:CONVIT] Creating timm model: {convit_name} (pretrained={pretrained})")
    model = create_model(convit_name, num_classes=num_labels, pretrained=pretrained).to(device)
    model.eval()

    # Load best checkpoint (align by suffix to tolerate wrappers/nn.DataParallel)
    best_ckpt = _choose_best_ckpt(run_dir)
    metrics = _find_metrics_path(run_dir)
    best_epoch = _maybe_best_from_metrics(metrics) if metrics else None

    if best_ckpt:
        try:
            raw_sd = _load_state_dict_candidates(best_ckpt)
            mapped = _remap_suffix_keys(raw_sd, set(model.state_dict().keys()))
            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if missing:
                print(f"[FACTORY:CONVIT] Missing keys (example): {list(missing)[:5]}")
            if unexpected:
                print(f"[FACTORY:CONVIT] Unexpected keys (example): {list(unexpected)[:5]}")
            print(f"[FACTORY:CONVIT] Loaded checkpoint: {os.path.basename(best_ckpt)}")
        except Exception as e:
            warnings.warn(f"[FACTORY:CONVIT] Checkpoint load failed; using random init. ({e})")
    else:
        print(f"[FACTORY:CONVIT] No checkpoint found under: {run_dir}")

    # Freeze & eval mode for loss landscapes
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # PAPER: use VALIDATION loader
    loader = dm.val_dataloader()

    # Convenience fields (used by your tools)
    model._loaded_ckpt = os.path.basename(best_ckpt) if best_ckpt else None
    model._loaded_epoch = int(best_epoch) if best_epoch is not None else None

    print(f"[FACTORY:CONVIT] Ready. dataset={dataset} classes={num_labels} "
          f"val_batches≈{len(loader)} build_time={time.time()-t0:.1f}s")
    return model, loader

# Public aliases (to match your ViT factory style)
def build_convit_vision_full(
    run_dir: str, device: str = "cuda", *,
    batch_size: int = 64, dataset: str | None = None,
    dataset_kwargs: dict | None = None, vit_variant: str | None = None,
    pretrained: bool = True
):
    return _build_convit_vision(run_dir, device, dataset=dataset, dataset_kwargs=dataset_kwargs,
                                vit_variant=vit_variant, batch_size=batch_size, pretrained=pretrained)

def build_convit_vision_full_cifar10(
    run_dir: str, device: str = "cuda", *,
    batch_size: int = 64, dataset_kwargs: dict | None = None,
    vit_variant: str = "tiny", pretrained: bool = True
):
    return _build_convit_vision(run_dir, device, dataset="cifar10", dataset_kwargs=dataset_kwargs,
                                vit_variant=vit_variant, batch_size=batch_size, pretrained=pretrained)

