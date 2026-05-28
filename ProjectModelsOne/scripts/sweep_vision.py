# scripts/sweep_vision.py
from __future__ import annotations

import dataclasses
import itertools
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# --- repo-local imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.targets import normalize_target_modules, target_tag

from scripts.sweep_common import (
    CommentArgParser,
    add_bool_arg,
    resolve_cli_env,
    set_all_seeds,
    str2bool,
    strip_method_suffixes,
    norm_recipe,
    build_ra_pairs,
    resolve_ba_alpha,
    metrics_exist,
    aggregate_and_choose,
    trainer_mode_kwargs,
)

from data.cifar10 import CIFAR10DataModule
from data.cifar100 import CIFAR100DataModule
from data.officehome import OfficeHomeDataModule
from data.terraincognita import TerraIncognitaDataModule
from data.caltech101 import Caltech101DataModule
from data.flowers102 import Flowers102DataModule
from data.svhn import SVHNDataModule
from data.pet import OxfordPetDataModule

from models.vit_classifier import ViTImageModel as ViTForImageClassification, resolve_pretrained_id

try:
    from transformers import AutoImageProcessor
except Exception:
    AutoImageProcessor = None

try:
    from models.convit_classifier import ConViTImageModel
except Exception:
    ConViTImageModel = None

from methods import FullFineTune, LoRAQV, LoRAQVLoRAOnly, LoRAQVWithHead, HeadOnlyFineTune
from methods.paca_qv import PaCAQV

from trainers.generic_trainer import GenericTrainer, TrainConfig
from trainers.callbacks import (
    ChainResetCallback,
    WandbCallback,
    BASparseFinalCallback,
    BASparsityLoggerCallback,
)

# Optional (don’t hard-fail if not present)
try:
    from trainers.callbacks import EarlyStoppingCallback  # type: ignore
except Exception:
    EarlyStoppingCallback = None  # type: ignore
    
try:
    from trainers.callbacks import FlopsCounterCallback  # type: ignore
except Exception:
    FlopsCounterCallback = None  # type: ignore
    
try:
    from trainers.callbacks import CudaPeakMemoryCallback  # type: ignore
except Exception:
    CudaPeakMemoryCallback = None  # type: ignore


DATASETS_ALL = [
    "cifar10",
    "cifar100",
    "officehome",
    "terraincognita",
    "caltech101",
    "flowers102",
    "svhn",
    "oxford_iiit_pet",
]
VIT_VARIANTS = ["tiny", "base", "large", "xxl"]

CHAIN_RECIPES = [
    "cola",
    "rac_a",
    "rac_b",
    "c3la",
    "shuffle",
    "sparse_shuffle",
    "sparse_c3la",
]
NONCHAIN_RECIPES = [
    "base",
    "asym_a",
    "asym_b",
    "cheap",
    "random_cheap",
    "sparse_cheap",
    "ba_sparse_lora",
    "ba_sparse_final",
    "ba_sparse_fix_mask",
]

PLUS_ALIASES = {"lora_plus", "lora+"}
PACA_VARIANTS = {"paca", "dpaca", "cpaca", "dcpaca"}


# ---------------------------
# ctor filtering (methods/callbacks/trainconfig)
# ---------------------------

def _filter_kwargs_for_ctor(Cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs against Cls.__init__ signature.
    If ctor has **kwargs, do not filter.
    """
    try:
        sig = inspect.signature(Cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _add_callback(callbacks: list, Cls: Any, *, strict: bool, **kwargs: Any) -> None:
    if Cls is None:
        if strict:
            raise RuntimeError("Requested callback is not available in this repo.")
        return
    filt = _filter_kwargs_for_ctor(Cls, kwargs)
    try:
        callbacks.append(Cls(**filt))
    except Exception as e:
        msg = f"[CALLBACK][ERROR] Failed to construct {getattr(Cls, '__name__', str(Cls))}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        print(msg)


_TCFG_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def _filter_trainconfig_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in _TCFG_FIELDS}


def _construct_method_with_optional_target(MethodClass: Any, *, target_modules: Any, **kwargs: Any):
    """
    Safe method construction:
      - filters kwargs to match ctor
      - injects target_modules if supported; else sets attribute post-hoc
    """
    tm = normalize_target_modules(target_modules)

    try:
        sig = inspect.signature(MethodClass.__init__)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_kwargs:
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        if "target_modules" in sig.parameters:
            kwargs["target_modules"] = tm
    except Exception:
        pass

    m = MethodClass(**kwargs)
    if tm is not None:
        try:
            if not hasattr(m, "target_modules"):
                setattr(m, "target_modules", tm)
        except Exception:
            pass
    return m


# ---------------------------
# small utils
# ---------------------------

def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_method_aliases(methods: List[str]) -> List[str]:
    out: List[str] = []
    for m in methods:
        mm = m.strip().lower()
        if mm == "paca_qv":
            mm = "paca"
        if mm == "base":
            mm = "lora"
        elif mm.startswith("base_"):
            mm = "lora_" + mm[len("base_") :]
        out.append(mm)
    return out


def _require_nonempty(name: str, xs: Optional[List[Any]]) -> None:
    if xs is None:
        return
    if len(xs) == 0:
        raise ValueError(f"{name} is empty (did you pass the flag with no values?)")


def _fmt_float(x: float) -> str:
    return f"{float(x):g}"


def _wandb_log_every_value(v: Any) -> int:
    if v is None:
        return 200
    try:
        iv = int(v)
    except Exception:
        return 200
    if iv < 0:
        return 200
    return iv


def _configure_cuda_fastpath(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass


# ---------------------------
# Data factory
# ---------------------------

def _maybe_add_val_split(dm_cls, kwargs: Dict[str, Any], val_split: Any) -> Dict[str, Any]:
    if val_split is None:
        return kwargs
    try:
        sig = inspect.signature(dm_cls.__init__)
        if "val_split" in sig.parameters:
            kwargs["val_split"] = float(val_split)
    except Exception:
        pass
    return kwargs


def get_data(cfg: Dict[str, Any], norm_mean=None, norm_std=None):
    dataset = cfg["dataset"].lower()
    batch_size = int(cfg["batch_size"])
    img_size = int(cfg["img_size"])
    seed = int(cfg["seed"])
    num_workers = int(cfg.get("num_workers", 4))
    val_split = cfg.get("val_split")

    use_randaugment = bool(cfg.get("randaugment", False))
    ra_n = int(cfg.get("ra_n", 2))
    ra_m = int(cfg.get("ra_m", 9))
    random_erasing = float(cfg.get("random_erasing", 0.0))
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.get("cutmix_alpha", 0.0))
    mix_prob = float(cfg.get("mix_prob", 0.0))
    switch_prob = float(cfg.get("switch_prob", 0.5))

    if dataset == "cifar10":
        kwargs = dict(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
            use_randaugment=use_randaugment,
            ra_n=ra_n,
            ra_m=ra_m,
            random_erasing_p=random_erasing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            switch_prob=switch_prob,
        )
        kwargs = _maybe_add_val_split(CIFAR10DataModule, kwargs, val_split)
        dm = CIFAR10DataModule(**kwargs)

    elif dataset == "cifar100":
        dm = CIFAR100DataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
        )

    elif dataset == "officehome":
        ratios = (0.7, 0.15, 0.15)
        if val_split is not None:
            v = float(val_split)
            v = max(0.0, min(0.49, v))
            ratios = (max(1e-6, 1.0 - 2.0 * v), v, v)

        officehome_url = cfg.get("officehome_url", None)

        dm = OfficeHomeDataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            split_ratios=ratios,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
            use_randaugment=use_randaugment,
            ra_n=ra_n,
            ra_m=ra_m,
            random_erasing_p=random_erasing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            switch_prob=switch_prob,
            download=bool(officehome_url),
            download_url=officehome_url,
        )

    elif dataset == "terraincognita":
        terrain_url = cfg.get("terraincognita_url", None)

        dm = TerraIncognitaDataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
            use_randaugment=use_randaugment,
            ra_n=ra_n,
            ra_m=ra_m,
            random_erasing_p=random_erasing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            switch_prob=switch_prob,
            download=bool(terrain_url),
            download_url=terrain_url,
        )

    elif dataset == "caltech101":
        dm = Caltech101DataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
        )

    elif dataset == "flowers102":
        dm = Flowers102DataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
        )

    elif dataset == "svhn":
        dm = SVHNDataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
        )

    elif dataset == "oxford_iiit_pet":
        dm = OxfordPetDataModule(
            data_root=cfg["data_root"],
            batch_size=batch_size,
            img_size=img_size,
            seed=seed,
            num_workers=num_workers,
            mean=tuple(norm_mean) if norm_mean else None,
            std=tuple(norm_std) if norm_std else None,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dm.setup()
    return dm


# ---------------------------
# Grid builder
# ---------------------------

def build_run_cfgs_for_dataset(
    dataset: str,
    vit_variant: str,
    methods: List[str],
    seeds: List[int],
    lrs: List[float],
    ranks: List[int],
    alphas: Optional[List[int]],
    common: Dict[str, Any],
    lrs_by_method: Optional[Dict[str, List[float]]] = None,
) -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []

    _require_nonempty("seeds", seeds)
    _require_nonempty("lrs", lrs)
    _require_nonempty("ranks", ranks)
    if alphas is not None:
        _require_nonempty("alphas", alphas)

    ra_pairs = build_ra_pairs(ranks, alphas)
    if not ra_pairs:
        raise ValueError("build_ra_pairs(...) returned no (rank, alpha) pairs; check --ranks/--alphas")

    tm_tag = target_tag(common.get("target_modules"))

    for method_name in methods:
        core0, train_head, only_flag = strip_method_suffixes(method_name)

        if core0 == "paca_qv":
            core0 = "paca"

        method_lrs = lrs_by_method.get(method_name.lower()) if lrs_by_method else None
        if method_lrs is None and lrs_by_method:
            method_lrs = lrs_by_method.get(core0)
        if method_lrs is None:
            method_lrs = lrs

        # FULL
        if core0 == "full":
            if train_head or only_flag:
                raise ValueError(f"Suffixes not allowed for 'full': {method_name}")
            for seed, lr in itertools.product(seeds, method_lrs):
                cfg = dict(common)
                base = f"{dataset}_vit-{vit_variant}_full{tm_tag}"
                cfg.update(
                    dataset=dataset,
                    vit_variant=vit_variant,
                    method="full",
                    seed=int(seed),
                    lr=float(lr),
                    rank=None,
                    alpha=None,
                    init_recipe=None,
                    chain_recipe=None,
                    chain_every_epochs=None,
                    chain_every_steps=None,
                    lora_plus=False,
                    group_name=base,
                    run_name=f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}",
                )
                grid.append(cfg)
            continue

        # HEADS ONLY
        if core0 in {"heads", "head", "head_only"}:
            if train_head or only_flag:
                raise ValueError(f"Suffixes not allowed for 'heads': {method_name}")
            for seed, lr in itertools.product(seeds, method_lrs):
                cfg = dict(common)
                base = f"{dataset}_vit-{vit_variant}_heads{tm_tag}"
                cfg.update(
                    dataset=dataset,
                    vit_variant=vit_variant,
                    method="heads",
                    seed=int(seed),
                    lr=float(lr),
                    rank=None,
                    alpha=None,
                    init_recipe=None,
                    chain_recipe=None,
                    chain_every_epochs=None,
                    chain_every_steps=None,
                    lora_plus=False,
                    group_name=base,
                    run_name=f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}",
                )
                grid.append(cfg)
            continue

        # PaCA family
        if core0 in PACA_VARIANTS:
            paca_recipe = None if core0 == "paca" else core0
            for seed, lr, (r, a) in itertools.product(seeds, method_lrs, ra_pairs):
                cfg = dict(common)
                cfg.update(
                    dataset=dataset,
                    vit_variant=vit_variant,
                    method=("paca_head" if train_head else "paca"),
                    seed=int(seed),
                    lr=float(lr),
                    rank=int(r),
                    alpha=int(a),
                    init_recipe=paca_recipe,
                    chain_recipe=(paca_recipe if paca_recipe in {"cpaca", "dcpaca"} else None),
                    lora_plus=False,
                )
                if cfg["chain_recipe"] and cfg.get("chain_every_epochs") is None and cfg.get("chain_every_steps") is None:
                    cfg["chain_every_epochs"] = 1

                tag = ("paca_head" if train_head else "paca") if paca_recipe is None else paca_recipe
                if train_head and paca_recipe is not None:
                    tag = f"{tag}_head"

                kp = cfg.get("paca_k_per_row", None)
                ktag = f"_k{int(kp)}" if kp is not None else ""
                base = f"{dataset}_vit-{vit_variant}_{tag}{ktag}{tm_tag}_r{int(r)}_a{int(a)}"
                cfg["group_name"] = base
                cfg["run_name"] = f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}"
                grid.append(cfg)
            continue

        # LoRA / non-chain / plus
        is_plus = core0 in PLUS_ALIASES
        core = core0

        if core in ("lora",) + tuple(NONCHAIN_RECIPES) or is_plus:
            recipe = "base" if (core in {"lora", "base"} or is_plus) else core
            for seed, lr, (r, a) in itertools.product(seeds, method_lrs, ra_pairs):
                if only_flag:
                    method_field = "lora_only"
                elif train_head:
                    method_field = "lora_head"
                else:
                    method_field = "lora"

                base_tag = "lora_plus" if is_plus else ("lora" if core in {"lora", "base"} else recipe)
                if train_head:
                    tag = f"{base_tag}_head"
                elif only_flag:
                    tag = f"{base_tag}_frozen_head"
                else:
                    tag = base_tag

                cfg = dict(common)
                cfg.update(
                    dataset=dataset,
                    vit_variant=vit_variant,
                    method=method_field,
                    seed=int(seed),
                    lr=float(lr),
                    rank=int(r),
                    alpha=int(a),
                    init_recipe=recipe,
                    chain_recipe=None,
                    chain_every_epochs=None,
                    chain_every_steps=None,
                    lora_plus=bool(is_plus),
                )
                base = f"{dataset}_vit-{vit_variant}_{tag}{tm_tag}_r{int(r)}_a{int(a)}"
                cfg["group_name"] = base
                cfg["run_name"] = f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}"
                grid.append(cfg)
            continue

        # Chain recipes
        recipe = norm_recipe(core)
        if recipe not in CHAIN_RECIPES:
            raise ValueError(f"Unknown method '{method_name}'. core='{core}'")

        for seed, lr, (r, a) in itertools.product(seeds, method_lrs, ra_pairs):
            method_field = "lora_head" if train_head else "lora_only" if only_flag else "lora"
            tag = recipe
            if train_head:
                tag = f"{tag}_head"
            elif only_flag:
                tag = f"{tag}_frozen_head"

            cfg = dict(common)
            cfg.update(
                dataset=dataset,
                vit_variant=vit_variant,
                method=method_field,
                seed=int(seed),
                lr=float(lr),
                rank=int(r),
                alpha=int(a),
                init_recipe=recipe,
                chain_recipe=recipe,
                lora_plus=False,
            )
            if cfg.get("chain_every_epochs") is None and cfg.get("chain_every_steps") is None:
                cfg["chain_every_epochs"] = 1

            base = f"{dataset}_vit-{vit_variant}_{tag}{tm_tag}_r{int(r)}_a{int(a)}"
            cfg["group_name"] = base
            cfg["run_name"] = f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}"
            grid.append(cfg)

    return grid


def _assert_unique_run_names(grid: List[Dict[str, Any]]) -> None:
    seen: Dict[str, int] = {}
    dups: List[str] = []
    for cfg in grid:
        rn = str(cfg.get("run_name"))
        if rn in seen:
            dups.append(rn)
        else:
            seen[rn] = 1
    if dups:
        show = "\n".join(f"  - {x}" for x in sorted(set(dups))[:25])
        raise RuntimeError(f"Duplicate run_name values detected (would overwrite outputs):\n{show}")


# ---------------------------
# Runner
# ---------------------------

def _resolve_norm_stats(cfg: Dict[str, Any], *, model_id: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    model_id_l = str(model_id).lower()
    in21k_like = ("in21k" in model_id_l) or ("imagenet21k" in model_id_l)

    default_mean = (0.5, 0.5, 0.5) if in21k_like else (0.485, 0.456, 0.406)
    default_std = (0.5, 0.5, 0.5) if in21k_like else (0.229, 0.224, 0.225)

    if cfg.get("backbone", "vit") == "convit":
        try:
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            return tuple(IMAGENET_DEFAULT_MEAN), tuple(IMAGENET_DEFAULT_STD)
        except Exception:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if bool(cfg.get("pretrained", True)) and AutoImageProcessor is not None:
        try:
            try:
                proc = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            except TypeError:
                proc = AutoImageProcessor.from_pretrained(model_id)
            mean = getattr(proc, "image_mean", None)
            std = getattr(proc, "image_std", None)
            if mean is not None and std is not None and len(mean) == 3 and len(std) == 3:
                return tuple(float(x) for x in mean), tuple(float(x) for x in std)
        except Exception:
            pass

    return default_mean, default_std


def _normalize_scheduler(s: Any) -> Any:
    if s is None:
        return None
    ss = str(s).lower().strip()
    if ss in {"none", "off", "false", ""}:
        return None
    return ss


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["fast_mode"] = bool(cfg.get("fast_mode", False))

    if cfg.get("chain_recipe"):
        init_norm = norm_recipe(cfg.get("init_recipe"))
        chain_norm = norm_recipe(cfg.get("chain_recipe"))
        if init_norm != chain_norm:
            raise ValueError(
                f"Chain requires init==chain recipe: init={cfg.get('init_recipe')} vs chain={cfg.get('chain_recipe')}"
            )

    set_all_seeds(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cuda_fastpath(device)

    model_id = resolve_pretrained_id(cfg["vit_variant"])
    norm_mean, norm_std = _resolve_norm_stats(cfg, model_id=model_id)

    dm = get_data(cfg, norm_mean=norm_mean, norm_std=norm_std)

    if cfg.get("backbone", "vit") == "convit":
        if ConViTImageModel is None:
            raise RuntimeError("ConViT requested but not available in this repo.")
        model = ConViTImageModel(
            variant=cfg["vit_variant"],
            num_classes=dm.num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
        )
    else:
        model = ViTForImageClassification(
            variant=cfg["vit_variant"],
            num_classes=dm.num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
        )

    from tasks.vision_image import VisionClassificationTask

    task_logic = VisionClassificationTask(
        num_classes=dm.num_classes,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        primary_metric=str(cfg.get("primary_metric", "loss")),
        maximize=bool(cfg.get("maximize", False)),
        track_mcc=bool(cfg.get("track_mcc", False)),
    )

    target_modules = cfg.get("target_modules")

    if cfg["method"] == "full":
        wd = float(cfg.get("weight_decay_full", 0.05))
        method = FullFineTune(lr=float(cfg["lr"]), weight_decay=wd)

    elif cfg["method"] == "heads":
        wd = float(cfg.get("weight_decay_full", 0.05))
        method = HeadOnlyFineTune(lr=float(cfg["lr"]), weight_decay=wd)

    elif cfg["method"] in ("paca", "paca_head"):
        wd = float(cfg.get("weight_decay_lora", 0.0))
        method = _construct_method_with_optional_target(
            PaCAQV,
            target_modules=target_modules,
            r=int(cfg["rank"]),
            alpha=int(cfg["alpha"]),
            seed=int(cfg["seed"]),
            k_per_row=cfg.get("paca_k_per_row", None),
            lr=float(cfg["lr"]),
            weight_decay=wd,
            train_head=(cfg["method"] == "paca_head"),
            recipe=cfg.get("init_recipe"),
        )
    else:
        wd = float(cfg.get("weight_decay_lora", 0.0))
        if cfg["method"] == "lora_only":
            MethodClass = LoRAQVLoRAOnly
        elif cfg["method"] == "lora_head":
            MethodClass = LoRAQVWithHead
        else:
            MethodClass = LoRAQV

        method = _construct_method_with_optional_target(
            MethodClass,
            target_modules=target_modules,
            r=int(cfg["rank"]),
            alpha=int(cfg["alpha"]),
            recipe=(cfg.get("init_recipe") or "base"),
            lr=float(cfg["lr"]),
            weight_decay=wd,
            mask_alpha=float(cfg.get("ba_alpha", 0.5)),
            mask_ties_keep=(str(cfg.get("ba_ties", "keep")) == "keep"),
            mask_rounding=str(cfg.get("ba_rounding", "ceil")),
            mask_recompute_every=int(cfg.get("ba_recompute_every", 1)),
        )

        if cfg.get("lora_plus"):
            try:
                setattr(method, "plus_gamma", float(cfg.get("lora_plus_gamma", 16.0)))
            except Exception:
                pass

    callbacks: List[Any] = []

    # Early stopping (only meaningful when not in fast_mode)
    if (not cfg.get("fast_mode", False)) and bool(cfg.get("enable_early_stopping", True)):
        pat = cfg.get("patience", None)
        if pat is not None and int(pat) > 0:
            _add_callback(
                callbacks,
                EarlyStoppingCallback,
                strict=False,
                patience=int(pat),
                min_delta=float(cfg.get("min_delta", 0.0)),
            )
            
    # FLOPs logging
    if bool(cfg.get("enable_flops", False)):
        _add_callback(callbacks, FlopsCounterCallback, strict=False)

    if bool(cfg.get("enable_cuda_peak_memory", False)):
        _add_callback(callbacks, CudaPeakMemoryCallback, strict=False)
        
    if cfg.get("chain_recipe") and (cfg.get("chain_every_steps") or cfg.get("chain_every_epochs")):
        _add_callback(
            callbacks,
            ChainResetCallback,
            strict=True,
            recipe=cfg["chain_recipe"],
            every_n_opt_steps=cfg.get("chain_every_steps"),
            every_n_epochs=cfg.get("chain_every_epochs"),
        )

    if cfg.get("init_recipe") == "ba_sparse_final":
        # Force front of callback list for masking behavior.
        tmp: List[Any] = []
        _add_callback(
            tmp,
            BASparseFinalCallback,
            strict=False,
            eval_masked_test=bool(cfg.get("ba_eval_masked_test", True)),
            respect_save_json_only=bool(cfg.get("save_json_only", False)),
        )
        callbacks = tmp + callbacks

    if cfg.get("init_recipe") in ("ba_sparse_lora", "ba_sparse_final", "ba_sparse_fix_mask"):
        _add_callback(callbacks, BASparsityLoggerCallback, strict=False, numeric_eps=1e-6)

    if cfg.get("wandb", False):
        log_every = _wandb_log_every_value(cfg.get("wandb_log_every"))
        _add_callback(
            callbacks,
            WandbCallback,
            strict=True,
            project=cfg.get("wandb_project") or "lora-vision",
            entity=cfg.get("wandb_entity"),
            group=cfg.get("wandb_group") or cfg.get("group_name") or f"{cfg['dataset']}-vit-{cfg['vit_variant']}-{cfg['method']}",
            mode=cfg.get("wandb_mode"),
            run_name=cfg.get("run_name"),
            tags=cfg.get("wandb_tags"),
            config={
                "sweep_params": cfg,
                "model_id": model_id,
                "norm_mean": list(norm_mean),
                "norm_std": list(norm_std),
            },
            log_every_n_steps=log_every,
            watch_model=bool(cfg.get("wandb_watch", False)),
            upload_artifacts=bool(cfg.get("wandb_upload_artifacts", False)),
            out_dir=cfg.get("out_dir"),
        )

    sparse_recipe = cfg.get("init_recipe") in ("ba_sparse_lora", "ba_sparse_final", "ba_sparse_fix_mask")
    use_amp_flag = False if sparse_recipe else bool(cfg.get("use_amp", True))
    prefer_bf16_flag = False if sparse_recipe else cfg.get("prefer_bf16", None)

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # --------- map old flags -> TrainConfig fields (ONLY those that exist) ----------
    save_json_only = bool(cfg.get("save_json_only", True))
    save_best_only = bool(cfg.get("save_best_only", False))

    if save_json_only:
        save_checkpoints = False
        save_best_to_disk = False
        save_all_epochs = False
        save_init_checkpoint = False
    else:
        save_checkpoints = True
        save_best_to_disk = True
        save_all_epochs = bool(not save_best_only)
        save_init_checkpoint = True

    sched = _normalize_scheduler(cfg.get("scheduler", "cosine"))

    base_tcfg_kwargs = dict(
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay_full", 0.0) if cfg["method"] in {"full", "heads"} else cfg.get("weight_decay_lora", 0.0)),
        use_amp=bool(use_amp_flag),
        prefer_bf16=prefer_bf16_flag,
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
        save_dir=cfg["out_dir"],
        save_name_prefix=cfg["run_name"],
        train_log_interval=float(cfg["train_log_interval"]),
        eval_log_interval=float(cfg["eval_log_interval"]),
        limit_train_batches=cfg.get("limit_train_batches"),
        limit_eval_batches=cfg.get("limit_eval_batches"),
        seed=int(cfg["seed"]),
        scheduler=sched,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        min_lr=cfg.get("min_lr"),
        save_checkpoints=bool(save_checkpoints),
        save_all_epochs=bool(save_all_epochs),
        save_init_checkpoint=bool(save_init_checkpoint),
        save_best_to_disk=bool(save_best_to_disk),
    )

    mode_kwargs_raw = trainer_mode_kwargs(fast_mode=bool(cfg.get("fast_mode", False)))
    mode_kwargs = _filter_trainconfig_kwargs(mode_kwargs_raw if isinstance(mode_kwargs_raw, dict) else {})

    tcfg = TrainConfig(**_filter_trainconfig_kwargs({**base_tcfg_kwargs, **mode_kwargs}))

    test_loader = dm.test_dataloader()
    if cfg.get("fallback_test_from_val", False) and (test_loader is None):
        test_loader = dm.val_dataloader()

    trainer = GenericTrainer(
        model=model,
        method=method,
        task_logic=task_logic,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        test_loader=test_loader,
        device=device,
        tcfg=tcfg,
        method_cfg_lr_override=float(cfg["lr"]),
        callbacks=callbacks,
        data_module=dm,
    )

    best_path, best_score = trainer.fit(cfg["run_name"])

    metrics_json = os.path.join(cfg["out_dir"], cfg["run_name"], f"{cfg['run_name']}-metrics.json")
    summary = None
    try:
        with open(metrics_json, "r") as f:
            summary = json.load(f)
    except Exception:
        pass

    return {
        "sweep_params": cfg,
        "primary_metric": trainer.primary_metric,
        "best_score": float(best_score),
        "best_checkpoint": best_path,
        "maximize": bool(trainer.maximize),
        "metrics_json": metrics_json if summary is not None else None,
        "summary": summary,
        "model_id": model_id,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = CommentArgParser(description="Vision sweep for ViT with Full/Heads/LoRA/PaCA.")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-api-key", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb-log-every", type=int, default=None)
    p.add_argument("--wandb-watch", action="store_true")
    p.add_argument("--wandb-upload-artifacts", action="store_true")

    # run controls
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run directory/metrics.")

    # training mode
    add_bool_arg(p, "--fast-mode", default=True, help="Timing/fast mode (reduced eval/logging for speed).")
    add_bool_arg(p, "--save-json-only", default=True, help="If true, disable checkpoint saving (metrics JSON only).")
    add_bool_arg(p, "--save-best-only", default=False, help="If true, keep only best checkpoint (if saving enabled).")
    add_bool_arg(p, "--enable-flops", default=False, help="Log FLOPs (adds FlopsCounterCallback if available).")

    # task metric
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--primary-metric", type=str, default="loss")
    add_bool_arg(p, "--maximize", default=False, help="If true, higher primary-metric is better.")

    # what to run
    p.add_argument("--datasets", type=str, nargs="*", default=None, choices=DATASETS_ALL)
    p.add_argument("--dataset", type=str, default=None, choices=DATASETS_ALL)
    p.add_argument("--methods", type=str, nargs="+", required=True)
    p.add_argument("--method", type=str, default=None)

    # model
    p.add_argument("--vit-variant", type=str, default="tiny", choices=VIT_VARIANTS)
    p.add_argument("--backbone", type=str, default="vit", choices=["vit", "convit"])
    p.add_argument("--pretrained", type=str2bool, default=True)

    # targets
    p.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="Adapter targets. Example: 'q,v' or 'qkv' or 'attn' or 'all' or 'query,key,value,dense'.",
    )

    # sweep axes
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-4])
    p.add_argument("--lr-full", type=float, default=None)
    p.add_argument("--lr-others", type=float, default=None)
    p.add_argument("--ranks", type=int, nargs="*", default=[8])
    p.add_argument("--alphas", type=int, nargs="*", default=None)
    p.add_argument("--chain-every-epochs", type=int, default=None)
    p.add_argument("--chain-every-steps", type=int, default=None)

    # PaCA knob
    p.add_argument("--paca-k-per-row", type=int, default=None)

    # dataset URLs
    p.add_argument("--officehome-url", type=str, default=None)
    p.add_argument("--terraincognita-url", type=str, default=None)

    # data / training
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--pretrain-data-root", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--train-log-interval", type=float, default=1.0)
    p.add_argument("--eval-log-interval", type=float, default=1.0)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-eval-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=None)
    add_bool_arg(p, "--fallback-test-from-val", default=True, help="If test loader missing, reuse val loader.")
    add_bool_arg(p, "--enable-cuda-peak-memory", default=False)


    # aug knobs
    p.add_argument("--randaugment", action="store_true")
    p.add_argument("--ra-n", type=int, default=2)
    p.add_argument("--ra-m", type=int, default=9)
    p.add_argument("--random-erasing", type=float, default=0.0)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=0.0)
    p.add_argument("--mix-prob", type=float, default=0.0)
    p.add_argument("--switch-prob", type=float, default=0.5)

    # scheduler
    p.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "none"])
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.05)

    # wd
    p.add_argument("--weight-decay-full", type=float, default=0.05)
    p.add_argument("--weight-decay-lora", type=float, default=0.0)

    # output
    p.add_argument("--out-dir", type=str, default=None)

    # LoRA+
    p.add_argument("--lora-plus-gamma", type=float, default=16.0)

    # BA mask knobs
    p.add_argument("--ba-alpha", type=float, default=0.5)
    p.add_argument("--ba-alpha-is-sparsity", action="store_true")
    p.add_argument("--ba-sparsity", type=float, default=None)
    p.add_argument("--ba-ties", type=str, default="keep", choices=["keep", "drop"])
    p.add_argument("--ba-rounding", type=str, default="ceil", choices=["ceil", "floor"])
    p.add_argument("--ba-recompute-every", type=int, default=1)
    p.add_argument("--ba-eval-masked-test", type=str2bool, default=True)

    args = p.parse_args()

    # env fallbacks
    args.wandb_project = resolve_cli_env(args.wandb_project, "WANDB_PROJECT", "lora-vision")
    args.wandb_entity = resolve_cli_env(args.wandb_entity, "WANDB_ENTITY", None)
    args.wandb_group = resolve_cli_env(args.wandb_group, "WANDB_GROUP", None)
    args.wandb_mode = resolve_cli_env(args.wandb_mode, "WANDB_MODE", None)
    args.wandb_log_every = resolve_cli_env(args.wandb_log_every, "WANDB_LOG_EVERY", 200)

    args.out_dir = resolve_cli_env(args.out_dir, "LORA_OUT_DIR", "sweep_out")
    args.data_root = resolve_cli_env(args.data_root, "LORA_DATA_ROOT", None)
    args.pretrain_data_root = resolve_cli_env(args.pretrain_data_root, "LORA_PRETRAIN_DATA_ROOT", None)

    args.officehome_url = resolve_cli_env(args.officehome_url, "OFFICEHOME_URL", None)
    args.terraincognita_url = resolve_cli_env(args.terraincognita_url, "TERRAINCOGNITA_URL", None)

    resolved_key = resolve_cli_env(args.wandb_api_key, "WANDB_API_KEY", None)
    if resolved_key:
        os.environ["WANDB_API_KEY"] = str(resolved_key)

    if args.data_root is None:
        p.error("--data-root not provided and LORA_DATA_ROOT env var is not set.")

    args.out_dir = str(Path(args.out_dir).resolve())
    args.data_root = str(Path(args.data_root).resolve())
    if args.pretrain_data_root is not None:
        args.pretrain_data_root = str(Path(args.pretrain_data_root).resolve())

    args.wandb_log_every = _wandb_log_every_value(args.wandb_log_every)

    args.target_modules = normalize_target_modules(args.target_modules)

    ds: List[str] = []
    if args.datasets:
        ds.extend(args.datasets)
    if args.dataset:
        ds.append(args.dataset)
    if not ds:
        p.error("Please supply --dataset or --datasets.")
    args.datasets = sorted(set(ds))

    ms: List[str] = []
    if args.methods:
        ms.extend(args.methods)
    if args.method:
        ms.append(args.method)
    ms = [m.lower() for m in ms]
    ms = _normalize_method_aliases(ms)
    args.methods = _dedupe_preserve_order(ms)

    _require_nonempty("seeds", args.seeds)
    _require_nonempty("lrs", args.lrs)
    _require_nonempty("ranks", args.ranks)
    if args.alphas is not None:
        _require_nonempty("alphas", args.alphas)

    # BA alpha resolution
    if args.ba_sparsity is not None:
        args.ba_alpha = args.ba_sparsity
        args.ba_alpha_is_sparsity = True

    keep, s = resolve_ba_alpha(args.ba_alpha, bool(args.ba_alpha_is_sparsity))
    print(
        f"[BA] Using per-axis keep α={keep:.4f} (target C-sparsity ≈ {s:.3f}) from input={args.ba_alpha} "
        f"mode={'sparsity' if args.ba_alpha_is_sparsity else 'keep'}"
    )
    args.ba_alpha_eff = float(keep)
    args.ba_target_sparsity = float(s)

    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    lrs_by_method = None
    if args.lr_full is not None or args.lr_others is not None:
        lrs_by_method = {}
        if args.lr_full is not None:
            lrs_by_method["full"] = [float(args.lr_full)]
        if args.lr_others is not None:
            other = [float(args.lr_others)]
            for m in (
                ["heads", "lora", "lora_only", "lora_head", "lora_plus", "lora+"]
                + NONCHAIN_RECIPES
                + CHAIN_RECIPES
                + ["paca", "paca_head", "dpaca", "cpaca", "dcpaca", "dpaca_head", "cpaca_head", "dcpaca_head"]
            ):
                lrs_by_method[m] = other

    common: Dict[str, Any] = {
        "fast_mode": bool(args.fast_mode),
        "epochs": int(args.epochs),
        "grad_accum_steps": int(args.grad_accum_steps),
        "batch_size": int(args.batch_size),
        "img_size": int(args.img_size),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "train_log_interval": float(args.train_log_interval),
        "eval_log_interval": float(args.eval_log_interval),
        "limit_train_batches": args.limit_train_batches,
        "limit_eval_batches": args.limit_eval_batches,
        "num_workers": int(args.num_workers),
        "val_split": args.val_split,
        "fallback_test_from_val": bool(args.fallback_test_from_val),
        "target_modules": args.target_modules,
        "out_dir": args.out_dir,
        "data_root": args.data_root,
        "pretrain_data_root": str(Path((args.pretrain_data_root or args.data_root)).resolve()),
        "officehome_url": (None if not args.officehome_url else str(args.officehome_url)),
        "terraincognita_url": (None if not args.terraincognita_url else str(args.terraincognita_url)),
        "wandb": bool(args.wandb),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
        "wandb_mode": args.wandb_mode,
        "wandb_tags": args.wandb_tags,
        "wandb_log_every": int(args.wandb_log_every),
        "wandb_watch": bool(args.wandb_watch),
        "wandb_upload_artifacts": bool(args.wandb_upload_artifacts),
        "chain_every_epochs": args.chain_every_epochs,
        "chain_every_steps": args.chain_every_steps,
        "weight_decay_full": float(args.weight_decay_full),
        "weight_decay_lora": float(args.weight_decay_lora),
        "pretrained": bool(args.pretrained),
        "backbone": args.backbone,
        "randaugment": bool(args.randaugment),
        "ra_n": int(args.ra_n),
        "ra_m": int(args.ra_m),
        "random_erasing": float(args.random_erasing),
        "mixup_alpha": float(args.mixup_alpha),
        "cutmix_alpha": float(args.cutmix_alpha),
        "mix_prob": float(args.mix_prob),
        "switch_prob": float(args.switch_prob),
        "scheduler": (None if args.scheduler == "none" else args.scheduler),
        "min_lr": args.min_lr,
        "warmup_ratio": float(args.warmup_ratio),
        "save_json_only": bool(args.save_json_only),
        "save_best_only": bool(args.save_best_only),
        "lora_plus_gamma": float(args.lora_plus_gamma),
        "paca_k_per_row": args.paca_k_per_row,
        "ba_alpha": float(args.ba_alpha_eff),
        "ba_ties": str(args.ba_ties),
        "ba_rounding": str(args.ba_rounding),
        "ba_recompute_every": int(args.ba_recompute_every),
        "ba_eval_masked_test": bool(args.ba_eval_masked_test),
        "ba_target_sparsity": float(args.ba_target_sparsity),
        "ba_alpha_input": float(args.ba_alpha),
        "ba_alpha_input_mode": "sparsity" if bool(args.ba_alpha_is_sparsity) else "keep_frac",
        "enable_early_stopping": bool(not args.fast_mode),
        "label_smoothing": float(args.label_smoothing),
        "primary_metric": str(args.primary_metric),
        "maximize": bool(args.maximize),
        "enable_flops": bool(args.enable_flops),
        "enable_cuda_peak_memory": bool(args.enable_cuda_peak_memory),
    }

    grid: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        grid.extend(
            build_run_cfgs_for_dataset(
                dataset=dataset,
                vit_variant=args.vit_variant,
                methods=args.methods,
                seeds=args.seeds,
                lrs=args.lrs,
                ranks=args.ranks,
                alphas=args.alphas,
                common=dict(common),
                lrs_by_method=lrs_by_method,
            )
        )

    _assert_unique_run_names(grid)

    print(f"[PLAN] datasets={args.datasets}, methods={args.methods}, vit_variant={args.vit_variant}, runs={len(grid)}")
    if args.dry_run:
        for r in grid[:25]:
            print("  ", r["run_name"])
        if len(grid) > 25:
            print(f"  ... ({len(grid)-25} more)")
        return

    results_path = os.path.join(args.out_dir, "sweep_vision_results.jsonl")
    all_runs = []

    with open(results_path, "a") as f:
        for i, cfg in enumerate(grid, 1):
            rn = cfg["run_name"]

            exists = metrics_exist(cfg["out_dir"], rn)
            if exists and args.resume:
                print(f"[SKIP] {i}/{len(grid)} exists -> {rn}")
                continue
            if exists and (not args.overwrite):
                raise RuntimeError(
                    f"Run already exists: {rn}\n"
                    f"Use --resume to skip existing runs or --overwrite to force re-run/overwrite."
                )

            print(f"[RUN] {i}/{len(grid)} -> {rn}")
            out = run_one(cfg)
            f.write(json.dumps(out) + "\n")
            f.flush()
            all_runs.append(out)

    key_fields = [
        "dataset",
        "vit_variant",
        "method",
        "rank",
        "alpha",
        "lr",
        "target_modules",
        "lora_plus",
        "lora_plus_gamma",
        "paca_k_per_row",
        "init_recipe",
        "chain_recipe",
        "chain_every_steps",
        "chain_every_epochs",
        "batch_size",
        "img_size",
        "epochs",
        "grad_accum_steps",
        "fast_mode",
        "save_json_only",
        "save_best_only",
        "scheduler",
        "min_lr",
        "warmup_ratio",
    ]
    summary, top1 = aggregate_and_choose(all_runs, args.out_dir, key_fields)
    if top1:
        print("\n[SWEEP][BEST-BY-MEAN]")
        print(json.dumps(top1, indent=2))
    print(f"[SWEEP] Done. Per-run results -> {results_path}")


if __name__ == "__main__":
    main()