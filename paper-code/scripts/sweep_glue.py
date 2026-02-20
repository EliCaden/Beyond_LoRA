# scripts/sweep_glue.py
from __future__ import annotations

import dataclasses
import itertools
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

# --- repo-local imports (force repo root onto sys.path) ---
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

from data.glue import GLUEDataModule
from tasks.glue_text import GlueTextTask
from models.roberta import RobertaGLUEModel

from methods import FullFineTune, HeadOnlyFineTune, LoRAQV, LoRAQVLoRAOnly, LoRAQVWithHead
from methods.paca_qv import PaCAQV

from trainers.generic_trainer import GenericTrainer, TrainConfig

from trainers.callbacks import (
    ChainResetCallback,
    WandbCallback,
    BASparseFinalCallback,
    BASparsityLoggerCallback,
)

# Optional callbacks (don’t hard-fail if not present)
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
try:
    from trainers.callbacks import ZeroShotEvalCallback  # type: ignore
except Exception:
    ZeroShotEvalCallback = None  # type: ignore


GLUE_ALL = ["sst2", "cola", "mrpc", "rte", "qqp", "qnli", "mnli", "stsb", "wnli"]
ROBERTA_VARIANTS = ["base", "large"]

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

_TOKENIZER_CACHE: dict[str, Any] = {}


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
    """
    Canonicalize method spellings early to avoid duplicate run names & confusing grids.
    - base -> lora (and base_* -> lora_*)
    - paca_qv -> paca
    """
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
    """
    Preserve 0 (don't coerce to default).
    """
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


def _normalize_scheduler(s: Any) -> Any:
    if s is None:
        return None
    ss = str(s).lower().strip()
    if ss in {"none", "off", "false", ""}:
        return None
    return ss


# ---------------------------
# Grid builder
# ---------------------------

def build_run_cfgs_for_task(
    task: str,
    backbone: str,
    variant: str,
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
                base = f"{task}_roberta-{variant}_full{tm_tag}"
                cfg.update(
                    task=task,
                    backbone=backbone,
                    variant=variant,
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
                base = f"{task}_roberta-{variant}_heads{tm_tag}"
                cfg.update(
                    task=task,
                    backbone=backbone,
                    variant=variant,
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
                    task=task,
                    backbone=backbone,
                    variant=variant,
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
                base = f"{task}_roberta-{variant}_{tag}{ktag}{tm_tag}_r{int(r)}_a{int(a)}"
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
                    task=task,
                    backbone=backbone,
                    variant=variant,
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
                base = f"{task}_roberta-{variant}_{tag}{tm_tag}_r{int(r)}_a{int(a)}"
                cfg["group_name"] = base
                cfg["run_name"] = f"{base}_lr{_fmt_float(lr)}_seed{int(seed)}"
                grid.append(cfg)
            continue

        # Chain
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
                task=task,
                backbone=backbone,
                variant=variant,
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

            base = f"{task}_roberta-{variant}_{tag}{tm_tag}_r{int(r)}_a{int(a)}"
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

def _get_tokenizer(backbone: str):
    tok = _TOKENIZER_CACHE.get(backbone)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        _TOKENIZER_CACHE[backbone] = tok
    return tok


def _glue_dm_kwargs(**kwargs):
    out = {}
    try:
        sig = inspect.signature(GLUEDataModule.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set(kwargs.keys())

    for k, v in kwargs.items():
        if k in allowed:
            out[k] = v
    return out


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["fast_mode"] = bool(cfg.get("fast_mode", False))

    # chain coherency
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

    model = RobertaGLUEModel(task=cfg["task"], variant=cfg["variant"])
    tokenizer = _get_tokenizer(cfg["backbone"])

    dm = GLUEDataModule(
        **_glue_dm_kwargs(
            task_name=cfg["task"],
            tokenizer=tokenizer,
            batch_size=int(cfg["batch_size"]),
            max_length=int(cfg["max_length"]),
            seed=int(cfg["seed"]),
            use_length_bucketing=bool(cfg.get("use_length_bucketing", True)),
            fallback_test_from_val=bool(cfg.get("fallback_test_from_val", True)),
            tokenize_num_proc=cfg.get("tokenize_num_proc", None),
        )
    )
    dm.setup()

    task_logic = GlueTextTask(cfg["task"])

    wd_full = float(cfg.get("weight_decay_full", 0.01))
    wd_lora = float(cfg.get("weight_decay_lora", 0.0))

    target_modules = cfg.get("target_modules")

    if cfg["method"] == "full":
        wd = wd_full
        method = FullFineTune(lr=float(cfg["lr"]), weight_decay=float(wd))

    elif cfg["method"] == "heads":
        wd = wd_full
        method = HeadOnlyFineTune(lr=float(cfg["lr"]), weight_decay=float(wd))

    elif cfg["method"] in ("paca", "paca_head"):
        wd = wd_lora
        method = _construct_method_with_optional_target(
            PaCAQV,
            target_modules=target_modules,
            r=int(cfg["rank"]),
            alpha=int(cfg["alpha"]),
            seed=int(cfg["seed"]),
            k_per_row=cfg.get("paca_k_per_row", None),
            lr=float(cfg["lr"]),
            weight_decay=float(wd),
            train_head=(cfg["method"] == "paca_head"),
            recipe=cfg.get("init_recipe"),
        )
    else:
        wd = wd_lora
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
            weight_decay=float(wd),
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

    if bool(cfg.get("enable_flops", False)):
        _add_callback(callbacks, FlopsCounterCallback, strict=False)

    if bool(cfg.get("enable_cuda_peak_memory", False)):
        _add_callback(callbacks, CudaPeakMemoryCallback, strict=False)

    if bool(cfg.get("enable_zero_shot_eval", False)):
        _add_callback(
            callbacks,
            ZeroShotEvalCallback,
            strict=False,
            # leave kwargs flexible; filtered by signature
            task=cfg.get("task"),
        )
        
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
            project=cfg.get("wandb_project") or "lora-glue",
            entity=cfg.get("wandb_entity"),
            group=cfg.get("wandb_group") or cfg.get("group_name") or f"{cfg['task']}-roberta-{cfg['variant']}-{cfg['method']}",
            mode=cfg.get("wandb_mode"),
            run_name=cfg.get("run_name"),
            tags=cfg.get("wandb_tags"),
            config={"sweep_params": cfg},
            log_every_n_steps=log_every,
            watch_model=bool(cfg.get("wandb_watch", False)),
            upload_artifacts=bool(cfg.get("wandb_upload_artifacts", False)),
            out_dir=cfg.get("out_dir"),
        )

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

    sched = _normalize_scheduler(cfg.get("scheduler", "linear"))

    base_tcfg_kwargs = dict(
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        weight_decay=float(wd),
        use_amp=bool(cfg.get("use_amp", True)),
        prefer_bf16=cfg.get("prefer_bf16", None),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
        save_dir=cfg["out_dir"],
        save_name_prefix=cfg["run_name"],
        train_log_interval=float(cfg["train_log_interval"]),
        eval_log_interval=float(cfg["eval_log_interval"]),
        limit_train_batches=cfg.get("limit_train_batches"),
        limit_eval_batches=cfg.get("limit_eval_batches"),
        seed=int(cfg["seed"]),
        scheduler=sched,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
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
    }


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = CommentArgParser(description="GLUE sweep for RoBERTa with Full/Heads/LoRA/PaCA.")

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
    p.add_argument("--overwrite", action="store_true")

    # training mode
    add_bool_arg(p, "--fast-mode", default=True)
    add_bool_arg(p, "--save-json-only", default=True)
    add_bool_arg(p, "--save-best-only", default=False)

    # tasks
    p.add_argument("--tasks", type=str, nargs="*", default=None, choices=GLUE_ALL)
    p.add_argument("--task", type=str, default=None, choices=GLUE_ALL)

    # model selection
    p.add_argument("--variant", type=str, default="base", choices=ROBERTA_VARIANTS)
    p.add_argument("--backbone", type=str, default=None, help="HF model id (default from --variant).")

    # targets
    p.add_argument("--target-modules", type=str, default=None)

    # methods / sweep axes
    p.add_argument("--methods", type=str, nargs="+", required=True)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lrs", type=float, nargs="+", default=[2e-4])
    p.add_argument("--ranks", type=int, nargs="*", default=[8])
    p.add_argument("--alphas", type=int, nargs="*", default=None)
    p.add_argument("--chain-every-epochs", type=int, default=None)
    p.add_argument("--chain-every-steps", type=int, default=None)
    p.add_argument("--paca-k-per-row", type=int, default=None)

    # data/training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--tokenize-num-proc", type=int, default=None)
    add_bool_arg(p, "--use-length-bucketing", default=True)
    add_bool_arg(p, "--fallback-test-from-val", default=True)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--train-log-interval", type=float, default=1.0)
    p.add_argument("--eval-log-interval", type=float, default=1.0)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-eval-batches", type=int, default=None)

    # scheduler
    p.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.1)

    # wd
    p.add_argument("--weight-decay-full", type=float, default=0.01)
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

    # optional diagnostics callbacks
    add_bool_arg(p, "--enable-flops", default=False)
    add_bool_arg(p, "--enable-cuda-peak-memory", default=False)
    add_bool_arg(p, "--enable-zero-shot-eval", default=False)

    args = p.parse_args()

    # env fallbacks
    args.wandb_project = resolve_cli_env(args.wandb_project, "WANDB_PROJECT", "lora-glue")
    args.wandb_entity = resolve_cli_env(args.wandb_entity, "WANDB_ENTITY", None)
    args.wandb_group = resolve_cli_env(args.wandb_group, "WANDB_GROUP", None)
    args.wandb_mode = resolve_cli_env(args.wandb_mode, "WANDB_MODE", None)
    args.wandb_log_every = resolve_cli_env(args.wandb_log_every, "WANDB_LOG_EVERY", 200)

    args.out_dir = resolve_cli_env(args.out_dir, "LORA_OUT_DIR", "sweep_out")

    resolved_key = resolve_cli_env(args.wandb_api_key, "WANDB_API_KEY", None)
    if resolved_key:
        os.environ["WANDB_API_KEY"] = str(resolved_key)

    args.out_dir = str(Path(args.out_dir).resolve())
    args.wandb_log_every = _wandb_log_every_value(args.wandb_log_every)

    # default backbone from variant
    if args.backbone is None:
        args.backbone = "roberta-base" if args.variant == "base" else "roberta-large"

    # tasks
    ts: List[str] = []
    if args.tasks:
        ts.extend(args.tasks)
    if args.task:
        ts.append(args.task)
    if not ts:
        p.error("Please supply --task or --tasks.")
    args.tasks = sorted(set(ts))

    ms: List[str] = []
    if args.methods:
        ms.extend(args.methods)
    if args.method:
        ms.append(args.method)
    ms = [m.lower() for m in ms]
    ms = _normalize_method_aliases(ms)
    args.methods = _dedupe_preserve_order(ms)

    args.target_modules = normalize_target_modules(args.target_modules)

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

    common: Dict[str, Any] = {
        "fast_mode": bool(args.fast_mode),
        "epochs": int(args.epochs),
        "grad_accum_steps": int(args.grad_accum_steps),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "tokenize_num_proc": args.tokenize_num_proc,
        "use_length_bucketing": bool(args.use_length_bucketing),
        "fallback_test_from_val": bool(args.fallback_test_from_val),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "train_log_interval": float(args.train_log_interval),
        "eval_log_interval": float(args.eval_log_interval),
        "limit_train_batches": args.limit_train_batches,
        "limit_eval_batches": args.limit_eval_batches,
        "target_modules": args.target_modules,
        "out_dir": args.out_dir,
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
        "enable_early_stopping": bool(not args.fast_mode),
        "enable_flops": bool(args.enable_flops),
        "enable_cuda_peak_memory": bool(args.enable_cuda_peak_memory),
        "enable_zero_shot_eval": bool(args.enable_zero_shot_eval),
    }

    grid: List[Dict[str, Any]] = []
    for task in args.tasks:
        grid.extend(
            build_run_cfgs_for_task(
                task=task,
                backbone=str(args.backbone),
                variant=str(args.variant),
                methods=args.methods,
                seeds=args.seeds,
                lrs=args.lrs,
                ranks=args.ranks,
                alphas=args.alphas,
                common=dict(common),
                lrs_by_method=None,
            )
        )

    _assert_unique_run_names(grid)

    print(f"[PLAN] tasks={args.tasks}, methods={args.methods}, variant={args.variant}, runs={len(grid)}")
    if args.dry_run:
        for r in grid[:25]:
            print("  ", r["run_name"])
        if len(grid) > 25:
            print(f"  ... ({len(grid)-25} more)")
        return

    results_path = os.path.join(args.out_dir, "sweep_glue_results.jsonl")
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
        "task",
        "variant",
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
        "max_length",
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