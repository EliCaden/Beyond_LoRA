# scripts/sweep_common.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch

T = TypeVar("T")


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def str2bool(v: Union[bool, str, int]) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else None


def resolve_cli_env(cli_val: Optional[T], env_key: str, default: T) -> T:
    """
    Resolution precedence: CLI value > ENV var > default.
    Note: ENV returns strings; callers should cast/parse as needed.
    """
    if cli_val is not None:
        return cli_val
    v = env_get(env_key, None)
    if v is None:
        return default
    return cast(T, v)


def add_bool_arg(p: argparse.ArgumentParser, flag: str, default: bool, help: Optional[str] = None) -> None:
    if not flag.startswith("--"):
        raise ValueError(f"flag must start with '--': got {flag!r}")
    dest = flag[2:].replace("-", "_")

    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(flag, dest=dest, default=default, action=argparse.BooleanOptionalAction, help=help)
        return

    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(flag, dest=dest, action="store_true", help=help)
    g.add_argument(f"--no-{flag[2:]}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(**{dest: default})


class CommentArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line: str):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith(";"):
            return []
        return shlex.split(s, comments=False, posix=True)


def strip_method_suffixes(tag: str) -> Tuple[str, bool, bool]:
    t = tag.strip().lower()
    train_head = False
    only_flag = False

    changed = True
    while changed:
        changed = False
        if t.endswith("_head"):
            train_head = True
            t = t[:-5]
            changed = True
        if t.endswith("_only"):
            only_flag = True
            t = t[:-5]
            changed = True

    if train_head and only_flag:
        raise ValueError(f"Invalid method tag '{tag}': can't be both _head and _only.")
    return t, train_head, only_flag


def norm_recipe(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    t = name.strip().lower().replace("-", "_")
    if t in {"rac", "rac_a"}:
        return "rac_a"
    if t in {"racb", "rac_b"}:
        return "rac_b"
    if t in {"c3la", "c^3la", "c3_lora", "c3-lora"}:
        return "c3la"
    if t in {"base", "lora"}:
        return "base"
    return t


def build_ra_pairs(ranks: List[int], alphas: Optional[List[int]]) -> List[Tuple[int, int]]:
    if not alphas:
        return [(int(r), int(2 * r)) for r in ranks]
    if len(alphas) == len(ranks):
        return [(int(r), int(a)) for r, a in zip(ranks, alphas)]
    return [(int(r), int(a)) for r in ranks for a in alphas]


def resolve_ba_alpha(alpha: float, as_sparsity: bool) -> Tuple[float, float]:
    a = float(alpha)
    if as_sparsity:
        s = max(0.0, min(1.0, a))
        keep = math.sqrt(max(0.0, 1.0 - s))
        return keep, s
    keep = max(0.0, min(1.0, a))
    s = 1.0 - keep * keep
    return keep, s


def metrics_exist(out_dir: str, run_name: str) -> bool:
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.isdir(run_dir):
        return False
    a = os.path.join(run_dir, f"{run_name}-metrics.json")
    b = os.path.join(run_dir, "metrics.json")
    return os.path.isfile(a) or os.path.isfile(b)


def aggregate_and_choose(
    all_runs: List[Dict[str, Any]],
    out_dir: str,
    key_fields: List[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Group runs by key_fields (pulled from sweep_params), compute mean/std of best_score over seeds,
    and write:
      - sweep_summary.json
      - sweep_top_config.json
    """
    from collections import defaultdict

    def group_key(d: Dict[str, Any]):
        sp = d.get("sweep_params", {}) or {}
        return tuple((k, sp.get(k)) for k in key_fields)

    buckets = defaultdict(list)
    for r in all_runs:
        buckets[group_key(r)].append(r)

    summary: List[Dict[str, Any]] = []

    for k, runs in buckets.items():
        maximize = bool(runs[0].get("maximize", True))
        metric_name = runs[0].get("primary_metric", "acc")

        good: List[Tuple[int, float]] = []
        for rr in runs:
            try:
                v = float(rr.get("best_score", float("nan")))
                if not math.isfinite(v):
                    continue

                sp = rr.get("sweep_params", {}) or {}
                seed_raw = sp.get("seed", -1)
                try:
                    seed = int(seed_raw)
                except Exception:
                    seed = -1

                good.append((seed, v))
            except Exception:
                continue

        if not good:
            continue

        vals = [v for (_, v) in good]
        entry = {
            "config": {kk: vv for kk, vv in k},
            "primary_metric": metric_name,
            "maximize": maximize,
            "mean_best_val": float(np.mean(vals)),
            "std_best_val": float(np.std(vals)),
            "num_seeds": int(len(vals)),
            "seeds": [s for (s, _) in good],
        }
        summary.append(entry)

    if not summary:
        return None, None

    # Guard against mixed maximize/minimize across the aggregated set.
    maximize_set = {bool(e["maximize"]) for e in summary}
    if len(maximize_set) != 1:
        raise ValueError(
            f"aggregate_and_choose(): mixed objective directions in summary: {sorted(maximize_set)}. "
            f"Ensure all runs agree on 'maximize' (or aggregate separately)."
        )

    maximize = bool(summary[0]["maximize"])
    summary.sort(key=lambda e: float(e["mean_best_val"]), reverse=maximize)

    top1 = summary[0]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "sweep_summary.json"), "w") as g:
        json.dump({"groups": summary, "best_by_mean": top1}, g, indent=2)
    with open(os.path.join(out_dir, "sweep_top_config.json"), "w") as g:
        json.dump(top1["config"], g, indent=2)

    return summary, top1


def trainer_mode_kwargs(
    *,
    fast_mode: bool,
    eval_every: int = 0,
    eval_loss_only: bool = True,
    primary_metric_only: bool = True,
    no_gpu_sync: bool = False,
    final_sync_eval: bool = True,
    disable_nonfinite_sync: bool = True,
    disable_early_stopping: bool = True,
) -> Dict[str, Any]:
    if fast_mode:
        ee = int(eval_every)
        if ee < 0:
            ee = 0

        return {
            "fast_mode": True,
            "fast_mode_eval_every": ee,
            "fast_mode_eval_loss_only": bool(eval_loss_only),
            "fast_mode_primary_metric_only": bool(primary_metric_only),
            "fast_mode_no_gpu_sync": bool(no_gpu_sync),
            "fast_mode_final_sync_eval": bool(final_sync_eval),
            "fast_mode_disable_nonfinite_sync": bool(disable_nonfinite_sync),
            "fast_mode_disable_early_stopping": bool(disable_early_stopping),
        }

    # Important: don't override non-fast defaults.
    return {"fast_mode": False}