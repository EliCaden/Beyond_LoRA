# scripts/sweep_e2e.py
# Manual-only sweep runner for GPT-2 on E2E NLG using causal LM (Full FT or LoRA).
# - Accepts --dataset and --data-root (only 'e2e' supported)
# - Avoids HF script datasets (no trust_remote_code): loads CSVs locally or from GitHub
# - Monkey-patches data.e2e._load_e2e so E2EDataModule uses the CSV dataset

import argparse
import json
import os
import random
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

# repo-local import path fix
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401

from data.e2e import E2EDataModule
from models.gpt2_lm import GPT2CausalLM
from methods import FullFineTune, LoRAQV
from tasks.causal_lm import CausalLMTask
from trainers.generic_trainer import GenericTrainer, TrainConfig
from trainers.callbacks import (
    EarlyStoppingCallback,
    ChainResetCallback,
    FlopsCounterCallback,
    WandbCallback,
    BestEpochTrainLossCallback,
    BASparseFinalCallback,
)
from models.lora_recipes import apply_recipe

def _build_e2e_from_github_csvs():
    """
    Build a DatasetDict directly from the official E2E CSVs (GitHub).
    Produces columns:
      - meaning_representation : str
      - references            : List[str]
      - human_reference       : str (first reference)
    """
    try:
        from datasets import Dataset, DatasetDict, load_dataset

        base = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master"

        def build_split(csv_url: str):
            ds = load_dataset("csv", data_files={"_": csv_url})["_"]
            mrs = ds["mr"]
            refs = ds["ref"]

            # Aggregate multiple references per MR
            buckets: Dict[str, List[str]] = {}
            for mr, ref in zip(mrs, refs):
                buckets.setdefault(mr, []).append(ref)

            meaning_representation, references, human_reference = [], [], []
            for mr, ref_list in buckets.items():
                seen = set()
                uniq: List[str] = []
                for r in ref_list:
                    if r not in seen:
                        seen.add(r)
                        uniq.append(r)
                meaning_representation.append(mr)
                references.append(uniq)
                human_reference.append(uniq[0])

            return Dataset.from_dict({
                "meaning_representation": meaning_representation,
                "references": references,
                "human_reference": human_reference,
            })

        dd = DatasetDict({
            "train": build_split(f"{base}/trainset.csv"),
            "validation": build_split(f"{base}/devset.csv"),
            "test": build_split(f"{base}/testset_w_refs.csv"),
        })
        print("[E2E override] Loaded CSVs from GitHub; skipping HF script datasets.")
        return dd
    except Exception as e:
        print(f"[E2E override] Failed to build dataset from GitHub CSVs: {e}")
        return None


def _build_e2e_from_local_csvs(data_root: str):
    """
    Build a DatasetDict from local CSV files under data_root.
    Accepts common filenames:
      train: train.csv | trainset.csv
      val:   validation.csv | dev.csv | devset.csv
      test:  test.csv | testset.csv | testset_w_refs.csv
    Returns whatever splits are found (train is required).
    """
    from datasets import load_dataset, DatasetDict

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"--data-root path does not exist: {root}")

    def first_existing(names: List[str]) -> Optional[str]:
        for n in names:
            p = root / n
            if p.exists():
                return str(p)
        return None

    files: Dict[str, str] = {}
    tr = first_existing(["train.csv", "trainset.csv"])
    va = first_existing(["validation.csv", "dev.csv", "devset.csv"])
    te = first_existing(["test.csv", "testset.csv", "testset_w_refs.csv"])

    if tr:
        files["train"] = tr
    if va:
        files["validation"] = va
    if te:
        files["test"] = te

    if "train" not in files:
        raise FileNotFoundError(
            f"No training CSV found in {root}. Expected one of: train.csv / trainset.csv"
        )

    ds = load_dataset("csv", data_files=files)
    print("[E2E override] Loaded CSVs from --data-root; skipping HF script datasets.")
    return DatasetDict(ds) if not isinstance(ds, DatasetDict) else ds


def _load_e2e_for_runner(dataset_name: str, data_root: Optional[str]):
    """
    Returns a DatasetDict to override data.e2e._load_e2e() or None to keep default behavior.
    We only support 'e2e' here. If data_root is provided, use local CSVs; otherwise use GitHub CSVs.
    """
    name = (dataset_name or "e2e").strip().lower()
    if name != "e2e":
        raise ValueError(f"Only --dataset e2e is supported by this runner. Got: {dataset_name}")

    if data_root:
        return _build_e2e_from_local_csvs(data_root)
    return _build_e2e_from_github_csvs()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


# ---------- comment-friendly args (like vision) ----------
class CommentArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line: str):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith(";"):
            return []
        return shlex.split(s, comments=False, posix=True)


def parse_args():
    p = CommentArgParser(description="GPT-2 on E2E NLG (Full FT / LoRA).")

    # logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="lora-e2e")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb-log-every", type=int, default=50)
    p.add_argument("--wandb-watch", action="store_true")
    p.add_argument("--wandb-upload-artifacts", action="store_true")

    # dataset options (only e2e here)
    p.add_argument("--dataset", type=str, default="e2e",
                   help="Accepted for compatibility; only 'e2e' is supported here.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Folder with E2E CSVs. If set, load CSVs instead of the GitHub mirror.")

    # run plan
    p.add_argument("--methods", type=str, nargs="+", required=True,
        help=("Choices: full lora base asym_a asym_b cheap random_cheap "
              "cola rac_a rac_b c3la shuffle "
              "sparse_cheap sparse_shuffle sparse_c3la "
              "ba_sparse_lora ba_sparse_final"))
    p.add_argument("--backbone", type=str, default="gpt2",
                   help="HF model id (e.g., gpt2, gpt2-medium, gpt2-large)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])

    # learning rates (now supports per-method overrides)
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-5],
                   help="Default LR(s) unless overridden by --lr-full/--lr-others")
    p.add_argument("--lr-full", type=float, default=None,
                   help="Learning rate for full fine-tuning (overrides --lrs for method=full)")
    p.add_argument("--lr-others", type=float, default=None,
                   help="Learning rate for all non-full methods (overrides --lrs for LoRA & chain recipes)")

    p.add_argument("--ranks", type=int, nargs="*", default=[8])
    p.add_argument("--alphas", type=int, nargs="*", default=None)

    # schedule
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--train-log-interval", type=float, default=0.25)
    p.add_argument("--eval-log-interval", type=float, default=0.25)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-eval-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pad-multiple", type=int, default=8)

    # chains (optional)
    p.add_argument("--chain-every-epochs", type=int, default=None)
    p.add_argument("--chain-every-steps", type=int, default=None)
    p.add_argument("--ref-strategy", type=str, default="all", choices=["all", "first", "random"])
    
    # BA mask knobs (NEW)
    p.add_argument("--ba-alpha", type=float, default=0.5,
                   help="BA-mask keep fraction alpha in (0,1]. Used by ba_sparse_*.")
    p.add_argument("--ba-ties", type=str, default="keep", choices=["keep", "drop"],
                   help="Row/col top-k ties policy for BA mask: keep (>=) or drop (>).")
    p.add_argument("--ba-rounding", type=str, default="ceil", choices=["ceil", "floor"],
                   help="Rounding for per-axis top-k counts.")
    p.add_argument("--ba-recompute-every", type=int, default=1,
                   help="Recompute BA mask every k steps for ba_sparse_lora.")
    p.add_argument("--ba-eval-masked-test", type=str2bool, default=True,
                   help="If true, evaluate test with BA-masked weights after finalize (ba_sparse_final).")

    # output
    p.add_argument("--out-dir", type=str, default="sweep_out_e2e")

    # pretrained toggle
    p.add_argument("--pretrained", type=str2bool, default=True)
    p.add_argument("--lora-plus-gamma", type=float, default=16.0,
                   help="LoRA+: set lr_B = gamma * lr_A (only for method=lora+).")

    # saving behavior
    p.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save epoch_00 and <prefix>-best.pt; skip per-epoch and best-full."
    )
    p.add_argument(
        "--save-json-only",
        action="store_true",
        help="Do not write any checkpoints at all; only JSON summaries/metrics. Test runs from in-memory best."
    )

    return p.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # derive save behavior
    save_json_only = bool(args.save_json_only)
    save_best_only = bool(args.save_best_only)
    _save_all_epochs = not (save_json_only or save_best_only)
    _save_init_ckpt = not save_json_only
    _save_best_full = not (save_json_only or save_best_only)

    # Per-method LR overrides
    lrs_by_method = None
    if args.lr_full is not None or args.lr_others is not None:
        lrs_by_method = {}
        if args.lr_full is not None:
            lrs_by_method["full"] = [float(args.lr_full)]
        if args.lr_others is not None:
            other = [float(args.lr_others)]
            for m in ["lora", "lora_plus", "lora+", "base", "asym_a", "asym_b",
                      "cheap", "random_cheap",
                      "cola", "rac_a", "rac_b", "c3la", "shuffle",
                      "sparse_cheap", "sparse_shuffle", "sparse_c3la",
                      "ba_sparse_lora", "ba_sparse_final"]:
                lrs_by_method[m] = other


    # RA pairing
    if not args.alphas:
        ra_pairs = [(r, 2 * r) for r in args.ranks]
    elif len(args.alphas) == len(args.ranks):
        ra_pairs = list(zip(args.ranks, args.alphas))
    else:
        ra_pairs = [(r, a) for r in args.ranks for a in args.alphas]

    results_path = os.path.join(args.out_dir, "sweep_results.jsonl")
    all_runs = []

    with open(results_path, "a") as f:
        for method in [m.lower() for m in args.methods]:
            method_lrs = (lrs_by_method.get(method) if lrs_by_method else None) or args.lrs
            for seed in args.seeds:
                set_all_seeds(seed)

                # tokenizer
                tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                # dataset override (CSV based), then let E2EDataModule call the patched _load_e2e()
                ds_override = _load_e2e_for_runner(args.dataset, args.data_root)
                if ds_override is not None:
                    import data.e2e as e2e_mod
                    e2e_mod._load_e2e = lambda: ds_override

                # datamodule
                dm = E2EDataModule(
                    tokenizer=tok,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    seed=seed,
                    num_workers=args.num_workers,
                    pad_to_multiple_of=args.pad_multiple,
                    fallback_test_from_val=True,
                    ref_strategy=args.ref_strategy,
                )
                dm.setup()

                # model
                model = GPT2CausalLM(variant=args.backbone, pretrained=bool(args.pretrained))

                # build per-lr / per-(r,a) configs
                for lr in method_lrs:
                    # method selection
                    if method == "full":
                        meth = FullFineTune(lr=lr, weight_decay=0.01)
                        run_name = f"e2e_{args.backbone}_full_lr{lr}_seed{seed}"

                        # trainer config
                        tcfg = TrainConfig(
                            epochs=args.epochs, lr=lr, weight_decay=0.01,
                            use_amp=True, prefer_bf16=None, grad_accum_steps=1,
                            save_dir=args.out_dir, save_name_prefix=run_name,
                            train_log_interval=args.train_log_interval, eval_log_interval=args.eval_log_interval,
                            limit_train_batches=args.limit_train_batches, limit_eval_batches=args.limit_eval_batches,
                            seed=seed,
                            save_best_only=save_best_only,
                            save_checkpoints=not save_json_only,
                            save_all_epochs=_save_all_epochs,
                            save_init_checkpoint=_save_init_ckpt,
                            save_best_with_optimizer=_save_best_full,
                        )

                        callbacks = [
                            EarlyStoppingCallback(patience=args.patience, min_delta=args.min_delta),
                            FlopsCounterCallback(),
                            BestEpochTrainLossCallback(monitor="loss", mode="min", eval_on_train=True),
                        ]
                        if args.wandb:
                            callbacks.append(
                                WandbCallback(
                                    project=args.wandb_project, entity=args.wandb_entity,
                                    group=f"e2e-{args.backbone}-full", mode=args.wandb_mode,
                                    run_name=run_name, tags=args.wandb_tags,
                                    config={"sweep_params": vars(args)},
                                    log_every_n_steps=int(args.wandb_log_every),
                                    watch_model=bool(args.wandb_watch),
                                    upload_artifacts=bool(args.wandb_upload_artifacts),
                                    out_dir=args.out_dir,
                                )
                            )

                        trainer = GenericTrainer(
                            model=model, method=meth, task_logic=CausalLMTask(),
                            train_loader=dm.train_dataloader(),
                            val_loader=dm.val_dataloader(),
                            test_loader=dm.test_dataloader(),
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            tcfg=tcfg, method_cfg_lr_override=lr, callbacks=callbacks, data_module=dm,
                        )

                        best_path, best_score = trainer.fit(run_name)
                        out = {
                            "run_name": run_name,
                            "primary_metric": trainer.primary_metric,
                            "best_score": float(best_score),
                            "best_checkpoint": best_path,
                            "maximize": bool(trainer.maximize),
                        }
                        f.write(json.dumps(out) + "\n")
                        f.flush()
                        all_runs.append(out)

                    elif method in ("lora", "base", "asym_a", "asym_b", "cheap", "random_cheap",
                                    "lora_plus", "lora+", "sparse_cheap",
                                    "ba_sparse_lora", "ba_sparse_final"):
                        recipe = "base" if method in ("lora", "base") else method
                        for (r, a) in ra_pairs:
                            meth = LoRAQV(r=int(r), alpha=int(a), recipe=recipe, lr=lr, weight_decay=0.0)
                            is_plus = (method in ("lora_plus", "lora+"))
                            if is_plus:
                                setattr(meth, "plus_gamma", float(args.lora_plus_gamma))
                            # (Re)apply BA recipe with CLI knobs when relevant (NEW)
                            if recipe in ("ba_sparse_lora", "ba_sparse_final"):
                                apply_recipe(
                                    model,
                                    recipe,
                                    mask_alpha=float(args.ba_alpha),
                                    mask_ties_keep=(args.ba_ties == "keep"),
                                    mask_rounding=str(args.ba_rounding),
                                    mask_recompute_every=int(args.ba_recompute_every),
                                )
                                
                            tag = "lora_plus" if is_plus else recipe
                            run_name = f"e2e_{args.backbone}_{tag}_r{r}_a{a}_lr{lr}_seed{seed}"

                            tcfg = TrainConfig(
                                epochs=args.epochs, lr=lr, weight_decay=0.0,
                                use_amp=True, prefer_bf16=None, grad_accum_steps=1,
                                save_dir=args.out_dir, save_name_prefix=run_name,
                                train_log_interval=args.train_log_interval, eval_log_interval=args.eval_log_interval,
                                limit_train_batches=args.limit_train_batches, limit_eval_batches=args.limit_eval_batches,
                                seed=seed,
                                save_best_only=save_best_only,
                                save_checkpoints=not save_json_only,
                                save_all_epochs=_save_all_epochs,
                                save_init_checkpoint=_save_init_ckpt,
                                save_best_with_optimizer=_save_best_full,
                            )

                            callbacks = [
                                EarlyStoppingCallback(patience=args.patience, min_delta=args.min_delta),
                                FlopsCounterCallback(),
                                BestEpochTrainLossCallback(monitor="loss", mode="min", eval_on_train=True),
                            ]
                            if recipe == "ba_sparse_final":
                                callbacks.insert(0, BASparseFinalCallback(
                                    eval_masked_test=bool(args.ba_eval_masked_test)
                                ))
                            if args.wandb:
                                callbacks.append(
                                    WandbCallback(
                                        project=args.wandb_project, entity=args.wandb_entity,
                                        group=f"e2e-{args.backbone}-{recipe}", mode=args.wandb_mode,
                                        run_name=run_name, tags=args.wandb_tags,
                                        config={"sweep_params": vars(args)},
                                        log_every_n_steps=int(args.wandb_log_every),
                                        watch_model=bool(args.wandb_watch),
                                        upload_artifacts=bool(args.wandb_upload_artifacts),
                                        out_dir=args.out_dir,
                                    )
                                )

                            trainer = GenericTrainer(
                                model=model, method=meth, task_logic=CausalLMTask(),
                                train_loader=dm.train_dataloader(),
                                val_loader=dm.val_dataloader(),
                                test_loader=dm.test_dataloader(),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                tcfg=tcfg, method_cfg_lr_override=lr, callbacks=callbacks, data_module=dm,
                            )
                            best_path, best_score = trainer.fit(run_name)
                            out = {
                                "run_name": run_name,
                                "primary_metric": trainer.primary_metric,
                                "best_score": float(best_score),
                                "best_checkpoint": best_path,
                                "maximize": bool(trainer.maximize),
                            }
                            f.write(json.dumps(out) + "\n")
                            f.flush()
                            all_runs.append(out)

                    else:
                        # chain-capable: cola / rac_a / rac_b / c3la / shuffle
                        recipe = method
                        for (r, a) in ra_pairs:
                            meth = LoRAQV(r=int(r), alpha=int(a), recipe=recipe, lr=lr, weight_decay=0.0)
                            run_name = f"e2e_{args.backbone}_{recipe}_r{r}_a{a}_lr{lr}_seed{seed}"

                            callbacks = [
                                EarlyStoppingCallback(patience=args.patience, min_delta=args.min_delta),
                                FlopsCounterCallback(),
                                BestEpochTrainLossCallback(monitor="loss", mode="min", eval_on_train=True),
                            ]
                            # default: chain each epoch unless overridden
                            cee = args.chain_every_epochs if (args.chain_every_epochs or args.chain_every_steps) else 1
                            ces = args.chain_every_steps
                            callbacks.append(
                                ChainResetCallback(recipe=method, every_n_opt_steps=ces, every_n_epochs=cee)
                            )
                            if args.wandb:
                                callbacks.append(
                                    WandbCallback(
                                        project=args.wandb_project, entity=args.wandb_entity,
                                        group=f"e2e-{args.backbone}-{recipe}", mode=args.wandb_mode,
                                        run_name=run_name, tags=args.wandb_tags,
                                        config={"sweep_params": vars(args)},
                                        log_every_n_steps=int(args.wandb_log_every),
                                        watch_model=bool(args.wandb_watch),
                                        upload_artifacts=bool(args.wandb_upload_artifacts),
                                        out_dir=args.out_dir,
                                    )
                                )

                            tcfg = TrainConfig(
                                epochs=args.epochs, lr=lr, weight_decay=0.0,
                                use_amp=True, prefer_bf16=None, grad_accum_steps=1,
                                save_dir=args.out_dir, save_name_prefix=run_name,
                                train_log_interval=args.train_log_interval, eval_log_interval=args.eval_log_interval,
                                limit_train_batches=args.limit_train_batches, limit_eval_batches=args.limit_eval_batches,
                                seed=seed,
                                save_best_only=save_best_only,
                                save_checkpoints=not save_json_only,
                                save_all_epochs=_save_all_epochs,
                                save_init_checkpoint=_save_init_ckpt,
                                save_best_with_optimizer=_save_best_full,
                            )

                            trainer = GenericTrainer(
                                model=model, method=meth, task_logic=CausalLMTask(),
                                train_loader=dm.train_dataloader(),
                                val_loader=dm.val_dataloader(),
                                test_loader=dm.test_dataloader(),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                tcfg=tcfg, method_cfg_lr_override=lr, callbacks=callbacks, data_module=dm,
                            )
                            best_path, best_score = trainer.fit(run_name)
                            out = {
                                "run_name": run_name,
                                "primary_metric": trainer.primary_metric,
                                "best_score": float(best_score),
                                "best_checkpoint": best_path,
                                "maximize": bool(trainer.maximize),
                            }
                            f.write(json.dumps(out) + "\n")
                            f.flush()
                            all_runs.append(out)

    # aggregate like the vision/glue sweepers so tests find sweep_summary.json
    from statistics import mean, pstdev
    groups: Dict[tuple, List[float]] = {}
    for r in all_runs:
        key = (r["primary_metric"], r["maximize"])
        groups.setdefault(key, []).append(float(r["best_score"]))
    best_by_mean = None
    for (metric, maximize), vals in groups.items():
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        entry = {
            "primary_metric": metric,
            "maximize": bool(maximize),
            "mean_best_val": float(m),
            "std_best_val": float(s),
            "num_runs": len(vals),
        }
        if best_by_mean is None:
            best_by_mean = entry
        else:
            better = (m > best_by_mean["mean_best_val"]) if maximize else (m < best_by_mean["mean_best_val"])
            if better:
                best_by_mean = entry
    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as g:
        json.dump({"best_by_mean": best_by_mean, "runs": all_runs}, g, indent=2)

    print(f"[E2E] Done. Results -> {results_path}")


if __name__ == "__main__":
    main()
