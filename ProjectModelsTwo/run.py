#!/usr/bin/env python3
"""
run.py — single entrypoint for all models/methods with API parity.

Folder layout (only 5 files total):
  run.py
  deberta_trainer.py
  deepseekcoder_trainer.py
  tinyllama_trainer.py
  llama3_trainer.py

Usage examples (order does NOT matter; you can mix positional aliases and key=value):
  python run.py
  python run.py deberta lora mrpc
  python run.py model=deberta_v2_xxl method=chain dataset=mrpcs epochs=50 rank=8
  python run.py llama3 plus openbookqa lr_ratio=32 learningRate=2e-5
  python run.py deepseek fft django maxLength=2048 batchSize=2

Accepted hyperparameters (only these):
  maxLength, epochs, batchSize, learningRate, rank, alpha, chainReset, seed, lr_ratio

Defaults:
  maxLength=256, epochs=3, batchSize=8, learningRate=1e-5,
  rank=16, alpha=2*rank (if alpha not provided or alpha==0),
  chainReset=5, seed=100, lr_ratio=16

Notes:
- Some methods/models won’t use some hyperparameters, but we still accept them for parity.
- We DO NOT use wandb here; trainers should print progress and return a metrics dict.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# --- Canonical choices + aliases --------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    # deepseek
    "deepseek": "deepseekcoder",
    "deepseekcoder": "deepseekcoder",
    # deberta v3 base
    "deberta": "deberta_v3_base",
    "deberta_v3": "deberta_v3_base",
    "deberta_base": "deberta_v3_base",
    "deberta_v3_base": "deberta_v3_base",
    # deberta v2 xxl
    "deberta_v2": "deberta_v2_xxl",
    "deberta_xxl": "deberta_v2_xxl",
    "deberta_v2_xxl": "deberta_v2_xxl",
    # tinyllama
    "tinyllama": "tinyllama",
    "tiny": "tinyllama",
    # llama3
    "llama3": "llama3",
    "llama": "llama3",
}

METHOD_ALIASES: Dict[str, str] = {
    # Chain-of-LoRA
    "chain": "chain",
    "cola": "chain",
    # Cheap-LoRA
    "cla": "cla",
    "cheap": "cla",
    # C3LA / Modest LoRA variant
    "c3la": "c3la",
    "modest": "c3la",
    # Vanilla LoRA
    "lora": "lora",
    "vanilla": "lora",
    # FixA / Asymmetric
    "fixa": "fixa",
    "asymmetric": "fixa",
    # Full fine-tuning
    "fft": "fft",
    "ft": "fft",
    # RAC / RAC-LoRA
    "rac": "rac",
    # LoRA+
    "plus": "plus",
    # rCLA
    "rcla": "rcla",
    "random": "rcla",
    # rC3LA
    "rc3la": "rc3la",
    "shuffle": "rc3la",
}

# Allowed datasets per canonical model
DATASETS_BY_MODEL: Dict[str, List[str]] = {
    "deberta_v2_xxl": ["mrpcs", "paws", "trec50"],
    "deberta_v3_base": ["mrpc", "rte", "sts-b", "trec50", "paws"],
    "tinyllama": ["openbookqa", "folio", "logiqa", "clutrr"],
    "llama3": ["openbookqa", "clutrr"],
    "deepseekcoder": ["django"],
}

# Default dataset per model (when user doesn’t specify one)
DEFAULT_DATASET_BY_MODEL: Dict[str, str] = {
    "deberta_v2_xxl": "mrpcs",
    "deberta_v3_base": "mrpc",
    "tinyllama": "clutrr",
    "llama3": "openbookqa",
    "deepseekcoder": "django",
}

DEFAULT_MODEL = "deberta_v3_base"
DEFAULT_METHOD = "lora"

HYPERPARAM_KEYS = {
    "maxlength": "maxLength",
    "maxLength": "maxLength",
    "epochs": "epochs",
    "batchsize": "batchSize",
    "batchSize": "batchSize",
    "learningrate": "learningRate",
    "learningRate": "learningRate",
    "rank": "rank",
    "alpha": "alpha",
    "chainreset": "chainReset",
    "chainReset": "chainReset",
    "seed": "seed",
    "lr_ratio": "lr_ratio",
    "lrratio": "lr_ratio",
}


# --- Config -----------------------------------------------------------------

@dataclass
class RunConfig:
    model: str = DEFAULT_MODEL
    method: str = DEFAULT_METHOD
    dataset: str = ""  # filled after model is finalized

    maxLength: int = 256
    epochs: int = 3
    batchSize: int = 8
    learningRate: float = 1e-5
    rank: int = 16
    alpha: int = 0            # if 0 -> set to 2*rank
    chainReset: int = 5
    seed: int = 100
    lr_ratio: int = 16

    def finalize(self) -> None:
        if not self.dataset:
            self.dataset = DEFAULT_DATASET_BY_MODEL.get(self.model, "")

        if self.alpha == 0:
            self.alpha = 2 * int(self.rank)


# --- Parsing ----------------------------------------------------------------

def _canonicalize_model(token: str) -> Optional[str]:
    if not token:
        return None
    return MODEL_ALIASES.get(token.strip().lower())

def _canonicalize_method(token: str) -> Optional[str]:
    if not token:
        return None
    return METHOD_ALIASES.get(token.strip().lower())

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _parse_value(key: str, raw: str):
    if key == "learningRate":
        return float(raw)
    if key in ("maxLength", "epochs", "batchSize", "rank", "alpha", "chainReset", "seed", "lr_ratio"):
        return int(float(raw))
    return raw

def parse_args(argv: List[str]) -> RunConfig:
    """
    Accepts both:
      - key=value tokens, any order
      - positional alias tokens for model/method/dataset (any order)
    """
    cfg = RunConfig()

    saw_model = False
    saw_method = False
    saw_dataset = False

    positional: List[str] = []

    # Pass 1: key=value
    for tok in argv:
        tok = tok.strip()
        if not tok:
            continue

        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip()
            v = v.strip()
            kl = k.lower()

            if kl == "model":
                cm = _canonicalize_model(v)
                if cm is None:
                    raise ValueError(f"Unknown model alias: {v}")
                cfg.model = cm
                saw_model = True
                continue

            if kl == "method":
                mm = _canonicalize_method(v)
                if mm is None:
                    raise ValueError(f"Unknown method alias: {v}")
                cfg.method = mm
                saw_method = True
                continue

            if kl == "dataset":
                cfg.dataset = v.strip().lower()
                saw_dataset = True
                continue

            hk = HYPERPARAM_KEYS.get(k, HYPERPARAM_KEYS.get(kl))
            if hk is None:
                raise ValueError(f"Unknown argument: {k}")
            setattr(cfg, hk, _parse_value(hk, v))
        else:
            positional.append(tok)

    # Pass 2: positional tokens (any order)
    for tok in positional:
        low = tok.lower().strip()

        cm = _canonicalize_model(low)
        if cm is not None:
            cfg.model = cm
            saw_model = True
            continue

        mm = _canonicalize_method(low)
        if mm is not None:
            cfg.method = mm
            saw_method = True
            continue

        if not saw_dataset:
            cfg.dataset = low
            saw_dataset = True
            continue

        if _is_number(low):
            raise ValueError(
                f"Positional numeric token '{tok}' is ambiguous. "
                f"Please pass hyperparameters as key=value (e.g., epochs=10)."
            )

        raise ValueError(f"Unrecognized token: {tok}")

    # Fill defaults + alpha rule
    cfg.finalize()

    # Validate dataset/model compatibility
    allowed = DATASETS_BY_MODEL.get(cfg.model, [])
    if cfg.dataset not in allowed:
        default_ds = DEFAULT_DATASET_BY_MODEL.get(cfg.model, allowed[0] if allowed else "")
        if not default_ds:
            raise ValueError(f"No datasets configured for model={cfg.model}")
        if saw_dataset:
            raise ValueError(
                f"Dataset '{cfg.dataset}' is not valid for model '{cfg.model}'. "
                f"Allowed: {allowed}"
            )
        cfg.dataset = default_ds

    return cfg


# --- Dispatch ----------------------------------------------------------------

def _import_trainer(model: str):
    """
    Trainers should expose:

      def Run(
          maxLength: int,
          epochs: int,
          batchSize: int,
          learningRate: float,
          rank: int,
          alpha: int,
          chainReset: int,
          seed: int,
          lr_ratio: int,
          dataset: str,
          method: str,
          model: str,
      ) -> dict

    Trainers may ignore unused args per method/model, but must accept them.
    They should print progress and optionally return a metrics dict.
    """
    if model in ("deberta_v3_base", "deberta_v2_xxl"):
        from deberta_trainer import Run as TrainerRun
        return TrainerRun
    if model == "tinyllama":
        from tinyllama_trainer import Run as TrainerRun
        return TrainerRun
    if model == "llama3":
        from llama3_trainer import Run as TrainerRun
        return TrainerRun
    if model == "deepseekcoder":
        from deepseekcoder_trainer import Run as TrainerRun
        return TrainerRun
    raise ValueError(f"No trainer wired for model={model}")


def main(argv: List[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0

    cfg = parse_args(argv)

    print("=" * 80)
    print("RunConfig:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print("=" * 80)

    trainer_run = _import_trainer(cfg.model)

    metrics = trainer_run(
        maxLength=cfg.maxLength,
        epochs=cfg.epochs,
        batchSize=cfg.batchSize,
        learningRate=cfg.learningRate,
        rank=cfg.rank,
        alpha=cfg.alpha,
        chainReset=cfg.chainReset,
        seed=cfg.seed,
        lr_ratio=cfg.lr_ratio,
        dataset=cfg.dataset,
        method=cfg.method,
        model=cfg.model,
    )

    if isinstance(metrics, dict) and metrics:
        print("\nFinal metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:
        print(f"\n[run.py] ERROR: {e}")
        print("Tip: use -h for usage examples.")
        raise