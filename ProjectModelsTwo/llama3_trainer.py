#!/usr/bin/env python3
# llama3_trainer.py
# Unified Llama 3 8B trainer for:
#   - openbookqa (parquet loader, no dataset script execution)
#   - clutrr (raw CSV loader)
#
# Methods (10):
#   fft, lora, chain/cola, cla/cheap, fixa, rac, plus, rcla/random, c3la/modest, rc3la/shuffle
#
# Rules:
#   - NO wandb
#   - NO numpy
#   - Count-based resets via _count_based_reset_epochs (ROUND-based, consistent with other trainers)
#   - Epoch print format (exact):
#       epoch x/total- train_loss:... train_acc:... val_loss:... val_acc:... test_loss:... test_acc:...
#   - Start print: "Run beginning with model: ..., dataset: ..., method: ..., ..." with only relevant hyperparams
#   - End print exactly: "run complete"
#   - Left padding (decoder-only sequence classification)
#   - 16-bit weights (bf16 if supported else fp16), grad checkpointing, KV cache disabled
#   - Prefer bitsandbytes AdamW8bit if available, else AdamW
#
# Notes:
#   - OpenBookQA labels: A/B/C/D -> 0..3
#   - CLUTRR labels: fixed 21-class relation set

from __future__ import annotations

import gc
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------
# Utilities
# ----------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_mp_dtype(dev: str) -> torch.dtype:
    if dev != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _canonicalize_dataset(name: str) -> str:
    n = name.strip().lower()
    if n in {"openbook", "openbookqa", "obqa"}:
        return "openbookqa"
    if n in {"clutrr"}:
        return "clutrr"
    return n


def _count_based_reset_epochs(epochs: int, chainReset: int) -> List[int]:
    """
    Count-based reset schedule: pick `chainReset` epochs evenly spaced in [2, epochs] (inclusive),
    then reset at START of those epochs.

    ROUND-based discretization (no numpy), consistent with the other trainers.
    """
    if chainReset <= 0 or epochs < 2:
        return []
    k = min(chainReset, max(0, epochs - 1))  # distinct points in [2..epochs] is at most (epochs-1)
    if k <= 0:
        return []

    if k == 1:
        pts = [epochs]
    else:
        pts = []
        span = epochs - 2  # distance from 2 to epochs
        for i in range(k):
            t = i / (k - 1)              # in [0,1]
            off = int(round(t * span))   # ROUND (not floor)
            pts.append(2 + off)
        pts[-1] = epochs  # ensure endpoint exact

    return sorted(set(int(x) for x in pts if 2 <= int(x) <= epochs))


def _print_run_beginning(
    model: str,
    dataset: str,
    method: str,
    maxLength: int,
    batchSize: int,
    learningRate: float,
    epochs: int,
    chainReset: int,
    rank: int,
    alpha: int,
    lr_ratio: float,
    seed: int,
) -> None:
    parts = [
        f"Run beginning with model: {model}",
        f"dataset: {dataset}",
        f"method: {method}",
        f"maxLength: {maxLength}",
        f"batchSize: {batchSize}",
        f"learningRate: {learningRate}",
        f"epochs: {epochs}",
        f"seed: {seed}",
    ]

    m = method.strip().lower()

    # fft/ft: only base hyperparams
    if m in {"fft", "ft", "full", "finetune", "full_finetune"}:
        print(", ".join(parts))
        return

    # lora/vanilla: rank/alpha
    if m in {"lora", "vanilla"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # cla/cheap: rank/alpha
    if m in {"cla", "cheap"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # fixa: rank/alpha
    if m in {"fixa"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rcla/random: rank/alpha
    if m in {"rcla", "random"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # plus: rank/alpha/lr_ratio
    if m in {"plus"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}", f"lr_ratio: {lr_ratio}"]
        print(", ".join(parts))
        return

    # chain/cola: chainReset/rank/alpha
    if m in {"chain", "cola"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rac: chainReset/rank/alpha
    if m in {"rac"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # modest/c3la: chainReset/rank/alpha
    if m in {"modest", "c3la"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rc3la/shuffle: chainReset/rank/alpha
    if m in {"rc3la", "shuffle"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # fallback
    parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}", f"lr_ratio: {lr_ratio}"]
    print(", ".join(parts))


def _acc_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


# ----------------------------
# Data
# ----------------------------

class HFDS(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds[i]


@dataclass
class TaskData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_labels: int


def _load_openbookqa_parquet():
    candidates = [
        {
            "base": "https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/",
            "files": {
                "train":      "train-00000-of-00001.parquet",
                "validation": "validation-00000-of-00001.parquet",
                "test":       "test-00000-of-00001.parquet",
            }
        },
        {
            "base": "https://huggingface.co/datasets/allenai/openbookqa/resolve/refs%2Fconvert%2Fparquet/main/",
            "files": {
                "train":      "train-00000-of-00001.parquet",
                "validation": "validation-00000-of-00001.parquet",
                "test":       "test-00000-of-00001.parquet",
            }
        },
        {
            "base": "https://huggingface.co/datasets/allenai/openbookqa/resolve/main/additional/",
            "files": {
                "train": "train-00000-of-00001.parquet",
                "test":  "test-00000-of-00001.parquet",
            }
        },
    ]
    last_err = None
    for cand in candidates:
        base = cand["base"]
        files = cand["files"]
        try:
            data_files = {}
            for split in ("train", "validation", "test"):
                if split in files:
                    data_files[split] = base + files[split]
            ds = load_dataset("parquet", data_files=data_files)
            return ds, set(data_files.keys())
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load OpenBookQA parquet files from known paths. Last error: {last_err}")


def _build_openbookqa_loaders(
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    ds, available = _load_openbookqa_parquet()

    # If validation missing, carve from train deterministically
    if "validation" not in available:
        train_ds = ds["train"]
        frac = min(500 / max(1, len(train_ds)), 0.1)
        split = train_ds.train_test_split(test_size=frac, seed=seed)
        ds = {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"] if "test" in ds else split["test"],
        }

    def _label_to_int(lbl) -> int:
        s = str(lbl).strip().upper()
        return "ABCD".index(s[0])

    def _choices_to_ABCD_texts(choices):
        if isinstance(choices, dict) and "text" in choices and "label" in choices:
            pairs = {str(L).upper(): str(T) for L, T in zip(choices["label"], choices["text"])}
        elif isinstance(choices, (list, tuple)):
            pairs = {str(c.get("label", "")).upper(): str(c.get("text", "")) for c in choices}
        else:
            pairs = {}
        out = ["", "", "", ""]
        for i, L in enumerate("ABCD"):
            if L in pairs:
                out[i] = pairs[L]
        return out

    def _format_example(q_stem, choices):
        A, B, C, D = _choices_to_ABCD_texts(choices)
        return (
            "Question: " + str(q_stem).strip() + "\n"
            "Options:\n"
            f"A) {A}\nB) {B}\nC) {C}\nD) {D}\n"
            "Answer:"
        )

    def tok(batch):
        texts = [_format_example(q, ch) for q, ch in zip(batch["question_stem"], batch["choices"])]
        enc = tokenizer(texts, truncation=True, max_length=maxLength)
        enc["labels"] = [_label_to_int(x) for x in batch["answerKey"]]
        return enc

    train = ds["train"].map(tok, batched=True, desc="Tokenizing train").shuffle(seed=seed)
    valid = ds["validation"].map(tok, batched=True, desc="Tokenizing validation").shuffle(seed=seed)
    test  = ds["test"].map(tok, batched=True, desc="Tokenizing test").shuffle(seed=seed)

    for split in (train, valid, test):
        split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    class CollatorWithLabels(DataCollatorWithPadding):
        def __init__(self, tok_):
            super().__init__(tok_, padding=True)
        def __call__(self, features):
            batch = super().__call__(features)
            batch["labels"] = batch["labels"].long()
            return batch

    collate = CollatorWithLabels(tokenizer)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(HFDS(train), batch_size=batchSize, shuffle=True,  collate_fn=collate, generator=g)
    val_loader   = DataLoader(HFDS(valid), batch_size=batchSize, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(HFDS(test),  batch_size=batchSize, shuffle=False, collate_fn=collate)

    return TaskData(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_labels=4)


def _build_clutrr_loaders(
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    subset = "gen_train234_test2to10"
    base = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/"
    data_files = {
        "train":      base + f"{subset}/train.csv",
        "validation": base + f"{subset}/validation.csv",
        "test":       base + f"{subset}/test.csv",
    }
    ds = load_dataset("csv", data_files=data_files)

    TARGETS = [
        "grandmother","grandfather","wife","husband","granddaughter","grandson",
        "mother","father","daughter","son","niece","nephew","aunt","uncle",
        "sister","brother","sister-in-law","brother-in-law","mother-in-law","father-in-law","cousin"
    ]
    name2id = {t: i for i, t in enumerate(TARGETS)}
    num_labels = len(TARGETS)

    def _label_to_int(v) -> int:
        s = str(v).strip()
        if s.isdigit():
            return int(s)
        return int(name2id.get(s.lower(), 0))

    def _format(story, query):
        return f"Story: {str(story).strip()}\nQuery: {str(query).strip()}\nAnswer:"

    def tok(batch):
        texts = [_format(s, q) for s, q in zip(batch["story"], batch["query"])]
        enc = tokenizer(texts, truncation=True, max_length=maxLength)
        enc["labels"] = [_label_to_int(x) for x in batch["target"]]
        return enc

    train = ds["train"].map(tok, batched=True, desc="Tokenizing train").shuffle(seed=seed)
    valid = ds["validation"].map(tok, batched=True, desc="Tokenizing validation").shuffle(seed=seed)
    test  = ds["test"].map(tok, batched=True, desc="Tokenizing test").shuffle(seed=seed)

    for split in (train, valid, test):
        split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    class CollatorWithLabels(DataCollatorWithPadding):
        def __init__(self, tok_):
            super().__init__(tok_, padding=True)
        def __call__(self, features):
            batch = super().__call__(features)
            batch["labels"] = batch["labels"].long()
            return batch

    collate = CollatorWithLabels(tokenizer)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(HFDS(train), batch_size=batchSize, shuffle=True,  collate_fn=collate, generator=g)
    val_loader   = DataLoader(HFDS(valid), batch_size=batchSize, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(HFDS(test),  batch_size=batchSize, shuffle=False, collate_fn=collate)

    return TaskData(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_labels=num_labels)


def _build_task_data(
    dataset: str,
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    d = _canonicalize_dataset(dataset)
    if d == "openbookqa":
        return _build_openbookqa_loaders(tokenizer, maxLength, batchSize, seed)
    if d == "clutrr":
        return _build_clutrr_loaders(tokenizer, maxLength, batchSize, seed)
    raise ValueError(f"Unknown/unsupported dataset for llama3_trainer: {dataset}")


# ----------------------------
# Adapter modules (Llama3)
# ----------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, train_A: bool = True, train_B: bool = True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)

        self.A = nn.Linear(base.in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)

        for p in self.A.parameters():
            p.requires_grad = bool(train_A)
        for p in self.B.parameters():
            p.requires_grad = bool(train_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * (float(self.alpha) / float(self.r))


class CheapLoRA(nn.Module):
    """
    Cheap-LoRA: A is a frozen identity selector at offset 0 (deterministic),
    B trainable (zero-init).
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)

        in_features = base.in_features
        assert self.r <= in_features, "rank r must be <= in_features"
        A = torch.zeros(self.r, in_features, dtype=base.weight.dtype, device=base.weight.device)
        idx = torch.arange(self.r, device=base.weight.device)
        A[idx, idx] = 1.0
        self.register_buffer("A", A, persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (float(self.alpha) / float(self.r))
        return self.base(x) + delta


class RandomFrozenA(nn.Module):
    """
    Random-frozen A (dense, row-normalized), B trainable (zero-init).
    Seeded per reset for chain/rac; for plain rcla/random uses the Run seed once.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)

        gen = torch.Generator(device=base.weight.device)
        gen.manual_seed(int(seed))
        A = torch.randn(self.r, base.in_features, generator=gen, dtype=base.weight.dtype, device=base.weight.device)
        A = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("A", A, persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (float(self.alpha) / float(self.r))
        return self.base(x) + delta


class ModestLoRA(nn.Module):
    """
    Modest/C3LA/RC3LA: A is a frozen identity window of width r at an offset.
      - c3la/modest: advance offset window (no merge)
      - rc3la/shuffle: fuse delta into base, shuffle offset, reset B
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)
        self.d_in = int(base.in_features)

        self.register_buffer("offset", torch.tensor(0, dtype=torch.long), persistent=False)

        A0 = torch.zeros(self.r, self.d_in, dtype=base.weight.dtype, device=base.weight.device)
        idx = torch.arange(self.r, device=base.weight.device)
        A0[idx, idx] = 1.0
        self.register_buffer("A", A0, persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (float(self.alpha) / float(self.r))
        return self.base(x) + delta

    @torch.no_grad()
    def advance_offset(self) -> None:
        max_start = self.d_in - self.r
        new_off = int(self.offset.item()) + self.r
        if new_off > max_start:
            new_off = 0
        self.offset.fill_(new_off)
        Anew = torch.zeros_like(self.A)
        cols = torch.arange(self.r, device=Anew.device) + new_off
        Anew[torch.arange(self.r, device=Anew.device), cols] = 1.0
        self.A.copy_(Anew)

    @torch.no_grad()
    def shuffle_offset(self) -> None:
        max_start = self.d_in - self.r
        new_off = int(torch.randint(low=0, high=max_start + 1, size=(1,), device=self.A.device).item())
        self.offset.fill_(new_off)
        Anew = torch.zeros_like(self.A)
        cols = torch.arange(self.r, device=Anew.device) + new_off
        Anew[torch.arange(self.r, device=Anew.device), cols] = 1.0
        self.A.copy_(Anew)

    @torch.no_grad()
    def reset_B(self) -> None:
        self.B.weight.zero_()


# ----------------------------
# Model helpers (Llama3)
# ----------------------------

def _locate_decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers
    raise RuntimeError("Could not locate decoder layers (model.model.layers) for Llama injection.")


def _freeze_all_but_head(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if (".score." in name) or name.startswith("score.") or (".classifier." in name) or name.startswith("classifier."):
            p.requires_grad = True


def _trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _make_optimizer(params, learningRate: float):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=learningRate)
    except Exception:
        return torch.optim.AdamW(params, lr=learningRate)


# ----------------------------
# Method wrappers
# ----------------------------

@torch.no_grad()
def _merge_lora_into_base(attn_module, proj_name: str, alpha: int, rank: int) -> None:
    mod = getattr(attn_module, proj_name)
    if isinstance(mod, LoRALinear):
        delta = (mod.B.weight @ mod.A.weight) * (float(alpha) / float(rank))
        mod.base.weight.data.add_(delta.to(dtype=mod.base.weight.dtype))
        setattr(attn_module, proj_name, mod.base)


class ChainLike:
    """
    For chain/cola and rac:
      - Merge LoRA into base on reset
      - Remove adapters (replace with base)
      - Reinject fresh adapters (new seed per reset if desired)
      - Rebuild optimizer after reset (since params changed)
    """
    def __init__(self, model: nn.Module, rank: int, alpha: int):
        self.model = model
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.layers = _locate_decoder_layers(model)

    def _inject(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, self.rank, self.alpha, train_A=True, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, self.rank, self.alpha, train_A=True, train_B=True)

    @torch.no_grad()
    def merge_and_reinject(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            _merge_lora_into_base(attn, "q_proj", self.alpha, self.rank)
            _merge_lora_into_base(attn, "v_proj", self.alpha, self.rank)
        self._inject()


class ModestWrapper:
    """
    For modest/c3la and rc3la/shuffle:
      - c3la/modest reset: advance offset only (no merge)
      - rc3la/shuffle reset: fuse delta into base, shuffle offset, reset B
    """
    def __init__(self, model: nn.Module, rank: int, alpha: int):
        self.model = model
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.layers = _locate_decoder_layers(model)

    def inject(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.q_proj = ModestLoRA(attn.q_proj, self.rank, self.alpha)
            attn.v_proj = ModestLoRA(attn.v_proj, self.rank, self.alpha)

    @torch.no_grad()
    def advance_only(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            for name in ("q_proj", "v_proj"):
                mod = getattr(attn, name)
                if isinstance(mod, ModestLoRA):
                    mod.advance_offset()

    @torch.no_grad()
    def fuse_shuffle_resetB(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            for name in ("q_proj", "v_proj"):
                mod = getattr(attn, name)
                if isinstance(mod, ModestLoRA):
                    # fuse into base
                    delta = (mod.B.weight @ mod.A) * (float(self.alpha) / float(self.rank))  # (out,in)
                    mod.base.weight.data.add_(delta.to(dtype=mod.base.weight.dtype))
                    # shuffle window + reset B
                    mod.shuffle_offset()
                    mod.reset_B()


# ----------------------------
# Train / eval
# ----------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn, dev: str, mp_dtype: torch.dtype) -> Tuple[float, float]:
    model.eval()
    totL = 0.0
    totA = 0.0
    use_autocast = (dev == "cuda")
    for b in loader:
        x = b["input_ids"].to(dev, non_blocking=True)
        m = b["attention_mask"].to(dev, non_blocking=True)
        y = b["labels"].to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=mp_dtype, enabled=use_autocast):
            logits = model(input_ids=x, attention_mask=m).logits
            L = loss_fn(logits, y)
        totL += L.item()
        totA += _acc_top1(logits, y)
    return _safe_div(totL, len(loader)), _safe_div(totA, len(loader))


def _train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, dev: str, mp_dtype: torch.dtype) -> Tuple[float, float]:
    model.train()
    totL = 0.0
    totA = 0.0
    use_autocast = (dev == "cuda")
    for b in loader:
        x = b["input_ids"].to(dev, non_blocking=True)
        m = b["attention_mask"].to(dev, non_blocking=True)
        y = b["labels"].to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=mp_dtype, enabled=use_autocast):
            logits = model(input_ids=x, attention_mask=m).logits
            L = loss_fn(logits, y)

        L.backward()
        optimizer.step()

        totL += L.item()
        totA += _acc_top1(logits, y)

    return _safe_div(totL, len(loader)), _safe_div(totA, len(loader))


# ----------------------------
# Public entry
# ----------------------------

def Run(
    model: str = "llama3_8b",
    dataset: str = "openbookqa",
    method: str = "lora",

    maxLength: int = 288,
    batchSize: int = 8,
    learningRate: float = 1e-4,
    epochs: int = 12,

    chainReset: int = 0,
    rank: int = 2,
    alpha: int = 0,
    lr_ratio: float = 16.0,

    seed: int = 42,
):
    _set_seed(seed)

    dataset = _canonicalize_dataset(dataset)
    method_l = method.strip().lower()

    if alpha == 0:
        alpha = 2 * rank

    # Validate dataset
    if dataset not in {"openbookqa", "clutrr"}:
        raise ValueError(f"Unknown/unsupported dataset for llama3_trainer: {dataset}")

    # Validate method
    valid_methods = {"fft", "lora", "vanilla", "chain", "cola", "cla", "cheap", "fixa", "rac", "plus", "rcla", "random", "c3la", "modest", "rc3la", "shuffle"}
    if method_l not in valid_methods:
        raise ValueError(f"Unknown/unsupported method in llama3_trainer: {method}")

    # Print run beginning
    _print_run_beginning(
        model=model,
        dataset=dataset,
        method=method,
        maxLength=maxLength,
        batchSize=batchSize,
        learningRate=learningRate,
        epochs=epochs,
        chainReset=chainReset,
        rank=rank,
        alpha=alpha,
        lr_ratio=lr_ratio,
        seed=seed,
    )

    dev = _device()
    mp_dtype = _pick_mp_dtype(dev)

    # Tokenizer (LEFT padding required)
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    task = _build_task_data(dataset, tokenizer, maxLength, batchSize, seed)

    # Model
    if dev == "cuda":
        torch.cuda.empty_cache()

    net = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=task.num_labels,
        torch_dtype=mp_dtype if dev == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(dev)

    net.config.pad_token_id = tokenizer.pad_token_id
    gen_cfg = getattr(net, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id

    # Memory efficiency
    net.config.use_cache = False
    net.gradient_checkpointing_enable()
    if hasattr(net, "enable_input_require_grads"):
        net.enable_input_require_grads()

    loss_fn = nn.CrossEntropyLoss()

    # Freeze strategy depends on method
    # - fft: train all parameters
    # - others: freeze all but head + adapters
    if method_l in {"fft", "ft", "full", "finetune", "full_finetune"}:
        for p in net.parameters():
            p.requires_grad = True
    else:
        _freeze_all_but_head(net)

    layers = _locate_decoder_layers(net)

    # Build method modules + optimizer
    chain_like: Optional[ChainLike] = None
    modest_wrap: Optional[ModestWrapper] = None

    def _reseed_for_reset(base_seed: int, reset_idx: int) -> int:
        # deterministic per reset event
        return int(base_seed) + 1000 * int(reset_idx) + 17

    if method_l in {"fft", "ft", "full", "finetune", "full_finetune"}:
        optimizer = _make_optimizer(net.parameters(), learningRate)

    elif method_l in {"lora", "vanilla"}:
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, train_A=True, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, train_A=True, train_B=True)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"cla", "cheap"}:
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = CheapLoRA(attn.q_proj, rank, alpha)
            attn.v_proj = CheapLoRA(attn.v_proj, rank, alpha)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"fixa"}:
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, train_A=False, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, train_A=False, train_B=True)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"rcla", "random"}:
        # single-seed random A (not reset unless method is chain/rac)
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = RandomFrozenA(attn.q_proj, rank, alpha, seed=seed)
            attn.v_proj = RandomFrozenA(attn.v_proj, rank, alpha, seed=seed)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"plus"}:
        if lr_ratio <= 0:
            raise ValueError("lr_ratio must be > 0 for plus method")

        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, train_A=True, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, train_A=True, train_B=True)

        group_A, group_B, group_other = [], [], []
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            if ".A." in n:
                group_A.append(p)
            elif ".B." in n:
                group_B.append(p)
            else:
                group_other.append(p)

        # Prefer AdamW8bit if possible with param groups
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                [
                    {"params": group_other, "lr": learningRate},
                    {"params": group_B, "lr": learningRate},
                    {"params": group_A, "lr": learningRate / lr_ratio},
                ]
            )
        except Exception:
            optimizer = torch.optim.AdamW(
                [
                    {"params": group_other, "lr": learningRate},
                    {"params": group_B, "lr": learningRate},
                    {"params": group_A, "lr": learningRate / lr_ratio},
                ]
            )

    elif method_l in {"chain", "cola"}:
        chain_like = ChainLike(net, rank, alpha)
        chain_like._inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"rac"}:
        chain_like = ChainLike(net, rank, alpha)
        chain_like._inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"modest", "c3la"}:
        modest_wrap = ModestWrapper(net, rank, alpha)
        modest_wrap.inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method_l in {"rc3la", "shuffle"}:
        modest_wrap = ModestWrapper(net, rank, alpha)
        modest_wrap.inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    else:
        raise ValueError(f"Unknown/unsupported method in llama3_trainer: {method}")

    # Reset schedule (count-based) for chain/rac/modest/rc3la
    reset_epochs: List[int] = []
    if method_l in {"chain", "cola", "rac", "modest", "c3la", "rc3la", "shuffle"}:
        reset_epochs = _count_based_reset_epochs(epochs, chainReset)

    reset_count = 0

    # Training loop
    for ep in range(1, epochs + 1):
        # resets at START of epoch
        if ep in reset_epochs:
            reset_count += 1
            reset_seed = _reseed_for_reset(seed, reset_count)
            _set_seed(reset_seed)

            if method_l in {"chain", "cola", "rac"}:
                assert chain_like is not None
                chain_like.merge_and_reinject()
                optimizer = _make_optimizer(_trainable_params(net), learningRate)

            elif method_l in {"modest", "c3la"}:
                assert modest_wrap is not None
                # c3la: advance window only (no merge)
                modest_wrap.advance_only()

            elif method_l in {"rc3la", "shuffle"}:
                assert modest_wrap is not None
                # rc3la: fuse + shuffle + reset B
                modest_wrap.fuse_shuffle_resetB()

        tr_loss, tr_acc = _train_one_epoch(net, task.train_loader, loss_fn, optimizer, dev, mp_dtype)
        val_loss, val_acc = _evaluate(net, task.val_loader, loss_fn, dev, mp_dtype)
        test_loss, test_acc = _evaluate(net, task.test_loader, loss_fn, dev, mp_dtype)

        print(
            f"epoch {ep}/{epochs}- "
            f"train_loss:{tr_loss} train_acc:{tr_acc} "
            f"val_loss:{val_loss} val_acc:{val_acc} "
            f"test_loss:{test_loss} test_acc:{test_acc}"
        )

    print("run complete")

    # Cleanup
    try:
        del optimizer
        del net
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    Run()