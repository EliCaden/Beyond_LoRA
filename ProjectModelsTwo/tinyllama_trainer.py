#!/usr/bin/env python3
# tinyllama_trainer.py
#
# Unified TinyLlama 1.1B trainer for:
#   - openbookqa (parquet loader, no dataset script execution)
#   - clutrr (raw CSV loader)
#   - folio (JSONL loader)
#   - logiqa (JSONL loader)
#
# Methods (10):
#   fft, lora, chain/cola, cla/cheap, fixa, rac, plus, rcla/random, c3la/modest, rc3la/shuffle
#
# Rules:
#   - NO wandb
#   - NO numpy
#   - Count-based resets via _count_based_reset_epochs (ROUND-based)
#   - Epoch print format (exact):
#       epoch x/total- train_loss:... train_acc:... val_loss:... val_acc:... test_loss:... test_acc:...
#   - End print exactly: "run complete"
#   - PRINT run beginning line (matches other trainers)
#   - Left padding (decoder-only sequence classification)
#   - bf16 if supported else fp16 (cuda), grad checkpointing, KV cache disabled
#   - Prefer bitsandbytes AdamW8bit if available, else AdamW

from __future__ import annotations

import gc
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    if n in {"folio"}:
        return "folio"
    if n in {"logiqa", "logiqa2", "logiqa2.0"}:
        return "logiqa"
    return n


def _canonicalize_method(name: str) -> str:
    m = name.strip().lower()
    aliases = {
        "ft": "fft",
        "full": "fft",
        "finetune": "fft",
        "full_finetune": "fft",

        "vanilla": "lora",

        "cola": "chain",

        "cheap": "cla",

        "random": "rcla",

        "modest": "c3la",
        "shuffle": "rc3la",
    }
    return aliases.get(m, m)


def _count_based_reset_epochs(epochs: int, chainReset: int) -> List[int]:
    """
    Count-based reset schedule: pick `chainReset` epochs evenly spaced in [2, epochs] (inclusive),
    then reset at START of those epochs.

    ROUND-based discretization (no numpy).
    """
    if chainReset <= 0 or epochs < 2:
        return []
    k = min(chainReset, max(0, epochs - 1))  # at most distinct points in [2..epochs]
    if k <= 0:
        return []
    if k == 1:
        pts = [epochs]
    else:
        pts = []
        span = epochs - 2
        for i in range(k):
            t = i / (k - 1)
            off = int(round(t * span))
            pts.append(2 + off)
        pts[-1] = epochs
    return sorted(set(int(x) for x in pts if 2 <= int(x) <= epochs))


def _acc_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _make_optimizer(params, learningRate: float):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=learningRate)
    except Exception:
        return torch.optim.AdamW(params, lr=learningRate)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _print_run_beginning(
    *,
    model_name: str,
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
    """
    Match the other trainers' "Run beginning with model: ..." style, and include
    method-conditional hyperparams (rank/alpha, lr_ratio, chainReset).
    """
    parts = [
        f"Run beginning with model: {model_name}",
        f"dataset: {dataset}",
        f"method: {method}",
        f"maxLength: {maxLength}",
        f"batchSize: {batchSize}",
        f"learningRate: {learningRate}",
        f"epochs: {epochs}",
        f"seed: {seed}",
    ]

    uses_rank_alpha = method in {"lora", "chain", "cla", "fixa", "rac", "plus", "rcla", "c3la", "rc3la"}
    uses_lr_ratio = method == "plus"
    uses_chainReset = method in {"chain", "rac", "c3la", "rc3la"}

    if uses_rank_alpha:
        parts.append(f"rank: {rank}")
        parts.append(f"alpha: {alpha}")
    if uses_lr_ratio:
        parts.append(f"lr_ratio: {lr_ratio}")
    if uses_chainReset:
        parts.append(f"chainReset: {chainReset}")

    print(" - ".join(parts))


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
    ]
    last_err = None
    for cand in candidates:
        try:
            data_files = {k: cand["base"] + v for k, v in cand["files"].items()}
            ds = load_dataset("parquet", data_files=data_files)
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load OpenBookQA parquet files. Last error: {last_err}")


def _build_openbookqa_loaders(
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    ds = _load_openbookqa_parquet()

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


def _build_folio_loaders(
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    base = "https://huggingface.co/datasets/tasksource/folio/resolve/main/"
    ds = load_dataset(
        "json",
        data_files={
            "train":      base + "folio_v2_train.jsonl",
            "validation": base + "folio_v2_validation.jsonl",
        }
    )

    if "test" not in ds:
        split = ds["validation"].train_test_split(test_size=0.5, seed=seed)
        ds["validation"] = split["train"]
        ds["test"] = split["test"]

    def normalize_label(x: str) -> str:
        s = str(x).strip().lower()
        if s in ("true", "entails", "entailed", "t"):
            return "True"
        if s in ("false", "contradiction", "contradicted", "f"):
            return "False"
        return "Uncertain"

    label2id = {"False": 0, "True": 1, "Uncertain": 2}
    num_labels = 3

    def _format_example(premises, conclusion):
        if isinstance(premises, list):
            premises_text = "\n".join(str(p).strip() for p in premises)
        else:
            premises_text = str(premises).strip()
        conclusion_text = str(conclusion).strip()
        return (
            "Premises:\n" + premises_text + "\n\n"
            "Statement:\n" + conclusion_text + "\n\n"
            "Label (True/False/Uncertain):"
        )

    def tok(batch):
        texts = [_format_example(p, c) for p, c in zip(batch["premises"], batch["conclusion"])]
        enc = tokenizer(texts, truncation=True, max_length=maxLength)
        enc["labels"] = [label2id[normalize_label(x)] for x in batch["label"]]
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


def _build_logiqa_loaders(
    tokenizer,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    base = "https://huggingface.co/datasets/datatune/LogiQA2.0/resolve/main/MRC/"
    ds = load_dataset(
        "json",
        data_files={
            "train":      base + "train.txt",
            "validation": base + "dev.txt",
            "test":       base + "test.txt",
        }
    )

    def _fmt(ctx, q, opts):
        a, b, c, d = opts[0], opts[1], opts[2], opts[3]
        return (
            "Context: " + str(ctx).strip() + "\n"
            "Question: " + str(q).strip() + "\n"
            "Options:\n"
            f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n"
            "Answer:"
        )

    def _label_to_int(lbl):
        if isinstance(lbl, int):
            return int(lbl)
        s = str(lbl).strip().upper()
        if s and s[0] in "ABCD":
            return "ABCD".index(s[0])
        return int(lbl)

    def tok(batch):
        contexts = batch["text"] if "text" in batch else batch.get("context", [""] * len(batch["question"]))
        queries  = batch["question"] if "question" in batch else batch.get("query", [""] * len(contexts))
        options  = batch["options"] if "options" in batch else batch["answers"]
        labels   = batch["answer"] if "answer" in batch else (batch["label"] if "label" in batch else batch["correct_option"])

        texts = [_fmt(c, q, o) for c, q, o in zip(contexts, queries, options)]
        enc = tokenizer(texts, truncation=True, max_length=maxLength)
        enc["labels"] = [_label_to_int(x) for x in labels]
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
    if d == "folio":
        return _build_folio_loaders(tokenizer, maxLength, batchSize, seed)
    if d == "logiqa":
        return _build_logiqa_loaders(tokenizer, maxLength, batchSize, seed)
    raise ValueError(f"Unknown/unsupported dataset for tinyllama_trainer: {dataset}")


# ----------------------------
# Adapter modules (Llama-like)
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
    # A fixed [I_r | 0], only B trainable
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = int(alpha)

        in_features = base.in_features
        if self.r > in_features:
            raise ValueError("rank r must be <= in_features")
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
    # Dense random A (row-normalized) frozen, B trainable
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
    # Identity window A that shifts; B trainable
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
# Model helpers
# ----------------------------

def _locate_decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers
    raise RuntimeError("Could not locate decoder layers (model.model.layers) for TinyLlama injection.")


def _freeze_all_but_head(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    # keep classification head trainable
    for name, p in model.named_parameters():
        if (".score." in name) or name.startswith("score.") or (".classifier." in name) or name.startswith("classifier."):
            p.requires_grad = True


@torch.no_grad()
def _merge_lora_into_base(attn_module, proj_name: str, alpha: int, rank: int) -> None:
    mod = getattr(attn_module, proj_name)
    if isinstance(mod, LoRALinear):
        delta = (mod.B.weight @ mod.A.weight) * (float(alpha) / float(rank))
        mod.base.weight.data.add_(delta.to(dtype=mod.base.weight.dtype))
        setattr(attn_module, proj_name, mod.base)


class ChainLike:
    # chain/cola and rac: merge+reinject
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
    # c3la/modest and rc3la/shuffle
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
                    delta = (mod.B.weight @ mod.A) * (float(self.alpha) / float(self.rank))
                    mod.base.weight.data.add_(delta.to(dtype=mod.base.weight.dtype))
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
    model: str = "tinyllama_1.1b",
    dataset: str = "openbookqa",
    method: str = "lora",

    maxLength: int = 288,
    batchSize: int = 8,
    learningRate: float = 4e-4,
    epochs: int = 12,

    chainReset: int = 0,
    rank: int = 2,
    alpha: int = 0,
    lr_ratio: float = 16.0,

    seed: int = 42,
):
    dataset = _canonicalize_dataset(dataset)
    method = _canonicalize_method(method)

    valid_datasets = {"openbookqa", "clutrr", "folio", "logiqa"}
    valid_methods = {"fft", "lora", "chain", "cla", "fixa", "rac", "plus", "rcla", "c3la", "rc3la"}

    if dataset not in valid_datasets:
        raise ValueError(f"Unknown/unsupported dataset for tinyllama_trainer: {dataset}")
    if method not in valid_methods:
        raise ValueError(f"Unknown/unsupported method in tinyllama_trainer: {method}")

    if alpha == 0:
        alpha = 2 * rank

    _print_run_beginning(
        model_name=model,
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

    _set_seed(seed)

    dev = _device()
    mp_dtype = _pick_mp_dtype(dev)

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required

    task = _build_task_data(dataset, tokenizer, maxLength, batchSize, seed)

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

    # Freeze strategy
    if method == "fft":
        for p in net.parameters():
            p.requires_grad = True
    else:
        _freeze_all_but_head(net)

    layers = _locate_decoder_layers(net)

    chain_like: Optional[ChainLike] = None
    modest_wrap: Optional[ModestWrapper] = None

    def _reseed_for_reset(base_seed: int, reset_idx: int) -> int:
        return int(base_seed) + 1000 * int(reset_idx) + 17

    # Build method modules + optimizer
    if method == "fft":
        optimizer = _make_optimizer(net.parameters(), learningRate)

    elif method == "lora":
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, train_A=True, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, train_A=True, train_B=True)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method == "cla":
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = CheapLoRA(attn.q_proj, rank, alpha)
            attn.v_proj = CheapLoRA(attn.v_proj, rank, alpha)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method == "fixa":
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, train_A=False, train_B=True)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, train_A=False, train_B=True)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method == "rcla":
        for layer in layers:
            attn = layer.self_attn
            attn.q_proj = RandomFrozenA(attn.q_proj, rank, alpha, seed=seed)
            attn.v_proj = RandomFrozenA(attn.v_proj, rank, alpha, seed=seed)
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method == "plus":
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

    elif method in {"chain", "rac"}:
        chain_like = ChainLike(net, rank, alpha)
        chain_like._inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    elif method in {"c3la", "rc3la"}:
        modest_wrap = ModestWrapper(net, rank, alpha)
        modest_wrap.inject()
        optimizer = _make_optimizer(_trainable_params(net), learningRate)

    else:
        raise ValueError(f"Unknown/unsupported method in tinyllama_trainer: {method}")

    # Reset schedule for chain/rac/c3la/rc3la
    reset_epochs: List[int] = []
    if method in {"chain", "rac", "c3la", "rc3la"}:
        reset_epochs = _count_based_reset_epochs(epochs, chainReset)

    reset_count = 0

    for ep in range(1, epochs + 1):
        # resets at START of epoch
        if ep in reset_epochs:
            reset_count += 1
            reset_seed = _reseed_for_reset(seed, reset_count)
            _set_seed(reset_seed)

            if method in {"chain", "rac"}:
                assert chain_like is not None
                chain_like.merge_and_reinject()
                optimizer = _make_optimizer(_trainable_params(net), learningRate)

            elif method == "c3la":
                assert modest_wrap is not None
                modest_wrap.advance_only()

            elif method == "rc3la":
                assert modest_wrap is not None
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