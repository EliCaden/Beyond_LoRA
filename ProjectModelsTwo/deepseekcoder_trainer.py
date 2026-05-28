#!/usr/bin/env python3
"""
deepseekcoder_trainer.py
Unified trainer for DeepSeek-Coder-1.3B on Django (NL→Code), covering:
  - fft/ft
  - lora/vanilla
  - chain/cola
  - cla/cheap
  - fixa
  - rac
  - plus
  - rcla/random
  - c3la/modest
  - rc3la/shuffle

Hard requirements implemented:
  - NO wandb
  - Print at start: "Run beginning with model: {model}, dataset: {dataset}, method: {method}, ..."
    and only hyperparams relevant to method (even if accepted).
  - Print each epoch (Django has test labels):
      epoch x/total- train_loss:x train_acc:x val_loss:x val_acc:x test_loss:x test_acc:x
  - Print end: "run complete"
  - alpha default: if alpha==0 => alpha = 2 * rank
  - NO early stopper (not even commented)

Convention fix:
  - chainReset is interpreted as COUNT-BASED for reset-capable methods:
      exactly `chainReset` resets spread evenly over training (epochs 2..epochs),
      not "every N epochs".
    This matches the count-based `np.linspace(...)` style used by the Modest TREC50 script.
"""

import math
import random
import time
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Dataset utilities (Django)
# ----------------------------

LIKELY_INPUT_KEYS  = ["question", "nl", "intent", "prompt", "src", "source", "docstring", "description"]
LIKELY_TARGET_KEYS = ["code", "snippet", "target", "solution", "tgt", "canonical_solution"]

def _infer_fields(example: Dict) -> Tuple[str, str]:
    keys = set(example.keys())
    ik = next((k for k in LIKELY_INPUT_KEYS  if k in keys), None)
    tk = next((k for k in LIKELY_TARGET_KEYS if k in keys), None)
    if ik is None or tk is None:
        raise KeyError(
            f"Could not infer fields from example keys={sorted(keys)}. "
            f"Tried input={LIKELY_INPUT_KEYS}, target={LIKELY_TARGET_KEYS}."
        )
    return ik, tk


class NLCodeCausalLMDataset(Dataset):
    """
    Builds causal-LM examples by concatenating:
      prompt = prefix + NL + suffix
      full   = prompt + code
    Masks all prompt tokens so loss/acc are only computed on code tokens.
    """
    def __init__(
        self,
        hf_split,
        tokenizer,
        max_length: int,
        prefix: str = "\"\"\"Task: ",
        suffix: str = "\"\"\"\n# Solution\n"
    ):
        self.items: List[Tuple[List[int], int]] = []
        self.tokenizer = tokenizer

        if len(hf_split) == 0:
            return

        in_key, tgt_key = _infer_fields(hf_split[0])

        for ex in hf_split:
            nl   = ex.get(in_key, None)
            code = ex.get(tgt_key, None)
            if not isinstance(nl, str) or not isinstance(code, str):
                continue

            prompt = f"{prefix}{nl}{suffix}"
            full   = prompt + code

            enc_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
            enc_full   = tokenizer(full,   truncation=True, max_length=max_length, padding=False)

            prompt_len = len(enc_prompt["input_ids"])
            full_ids   = enc_full["input_ids"]
            if prompt_len >= len(full_ids):
                continue

            self.items.append((full_ids, prompt_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        full_ids, prompt_len = self.items[idx]
        return {"input_ids": full_ids, "prompt_len": prompt_len}


def collate(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    ids_list    = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)

    padded = torch.nn.utils.rnn.pad_sequence(
        ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention = padded.ne(tokenizer.pad_token_id).long()

    labels = padded.clone()
    for i, pl in enumerate(prompt_lens.tolist()):
        labels[i, :pl] = -100
        labels[i, attention[i] == 0] = -100

    return {"input_ids": padded, "attention_mask": attention, "labels": labels, "prompt_len": prompt_lens}


def _ensure_splits(dsdict: DatasetDict, seed: int):
    """
    Ensure train/validation/test exist. If not, create deterministic splits.
    """
    if set(dsdict.keys()) >= {"train", "validation", "test"}:
        return dsdict["train"], dsdict["validation"], dsdict["test"]

    base = dsdict["train"] if "train" in dsdict else dsdict[list(dsdict.keys())[0]]
    split1 = base.train_test_split(test_size=0.2, seed=seed)
    tv, test = split1["train"], split1["test"]
    split2 = tv.train_test_split(test_size=0.125, seed=seed)  # 0.125 of 0.8 ≈ 0.10
    train, val = split2["train"], split2["test"]
    return train, val, test


# ----------------------------
# Reset scheduling (count-based)
# ----------------------------

def _count_based_reset_epochs(chainReset: int, epochs: int) -> List[int]:
    """
    Interpret chainReset as "number of resets" spread evenly over training.
    Matches np.linspace(2, epochs, num=chainReset, dtype=int) style, without numpy.
    - Resets are in [2, epochs] (never at epoch 1)
    - Returned list is sorted, unique
    """
    if chainReset <= 0 or epochs < 2:
        return []

    # Evenly spaced points between 2 and epochs inclusive.
    # Use rounding to emulate dtype=int behavior reasonably, then de-dup.
    resets = set()
    for i in range(1, chainReset + 1):
        # map i in [1..chainReset] to t in [0..1]
        t = i / chainReset
        # position in [2..epochs]
        pos = 2 + t * (epochs - 2)
        ep = int(round(pos))
        ep = max(2, min(epochs, ep))
        resets.add(ep)

    return sorted(resets)


# ----------------------------
# Model adapter plumbing
# ----------------------------

def _attn_modules(model) -> List[nn.Module]:
    """
    DeepSeek-Coder / LLaMA-like: model.model.layers[*].self_attn with q_proj/v_proj as nn.Linear.
    """
    return [layer.self_attn for layer in model.model.layers]


class LoRALinear(nn.Module):
    """
    Standard LoRA: base(x) + (alpha/r) * B(A(x))
    A and B are trainable by default; can freeze A for FixA/other variants.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, *, freeze_A: bool = False, init_A: str = "kaiming", seed: Optional[int] = None):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        if r > base.in_features:
            raise ValueError(f"rank r={r} exceeds in_features={base.in_features}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)

        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)

        # init
        if seed is not None:
            torch.manual_seed(int(seed))

        if init_A == "kaiming":
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        elif init_A == "zeros":
            nn.init.zeros_(self.A.weight)
        elif init_A == "normal":
            with torch.no_grad():
                self.A.weight.normal_(mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init_A='{init_A}'")

        nn.init.zeros_(self.B.weight)

        if freeze_A:
            for p in self.A.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * (self.alpha / self.r)

    @torch.no_grad()
    def merge_into_base(self):
        # base.W += (alpha/r) * (B @ A)
        delta = (self.B.weight @ self.A.weight) * (self.alpha / self.r)  # (out, in)
        self.base.weight.data += delta.to(self.base.weight.dtype)


class CheapLinear(nn.Module):
    """
    Cheap/CLA: base(x) + (alpha/r) * B(S(x)), where S selects a fixed contiguous r-wide block.
    Only B is trainable. Selection is via slicing x[:, off:off+r].
    We keep offset fixed at 0 for this "cheap" variant.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        if r <= 0:
            raise ValueError("Cheap rank must be > 0")
        if r > base.in_features:
            raise ValueError(f"rank r={r} exceeds in_features={base.in_features}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.in_features = base.in_features
        self.offset = 0  # fixed

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        orig = x.shape[:-1]
        x_flat = x.view(-1, x.size(-1))
        s = x_flat[:, self.offset:self.offset + self.r]
        upd = self.B(s) * (self.alpha / self.r)
        out = self.base(x_flat) + upd
        return out.view(*orig, out.size(-1))

    @torch.no_grad()
    def merge_into_base(self):
        off = self.offset
        self.base.weight.data[:, off:off + self.r] += (self.B.weight.data * (self.alpha / self.r)).to(self.base.weight.dtype)


class RandomSelectLinear(nn.Module):
    """
    RCLA/Random: base(x) + (alpha/r) * B(S_rand(x))
    - S_rand selects a fixed random subset of r input dimensions (frozen)
    - Only B is trainable
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, seed: int):
        super().__init__()
        if r <= 0:
            raise ValueError("Random rank must be > 0")
        if r > base.in_features:
            raise ValueError(f"rank r={r} exceeds in_features={base.in_features}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.in_features = base.in_features

        rng = random.Random(int(seed))
        idx = list(range(self.in_features))
        rng.shuffle(idx)
        sel = sorted(idx[:self.r])
        self.register_buffer("sel_idx", torch.tensor(sel, dtype=torch.long), persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        orig = x.shape[:-1]
        x_flat = x.view(-1, x.size(-1))
        s = x_flat.index_select(dim=1, index=self.sel_idx)  # (N, r)
        upd = self.B(s) * (self.alpha / self.r)
        out = self.base(x_flat) + upd
        return out.view(*orig, out.size(-1))

    @torch.no_grad()
    def merge_into_base(self):
        delta_cols = (self.B.weight.data * (self.alpha / self.r)).to(self.base.weight.dtype)  # (out, r)
        self.base.weight.data.index_add_(dim=1, index=self.sel_idx, source=delta_cols)


class ModestLinear(nn.Module):
    """
    C3LA/Modest: base(x) + (alpha/r) * B(S_window(x))
    - S_window selects a contiguous r-wide block
    - Only B trainable
    - reset advances the window by r (wrap)
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        if r <= 0:
            raise ValueError("Modest rank must be > 0")
        if r > base.in_features:
            raise ValueError(f"rank r={r} exceeds in_features={base.in_features}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.in_features = base.in_features
        self.register_buffer("offset", torch.tensor(0, dtype=torch.long), persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        orig = x.shape[:-1]
        x_flat = x.view(-1, x.size(-1))
        off = int(self.offset.item())
        s = x_flat[:, off:off + self.r]
        upd = self.B(s) * (self.alpha / self.r)
        out = self.base(x_flat) + upd
        return out.view(*orig, out.size(-1))

    @torch.no_grad()
    def advance_offset(self):
        max_start = self.in_features - self.r
        new_off = int(self.offset.item()) + self.r
        if new_off > max_start:
            new_off = 0
        self.offset.fill_(new_off)

    @torch.no_grad()
    def merge_into_base(self):
        off = int(self.offset.item())
        self.base.weight.data[:, off:off + self.r] += (self.B.weight.data * (self.alpha / self.r)).to(self.base.weight.dtype)


class ShuffleLinear(nn.Module):
    """
    RC3LA/Shuffle:
      base(x) + (alpha/r) * B(S_window(x))
    - Only B trainable
    - reset: fuse current delta into base and randomly re-place the window, reset B->0
    """
    def __init__(self, base: nn.Linear, r: int, alpha: int, seed: int):
        super().__init__()
        if r <= 0:
            raise ValueError("Shuffle rank must be > 0")
        if r > base.in_features:
            raise ValueError(f"rank r={r} exceeds in_features={base.in_features}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.in_features = base.in_features
        self.register_buffer("offset", torch.tensor(0, dtype=torch.long), persistent=False)

        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

        self._rng = random.Random(int(seed))
        self._sample_new_offset(init=True)

    def _sample_new_offset(self, init: bool):
        span = max(1, self.in_features - self.r + 1)
        if span == 1:
            new_off = 0
        else:
            prev = int(self.offset.item())
            new_off = self._rng.randrange(0, span)
            if (not init) and (new_off == prev) and span > 1:
                new_off = (new_off + self._rng.randrange(1, span)) % span
        self.offset.fill_(new_off)

    def forward(self, x):
        orig = x.shape[:-1]
        x_flat = x.view(-1, x.size(-1))
        off = int(self.offset.item())
        s = x_flat[:, off:off + self.r]
        upd = self.B(s) * (self.alpha / self.r)
        out = self.base(x_flat) + upd
        return out.view(*orig, out.size(-1))

    @torch.no_grad()
    def fuse_and_shuffle(self):
        off = int(self.offset.item())
        self.base.weight.data[:, off:off + self.r] += (self.B.weight.data * (self.alpha / self.r)).to(self.base.weight.dtype)
        nn.init.zeros_(self.B.weight)
        self._sample_new_offset(init=False)

    @torch.no_grad()
    def merge_into_base(self):
        off = int(self.offset.item())
        self.base.weight.data[:, off:off + self.r] += (self.B.weight.data * (self.alpha / self.r)).to(self.base.weight.dtype)


def _inject_adapters(model, adapter_kind: str, rank: int, alpha: int, seed: int):
    """
    Inject adapter modules into q_proj and v_proj across all layers.
    """
    attns = _attn_modules(model)
    rng = random.Random(int(seed))

    for attn in attns:
        # q_proj
        if isinstance(attn.q_proj, nn.Linear):
            if adapter_kind in ("lora", "chain", "rac", "plus"):
                attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, freeze_A=False, init_A="kaiming", seed=rng.randrange(10**9))
            elif adapter_kind == "fixa":
                attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, freeze_A=True, init_A="kaiming", seed=rng.randrange(10**9))
            elif adapter_kind == "cla":
                attn.q_proj = CheapLinear(attn.q_proj, rank, alpha)
            elif adapter_kind == "rcla":
                attn.q_proj = RandomSelectLinear(attn.q_proj, rank, alpha, seed=rng.randrange(10**9))
            elif adapter_kind == "c3la":
                attn.q_proj = ModestLinear(attn.q_proj, rank, alpha)
            elif adapter_kind == "rc3la":
                attn.q_proj = ShuffleLinear(attn.q_proj, rank, alpha, seed=rng.randrange(10**9))
            else:
                raise ValueError(f"Unknown adapter_kind={adapter_kind}")

        # v_proj
        if isinstance(attn.v_proj, nn.Linear):
            if adapter_kind in ("lora", "chain", "rac", "plus"):
                attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, freeze_A=False, init_A="kaiming", seed=rng.randrange(10**9))
            elif adapter_kind == "fixa":
                attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, freeze_A=True, init_A="kaiming", seed=rng.randrange(10**9))
            elif adapter_kind == "cla":
                attn.v_proj = CheapLinear(attn.v_proj, rank, alpha)
            elif adapter_kind == "rcla":
                attn.v_proj = RandomSelectLinear(attn.v_proj, rank, alpha, seed=rng.randrange(10**9))
            elif adapter_kind == "c3la":
                attn.v_proj = ModestLinear(attn.v_proj, rank, alpha)
            elif adapter_kind == "rc3la":
                attn.v_proj = ShuffleLinear(attn.v_proj, rank, alpha, seed=rng.randrange(10**9))
            else:
                raise ValueError(f"Unknown adapter_kind={adapter_kind}")


def _iter_adapters(model) -> List[nn.Module]:
    mods = []
    for attn in _attn_modules(model):
        for name in ("q_proj", "v_proj"):
            m = getattr(attn, name, None)
            if isinstance(m, (LoRALinear, CheapLinear, RandomSelectLinear, ModestLinear, ShuffleLinear)):
                mods.append(m)
    return mods


@torch.no_grad()
def _merge_all_adapters_into_base(model):
    for m in _iter_adapters(model):
        m.merge_into_base()


@torch.no_grad()
def _reset_c3la_windows(model):
    for m in _iter_adapters(model):
        if isinstance(m, ModestLinear):
            m.advance_offset()


@torch.no_grad()
def _reset_rc3la_shuffle(model):
    for m in _iter_adapters(model):
        if isinstance(m, ShuffleLinear):
            m.fuse_and_shuffle()


@torch.no_grad()
def _merge_reinject_lora_style(model, adapter_kind: str, rank: int, alpha: int, seed: int):
    """
    For chain/rac: merge adapters into base weights, then reinject fresh adapters.
    """
    _merge_all_adapters_into_base(model)

    # Strip adapters back to bare base Linear modules
    for attn in _attn_modules(model):
        if isinstance(attn.q_proj, (LoRALinear, CheapLinear, RandomSelectLinear, ModestLinear, ShuffleLinear)):
            attn.q_proj = attn.q_proj.base
        if isinstance(attn.v_proj, (LoRALinear, CheapLinear, RandomSelectLinear, ModestLinear, ShuffleLinear)):
            attn.v_proj = attn.v_proj.base

    _inject_adapters(model, adapter_kind=adapter_kind, rank=rank, alpha=alpha, seed=seed)


# ----------------------------
# Training/eval helpers
# ----------------------------

def token_acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    mask = labels.ne(-100)
    if not mask.any():
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()


@torch.no_grad()
def eval_causal_lm(model, loader, device, amp_dtype) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    use_amp = torch.cuda.is_available()
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

    with ctx:
        for batch in loader:
            inp = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            out = model(**inp)
            total_loss += float(out.loss.item())
            total_acc  += float(token_acc_from_logits(out.logits, inp["labels"]))
            n += 1

    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_acc / n


# ----------------------------
# Printing mechanics
# ----------------------------

def _print_run_beginning(
    model: str,
    dataset: str,
    method: str,
    *,
    maxLength: int,
    batchSize: int,
    learningRate: float,
    epochs: int,
    chainReset: int,
    rank: int,
    alpha: int,
    lr_ratio: float,
    seed: int,
):
    method_l = method.lower().strip()

    include = {
        "maxLength": True,
        "batchSize": True,
        "learningRate": True,
        "epochs": True,
        "seed": True,
        "chainReset": False,
        "rank": False,
        "alpha": False,
        "lr_ratio": False,
    }

    if method_l in ("fft", "ft"):
        pass
    elif method_l in ("lora", "vanilla"):
        include["rank"] = True
        include["alpha"] = True
    elif method_l == "rac":
        include["chainReset"] = True
        include["rank"] = True
        include["alpha"] = True
    elif method_l == "plus":
        include["rank"] = True
        include["alpha"] = True
        include["lr_ratio"] = True
    elif method_l in ("cola", "chain"):
        include["chainReset"] = True
        include["rank"] = True
        include["alpha"] = True
    elif method_l in ("cheap", "cla"):
        include["rank"] = True
        include["alpha"] = True
    elif method_l in ("rcla", "random"):
        include["rank"] = True
        include["alpha"] = True
    elif method_l == "fixa":
        include["rank"] = True
        include["alpha"] = True
    elif method_l in ("modest", "c3la"):
        include["chainReset"] = True
        include["rank"] = True
        include["alpha"] = True
    elif method_l in ("rc3la", "shuffle"):
        include["chainReset"] = True
        include["rank"] = True
        include["alpha"] = True
    else:
        raise ValueError(f"Unknown method for printing: {method}")

    parts = [f"Run beginning with model: {model}, dataset: {dataset}, method: {method}"]
    if include["maxLength"]:
        parts.append(f"maxLength: {maxLength}")
    if include["batchSize"]:
        parts.append(f"batchSize: {batchSize}")
    if include["learningRate"]:
        parts.append(f"learningRate: {learningRate}")
    if include["epochs"]:
        parts.append(f"epochs: {epochs}")
    if include["chainReset"]:
        parts.append(f"chainReset: {chainReset}")
    if include["rank"]:
        parts.append(f"rank: {rank}")
    if include["alpha"]:
        parts.append(f"alpha: {alpha}")
    if include["lr_ratio"]:
        parts.append(f"lr_ratio: {lr_ratio}")
    if include["seed"]:
        parts.append(f"seed: {seed}")

    print(", ".join(parts))


# ----------------------------
# Unified Run()
# ----------------------------

def Run(
    model: str = "deepseek-ai/deepseek-coder-1.3b-base",
    dataset: str = "django",
    method: str = "lora",
    maxLength: int = 2048,
    batchSize: int = 4,
    learningRate: float = 2e-4,
    epochs: int = 20,
    chainReset: int = 5,
    rank: int = 8,
    alpha: int = 0,
    lr_ratio: float = 4.0,
    seed: int = 42,
):
    # Repro
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = torch.cuda.is_available()

    # alpha rule
    if alpha == 0:
        alpha = 2 * rank

    # Only supported dataset in this trainer
    dataset_l = dataset.lower().strip()
    if dataset_l != "django":
        raise ValueError(f"deepseekcoder_trainer.py supports only dataset='django' (got '{dataset}')")

    method_l = method.lower().strip()

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

    # Load data
    ds = load_dataset("AhmedSSoliman/DJANGO")
    train_raw, val_raw, test_raw = _ensure_splits(ds, seed=seed)

    # Tokenizer/model
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    net = AutoModelForCausalLM.from_pretrained(model)
    net.resize_token_embeddings(len(tok))
    net.config.pad_token_id = tok.pad_token_id
    net.to(device)

    prefix = "\"\"\"Task: "
    suffix = "\"\"\"\n# Solution\n"

    train_ds = NLCodeCausalLMDataset(train_raw, tok, maxLength, prefix, suffix)
    val_ds   = NLCodeCausalLMDataset(val_raw,   tok, maxLength, prefix, suffix)
    test_ds  = NLCodeCausalLMDataset(test_raw,  tok, maxLength, prefix, suffix)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batchSize, shuffle=True,
        collate_fn=lambda b: collate(b, tok),
        generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batchSize, shuffle=False,
        collate_fn=lambda b: collate(b, tok),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batchSize, shuffle=False,
        collate_fn=lambda b: collate(b, tok),
    )

    # Configure training by method
    for p in net.parameters():
        p.requires_grad = False

    adapter_kind: Optional[str] = None

    if method_l in ("fft", "ft"):
        for p in net.parameters():
            p.requires_grad = True

    elif method_l in ("lora", "vanilla"):
        adapter_kind = "lora"
        _inject_adapters(net, adapter_kind=adapter_kind, rank=rank, alpha=alpha, seed=seed)

    elif method_l in ("chain", "cola"):
        adapter_kind = "chain"
        _inject_adapters(net, adapter_kind="chain", rank=rank, alpha=alpha, seed=seed)

    elif method_l in ("cheap", "cla"):
        adapter_kind = "cla"
        _inject_adapters(net, adapter_kind="cla", rank=rank, alpha=alpha, seed=seed)

    elif method_l == "fixa":
        adapter_kind = "fixa"
        _inject_adapters(net, adapter_kind="fixa", rank=rank, alpha=alpha, seed=seed)

    elif method_l == "rac":
        adapter_kind = "rac"
        _inject_adapters(net, adapter_kind="rac", rank=rank, alpha=alpha, seed=seed)

    elif method_l == "plus":
        adapter_kind = "plus"
        _inject_adapters(net, adapter_kind="plus", rank=rank, alpha=alpha, seed=seed)

    elif method_l in ("rcla", "random"):
        adapter_kind = "rcla"
        _inject_adapters(net, adapter_kind="rcla", rank=rank, alpha=alpha, seed=seed)

    elif method_l in ("c3la", "modest"):
        adapter_kind = "c3la"
        _inject_adapters(net, adapter_kind="c3la", rank=rank, alpha=alpha, seed=seed)

    elif method_l in ("rc3la", "shuffle"):
        adapter_kind = "rc3la"
        _inject_adapters(net, adapter_kind="rc3la", rank=rank, alpha=alpha, seed=seed)

    else:
        raise ValueError(
            "Unknown method. Expected one of: "
            "fft, ft, lora, vanilla, chain, cola, cla, cheap, fixa, rac, plus, rcla, random, c3la, modest, rc3la, shuffle"
        )

    # Reset epochs (COUNT-BASED) for methods that use chainReset
    reset_epochs: List[int] = []
    if method_l in ("chain", "cola", "rac", "c3la", "modest", "rc3la", "shuffle"):
        reset_epochs = _count_based_reset_epochs(chainReset, epochs)

    # Build optimizer
    def _trainable_params():
        return [p for p in net.parameters() if p.requires_grad]

    if method_l == "plus":
        # LoRA+ convention:
        #   B uses learningRate
        #   A uses learningRate / lr_ratio
        A_params = []
        B_params = []
        other_params = []

        for m in _iter_adapters(net):
            if isinstance(m, LoRALinear):
                A_params += list(m.A.parameters())
                B_params += list(m.B.parameters())
            else:
                other_params += [p for p in m.parameters() if p.requires_grad]

        for p in net.parameters():
            if p.requires_grad and (p not in A_params) and (p not in B_params):
                other_params.append(p)

        if lr_ratio <= 0:
            raise ValueError("lr_ratio must be > 0 for plus method")

        param_groups = []
        if len(A_params) > 0:
            param_groups.append({"params": A_params, "lr": learningRate / lr_ratio})
        if len(B_params) > 0:
            param_groups.append({"params": B_params, "lr": learningRate})
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": learningRate})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)

    else:
        optimizer = torch.optim.AdamW(_trainable_params(), lr=learningRate, betas=(0.9, 0.95), weight_decay=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Training loop
    for ep in range(1, epochs + 1):
        # Resets at START of selected epochs (count-based)
        if ep in reset_epochs:
            if method_l in ("chain", "cola"):
                _merge_reinject_lora_style(net, adapter_kind="chain", rank=rank, alpha=alpha, seed=seed + ep)
                optimizer = torch.optim.AdamW(_trainable_params(), lr=learningRate, betas=(0.9, 0.95), weight_decay=0.1)

            elif method_l == "rac":
                _merge_reinject_lora_style(net, adapter_kind="rac", rank=rank, alpha=alpha, seed=seed + 999 + ep)
                optimizer = torch.optim.AdamW(_trainable_params(), lr=learningRate, betas=(0.9, 0.95), weight_decay=0.1)

            elif method_l in ("c3la", "modest"):
                _reset_c3la_windows(net)

            elif method_l in ("rc3la", "shuffle"):
                _reset_rc3la_shuffle(net)
                optimizer = torch.optim.AdamW(_trainable_params(), lr=learningRate, betas=(0.9, 0.95), weight_decay=0.1)

        net.train()
        tr_loss = 0.0
        tr_acc = 0.0
        steps = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            inp = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    out = net(**inp)
                    loss = out.loss
                acc = token_acc_from_logits(out.logits.detach(), inp["labels"])

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = net(**inp)
                loss = out.loss
                acc = token_acc_from_logits(out.logits.detach(), inp["labels"])
                loss.backward()
                optimizer.step()

            tr_loss += float(loss.item())
            tr_acc  += float(acc)
            steps += 1

        train_loss = tr_loss / steps if steps else 0.0
        train_acc  = tr_acc / steps if steps else 0.0

        val_loss, val_acc   = eval_causal_lm(net, val_loader, device, amp_dtype)
        test_loss, test_acc = eval_causal_lm(net, test_loader, device, amp_dtype)

        print(
            f"epoch {ep}/{epochs}- "
            f"train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} "
            f"val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} "
            f"test_loss:{test_loss:.4f} test_acc:{test_acc:.4f}"
        )

    print("run complete")


if __name__ == "__main__":
    Run()