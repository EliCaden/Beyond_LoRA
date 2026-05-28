# deberta_trainer.py
# Unified DeBERTa trainer (v3-base + v2-xxlarge).
#
# Printing:
# - Start: "Run beginning with model: ..., dataset: ..., method: ..., ..." in the required order
#          and ONLY the hyperparams relevant to that method.
# - Each epoch:
#   * If no public test labels (rte, sst2, stsb): print train/val only
#   * Else: print train/val/test
# - End: print exactly "run complete"
#
# NO early stopping (not included, not mentioned)

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import DebertaV2Model, DebertaV2Tokenizer, DataCollatorWithPadding


# ----------------------------
# Utilities
# ----------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _canonicalize_dataset(name: str) -> str:
    n = name.strip().lower()

    if n == "sst":
        return "sst2"

    # common STS-B spellings
    if n in {"sts-b", "sts_b", "stsb"}:
        return "stsb"

    # Canonicalize legacy MRPC spelling.
    if n == "mrpcs":
        return "mrpc"

    return n


def _is_regression_task(task: str) -> bool:
    return _canonicalize_dataset(task) == "stsb"


def _glue_has_public_test_labels(task: str) -> bool:
    # GLUE test labels are hidden for several tasks:
    # - Do not print test metrics for:
    #   rte, sst2, stsb
    t = _canonicalize_dataset(task)
    return t not in {"rte", "sst2", "stsb"}


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm()).item()
    if denom == 0.0:
        return 0.0
    return (vx @ vy).item() / denom


def _count_based_reset_epochs(chainReset: int, epochs: int) -> List[int]:
    """
    Count-based interpretation:
      exactly `chainReset` resets spread evenly across training, in [2, epochs].
    numpy-free linspace-ish implementation.
    """
    if chainReset <= 0 or epochs < 2:
        return []

    resets = set()
    for i in range(1, chainReset + 1):
        t = i / chainReset  # (0,1]
        pos = 2 + t * (epochs - 2)  # in [2, epochs]
        ep = int(round(pos))
        ep = max(2, min(epochs, ep))
        resets.add(ep)

    return sorted(resets)


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
) -> None:
    parts = [
        f"Run beginning with model: {model}",
        f"dataset: {dataset}",
        f"method: {method}",
        f"maxLength: {maxLength}",
        f"batchSize: {batchSize}",
        f"learningRate: {learningRate}",
        f"epochs: {epochs}",
    ]

    m = method.lower().strip()

    # fft/ft: doesn't need lr_ratio, chainReset, rank, alpha
    if m in {"fft", "ft", "full", "finetune", "full_finetune"}:
        print(", ".join(parts))
        return

    # lora/vanilla: doesn't need chainReset or lr_ratio
    if m in {"lora", "vanilla"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rac: doesn't need lr_ratio
    if m in {"rac"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # plus: doesn't need chainReset
    if m in {"plus"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}", f"lr_ratio: {lr_ratio}"]
        print(", ".join(parts))
        return

    # cola/chain: doesn't need lr_ratio
    if m in {"cola", "chain"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # cheap/cla: doesn't need chainReset or lr_ratio
    if m in {"cheap", "cla"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rcla/random: doesn't need chainReset or lr_ratio
    if m in {"rcla", "random"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # fixa: doesn't need chainReset or lr_ratio
    if m in {"fixa"}:
        parts += [f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # modest/c3la: doesn't need lr_ratio
    if m in {"modest", "c3la"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # rc3la/shuffle: doesn't need lr_ratio
    if m in {"rc3la", "shuffle"}:
        parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}"]
        print(", ".join(parts))
        return

    # fallback
    parts += [f"chainReset: {chainReset}", f"rank: {rank}", f"alpha: {alpha}", f"lr_ratio: {lr_ratio}"]
    print(", ".join(parts))


# ----------------------------
# Dataset + collators
# ----------------------------

class HFDS(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds[i]


class CollatorWithLabelsCls(DataCollatorWithPadding):
    """Classification: labels -> LongTensor"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, feats):
        labs = torch.tensor([f["labels"] for f in feats], dtype=torch.long)
        batch = super().__call__(feats)
        batch["labels"] = labs
        return batch


class CollatorWithLabelsReg(DataCollatorWithPadding):
    """Regression: labels -> FloatTensor"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, feats):
        labs = torch.tensor([float(f["labels"]) for f in feats], dtype=torch.float)
        batch = super().__call__(feats)
        batch["labels"] = labs
        return batch


@dataclass
class TaskData:
    tokenizer: DebertaV2Tokenizer
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    task_type: str  # "classification" or "regression"
    num_classes: Optional[int]
    has_test_labels: bool


def _build_glue_loaders(
    glue_task: str,
    model_ckpt: str,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    task = _canonicalize_dataset(glue_task)
    ds = load_dataset("glue", task)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_ckpt)

    is_reg = _is_regression_task(task)

    def tok(batch):
        if "sentence1" in batch and "sentence2" in batch:
            out = tokenizer(batch["sentence1"], batch["sentence2"], padding=True, truncation=True, max_length=maxLength)
        elif "premise" in batch and "hypothesis" in batch:
            out = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, max_length=maxLength)
        elif "question" in batch and "sentence" in batch:
            out = tokenizer(batch["question"], batch["sentence"], padding=True, truncation=True, max_length=maxLength)
        else:
            out = tokenizer(batch.get("sentence", ""), padding=True, truncation=True, max_length=maxLength)

        out["labels"] = batch["label"]
        return out

    train = ds["train"].map(tok, batched=True).shuffle(seed=seed)
    valid = ds["validation"].map(tok, batched=True).shuffle(seed=seed)

    has_public_test = _glue_has_public_test_labels(task)
    test_loader = None
    has_test_labels = False

    if has_public_test and "test" in ds:
        test = ds["test"].map(tok, batched=True).shuffle(seed=seed)
        try:
            labs = test["labels"] if "labels" in test.column_names else test["label"]
            sample = labs[: min(256, len(labs))]
            has_test_labels = any(float(x) >= 0.0 for x in sample)
        except Exception:
            has_test_labels = False

        if has_test_labels:
            test.set_format("torch", columns=["labels", "input_ids", "attention_mask"])

    train.set_format("torch", columns=["labels", "input_ids", "attention_mask"])
    valid.set_format("torch", columns=["labels", "input_ids", "attention_mask"])

    collate = CollatorWithLabelsReg(tokenizer) if is_reg else CollatorWithLabelsCls(tokenizer)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(HFDS(train), batch_size=batchSize, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(HFDS(valid), batch_size=batchSize, shuffle=False, collate_fn=collate)

    if has_public_test and has_test_labels and "test" in ds:
        test = ds["test"].map(tok, batched=True).shuffle(seed=seed)
        test.set_format("torch", columns=["labels", "input_ids", "attention_mask"])
        test_loader = DataLoader(HFDS(test), batch_size=batchSize, shuffle=False, collate_fn=collate)

    if is_reg:
        num_classes = None
        task_type = "regression"
    else:
        num_classes = int(ds["train"].features["label"].num_classes)
        task_type = "classification"

    return TaskData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type=task_type,
        num_classes=num_classes,
        has_test_labels=bool(test_loader is not None),
    )


def _build_trec50_loaders(
    model_ckpt: str,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    ds = load_dataset("trec", "default")
    if "fine_label" in ds["train"].column_names:
        ds = ds.rename_column("fine_label", "label-fine")
    num_classes = ds["train"].features["label-fine"].num_classes

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_ckpt)

    def tok(batch):
        enc = tokenizer(batch["text"], padding=True, truncation=True, max_length=maxLength)
        enc["labels"] = batch["label-fine"]
        return enc

    tv = ds["train"].train_test_split(test_size=0.1, seed=seed)
    train = tv["train"].map(tok, batched=True).shuffle(seed=seed)
    valid = tv["test"].map(tok, batched=True).shuffle(seed=seed)
    test = ds["test"].map(tok, batched=True).shuffle(seed=seed)

    for split in (train, valid, test):
        split.set_format("torch", columns=["labels", "input_ids", "attention_mask"])

    collate = CollatorWithLabelsCls(tokenizer)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(HFDS(train), batch_size=batchSize, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(HFDS(valid), batch_size=batchSize, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(HFDS(test), batch_size=batchSize, shuffle=False, collate_fn=collate)

    return TaskData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type="classification",
        num_classes=int(num_classes),
        has_test_labels=True,
    )


def _build_paws_loaders(
    model_ckpt: str,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    ds = load_dataset("paws", "labeled_final")
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_ckpt)

    def tok(batch):
        out = tokenizer(batch["sentence1"], batch["sentence2"], padding=True, truncation=True, max_length=maxLength)
        out["labels"] = batch["label"]
        return out

    train = ds["train"].map(tok, batched=True).shuffle(seed=seed)
    valid = ds["validation"].map(tok, batched=True).shuffle(seed=seed)
    test = ds["test"].map(tok, batched=True).shuffle(seed=seed)

    for split in (train, valid, test):
        split.set_format("torch", columns=["labels", "input_ids", "attention_mask"])

    collate = CollatorWithLabelsCls(tokenizer)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(HFDS(train), batch_size=batchSize, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(HFDS(valid), batch_size=batchSize, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(HFDS(test), batch_size=batchSize, shuffle=False, collate_fn=collate)

    return TaskData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type="classification",
        num_classes=2,
        has_test_labels=True,
    )


def _build_task_data(
    dataset: str,
    model_ckpt: str,
    maxLength: int,
    batchSize: int,
    seed: int,
) -> TaskData:
    d = _canonicalize_dataset(dataset)

    if d in {"mrpc", "rte", "sst2", "qnli", "stsb"}:
        return _build_glue_loaders(d, model_ckpt, maxLength, batchSize, seed)

    if d in {"trec50", "trec"}:
        return _build_trec50_loaders(model_ckpt, maxLength, batchSize, seed)

    if d in {"paws"}:
        return _build_paws_loaders(model_ckpt, maxLength, batchSize, seed)

    raise ValueError(f"Unknown/unsupported dataset for deberta_trainer: {dataset}")


# ----------------------------
# Adapter modules (DeBERTa)
# ----------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, train_A: bool = True, train_B: bool = True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha

        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        for p in self.A.parameters():
            p.requires_grad = bool(train_A)
        for p in self.B.parameters():
            p.requires_grad = bool(train_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * (self.alpha / self.r)


class CheapLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha

        eye = torch.eye(r, base.in_features, dtype=base.weight.dtype)
        self.register_buffer("A", eye, persistent=False)

        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (self.alpha / self.r)
        return self.base(x) + delta


class RandomFrozenA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha

        gen = torch.Generator(device=base.weight.device)
        gen.manual_seed(seed)
        A = torch.randn(r, base.in_features, generator=gen, dtype=base.weight.dtype, device=base.weight.device)
        A = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("A", A, persistent=False)

        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (self.alpha / self.r)
        return self.base(x) + delta


class ModestLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.d_in = base.in_features

        self.register_buffer("offset", torch.tensor(0, dtype=torch.long))
        A0 = torch.zeros(r, self.d_in, dtype=base.weight.dtype)
        A0[torch.arange(r), torch.arange(r)] = 1.0
        self.register_buffer("A", A0, persistent=False)

        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.A)
        delta = self.B(y) * (self.alpha / self.r)
        return self.base(x) + delta

    @torch.no_grad()
    def advance_offset(self):
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
    def shuffle_offset(self):
        max_start = self.d_in - self.r
        new_off = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
        self.offset.fill_(new_off)

        Anew = torch.zeros_like(self.A)
        cols = torch.arange(self.r, device=Anew.device) + new_off
        Anew[torch.arange(self.r, device=Anew.device), cols] = 1.0
        self.A.copy_(Anew)


# ----------------------------
# Model builders
# ----------------------------

def _pick_ckpt(model: str) -> str:
    m = model.strip().lower()
    if m in {"deberta_v3_base", "deberta3", "deberta-v3-base", "deberta_v3"}:
        return "microsoft/deberta-v3-base"
    if m in {"deberta_v2_xxl", "deberta2xxl", "deberta-v2-xxlarge", "deberta_v2_xxlarge", "deberta_v2_xxl"}:
        return "microsoft/deberta-v2-xxlarge"
    raise ValueError(f"Unknown deberta model: {model}")


class DebertaCLS_FFT(nn.Module):
    def __init__(self, ckpt: str, task_type: str, num_classes: Optional[int]):
        super().__init__()
        self.task_type = task_type
        self.enc = DebertaV2Model.from_pretrained(ckpt)
        hidden = self.enc.config.hidden_size
        self.fc = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)

        if task_type == "regression":
            self.head = nn.Linear(hidden, 1)
        else:
            assert num_classes is not None
            self.head = nn.Linear(hidden, int(num_classes))

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        h = self.drop(self.act(self.fc(h)))
        out = self.head(h)
        return out


class Deberta_QV_Adapter_Model(nn.Module):
    def __init__(
        self,
        ckpt: str,
        task_type: str,
        num_classes: Optional[int],
        make_adapter: Callable[[nn.Linear], nn.Module],
        head_fc_trainable: bool,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.task_type = task_type

        self.enc = DebertaV2Model.from_pretrained(ckpt)
        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False

        for layer in self.enc.encoder.layer:
            sa = layer.attention.self
            sa.query_proj = make_adapter(sa.query_proj)
            sa.value_proj = make_adapter(sa.value_proj)

        hidden = self.enc.config.hidden_size
        self.fc = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)

        for p in self.fc.parameters():
            p.requires_grad = bool(head_fc_trainable)

        if task_type == "regression":
            self.cls = nn.Linear(hidden, 1)
        else:
            assert num_classes is not None
            self.cls = nn.Linear(hidden, int(num_classes))

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        h = self.drop(self.act(self.fc(h)))
        return self.cls(h)


class ChainLikeWrapper(nn.Module):
    """
    For methods that do:
      (1) merge delta into base weight
      (2) replace adapter with base
      (3) reinject fresh adapters
    """
    def __init__(
        self,
        ckpt: str,
        task_type: str,
        num_classes: Optional[int],
        rank: int,
        alpha: int,
        head_fc_trainable: bool,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        def make_adapter(base: nn.Linear) -> nn.Module:
            return LoRALinear(base, rank, alpha).to(base.weight.device)

        self.model = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=head_fc_trainable,
            freeze_encoder=True,
        )

    @torch.no_grad()
    def merge_and_reinject(self, new_seed: Optional[int] = None):
        # merge
        for layer in self.model.enc.encoder.layer:
            sa = layer.attention.self
            for name in ("query_proj", "value_proj"):
                lora = getattr(sa, name)
                if not isinstance(lora, LoRALinear):
                    continue
                delta = lora.B.weight @ lora.A.weight * (self.alpha / self.rank)
                lora.base.weight.data += delta
                setattr(sa, name, lora.base)

        # reinject (optionally seed torch for fresh A init)
        if new_seed is not None:
            torch.manual_seed(int(new_seed))
            torch.cuda.manual_seed_all(int(new_seed))

        for layer in self.model.enc.encoder.layer:
            sa = layer.attention.self
            sa.query_proj = LoRALinear(sa.query_proj, self.rank, self.alpha).to(sa.query_proj.weight.device)
            sa.value_proj = LoRALinear(sa.value_proj, self.rank, self.alpha).to(sa.value_proj.weight.device)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


class ModestWrapper(nn.Module):
    def __init__(self, ckpt: str, task_type: str, num_classes: Optional[int], rank: int, alpha: int, shuffle: bool):
        super().__init__()
        self.shuffle = shuffle

        def make_adapter(base: nn.Linear) -> nn.Module:
            return ModestLoRA(base, rank, alpha).to(base.weight.device)

        self.model = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=True,
            freeze_encoder=True,
        )

    @torch.no_grad()
    def reset_chain(self):
        for layer in self.model.enc.encoder.layer:
            sa = layer.attention.self
            for name in ("query_proj", "value_proj"):
                proj = getattr(sa, name)
                if isinstance(proj, ModestLoRA):
                    if self.shuffle:
                        proj.shuffle_offset()
                    else:
                        proj.advance_offset()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


# ----------------------------
# Training / evaluation
# ----------------------------

def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    task_type: str,
    dev: str,
) -> Tuple[float, float]:
    model.eval()
    totL = 0.0

    if task_type == "classification":
        totA = 0.0
        with torch.no_grad():
            for b in loader:
                x = b["input_ids"].to(dev)
                m = b["attention_mask"].to(dev)
                y = b["labels"].to(dev)
                logits = model(x, m)
                totL += loss_fn(logits, y).item()
                totA += (logits.argmax(dim=-1) == y).float().mean().item()
        return _safe_div(totL, len(loader)), _safe_div(totA, len(loader))

    preds_all = []
    y_all = []
    with torch.no_grad():
        for b in loader:
            x = b["input_ids"].to(dev)
            m = b["attention_mask"].to(dev)
            y = b["labels"].to(dev)
            out = model(x, m).squeeze(-1)
            totL += loss_fn(out, y).item()
            preds_all.append(out.detach().cpu())
            y_all.append(y.detach().cpu())
    preds = torch.cat(preds_all, dim=0)
    ys = torch.cat(y_all, dim=0)
    pear = _pearson_corr(preds, ys)
    return _safe_div(totL, len(loader)), pear


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer,
    task_type: str,
    dev: str,
) -> Tuple[float, float]:
    model.train()
    totL = 0.0

    if task_type == "classification":
        totA = 0.0
        for b in loader:
            x = b["input_ids"].to(dev)
            m = b["attention_mask"].to(dev)
            y = b["labels"].to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, m)
            L = loss_fn(logits, y)
            L.backward()
            optimizer.step()
            totL += L.item()
            totA += (logits.argmax(dim=-1) == y).float().mean().item()
        return _safe_div(totL, len(loader)), _safe_div(totA, len(loader))

    preds_all = []
    y_all = []
    for b in loader:
        x = b["input_ids"].to(dev)
        m = b["attention_mask"].to(dev)
        y = b["labels"].to(dev)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, m).squeeze(-1)
        L = loss_fn(out, y)
        L.backward()
        optimizer.step()
        totL += L.item()
        preds_all.append(out.detach().cpu())
        y_all.append(y.detach().cpu())

    preds = torch.cat(preds_all, dim=0)
    ys = torch.cat(y_all, dim=0)
    pear = _pearson_corr(preds, ys)
    return _safe_div(totL, len(loader)), pear


# ----------------------------
# Public entry
# ----------------------------

def Run(
    model: str = "deberta_v3_base",
    dataset: str = "mrpc",
    method: str = "lora",

    maxLength: int = 288,
    batchSize: int = 8,
    learningRate: float = 2e-4,
    epochs: int = 100,

    chainReset: int = 0,
    rank: int = 4,
    alpha: int = 0,
    lr_ratio: float = 1.0,

    seed: int = 42,
):
    _set_seed(seed)

    dataset = _canonicalize_dataset(dataset)
    method_l = method.strip().lower()
    ckpt = _pick_ckpt(model)

    if alpha == 0:
        alpha = 2 * rank

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
    )

    dev = _device()
    task = _build_task_data(dataset, ckpt, maxLength, batchSize, seed)
    task_type = task.task_type
    num_classes = task.num_classes

    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    if method_l in {"fft", "ft", "full", "finetune", "full_finetune"}:
        net: nn.Module = DebertaCLS_FFT(ckpt, task_type, num_classes).to(dev)
        optimizer = torch.optim.AdamW(list(net.parameters()), lr=learningRate)

    elif method_l in {"lora", "vanilla"}:
        def make_adapter(base: nn.Linear) -> nn.Module:
            return LoRALinear(base, rank, alpha, train_A=True, train_B=True).to(base.weight.device)

        net = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=False,
            freeze_encoder=True,
        ).to(dev)

        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"cheap", "cla"}:
        def make_adapter(base: nn.Linear) -> nn.Module:
            return CheapLoRA(base, rank, alpha).to(base.weight.device)

        net = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=False,
            freeze_encoder=True,
        ).to(dev)

        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"fixa"}:
        def make_adapter(base: nn.Linear) -> nn.Module:
            return LoRALinear(base, rank, alpha, train_A=False, train_B=True).to(base.weight.device)

        net = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=False,
            freeze_encoder=True,
        ).to(dev)

        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"rcla", "random"}:
        def make_adapter(base: nn.Linear) -> nn.Module:
            return RandomFrozenA(base, rank, alpha, seed=seed).to(base.weight.device)

        net = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=False,
            freeze_encoder=True,
        ).to(dev)

        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"cola", "chain"}:
        wrapper = ChainLikeWrapper(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            rank=rank,
            alpha=alpha,
            head_fc_trainable=False,
        ).to(dev)
        net = wrapper
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"rac"}:
        wrapper = ChainLikeWrapper(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            rank=rank,
            alpha=alpha,
            head_fc_trainable=False,
        ).to(dev)
        net = wrapper
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"modest", "c3la"}:
        net = ModestWrapper(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            rank=rank,
            alpha=alpha,
            shuffle=False,
        ).to(dev)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"rc3la", "shuffle"}:
        net = ModestWrapper(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            rank=rank,
            alpha=alpha,
            shuffle=True,
        ).to(dev)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

    elif method_l in {"plus"}:
        # LoRA+ convention:
        #   - A gets learningRate / lr_ratio
        #   - B gets learningRate
        #   - other trainables (if any) get learningRate
        def make_adapter(base: nn.Linear) -> nn.Module:
            return LoRALinear(base, rank, alpha, train_A=True, train_B=True).to(base.weight.device)

        net = Deberta_QV_Adapter_Model(
            ckpt=ckpt,
            task_type=task_type,
            num_classes=num_classes,
            make_adapter=make_adapter,
            head_fc_trainable=False,
            freeze_encoder=True,
        ).to(dev)

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

        if lr_ratio <= 0:
            raise ValueError("lr_ratio must be > 0 for plus method")

        optimizer = torch.optim.AdamW(
            [
                {"params": group_other, "lr": learningRate},
                {"params": group_B, "lr": learningRate},
                {"params": group_A, "lr": learningRate / lr_ratio},
            ]
        )

    else:
        raise ValueError(f"Unknown/unsupported method in deberta_trainer: {method}")

    has_test = bool(task.has_test_labels and task.test_loader is not None)

    # Count-based resets (consistent with deepseek trainer)
    modest_reset_epochs = _count_based_reset_epochs(chainReset, epochs) if method_l in {"modest", "c3la", "rc3la", "shuffle"} else []
    chain_reset_epochs = _count_based_reset_epochs(chainReset, epochs) if method_l in {"cola", "chain"} else []
    rac_reset_epochs = _count_based_reset_epochs(chainReset, epochs) if method_l == "rac" else []

    for ep in range(1, epochs + 1):
        # resets at START of epoch (count-based)
        if method_l in {"modest", "c3la", "rc3la", "shuffle"} and ep in modest_reset_epochs:
            net.reset_chain()

        elif method_l in {"cola", "chain"} and ep in chain_reset_epochs:
            # new_seed per reset to refresh A init deterministically but differently
            net.merge_and_reinject(new_seed=seed + 1000 + ep)  # type: ignore[attr-defined]
            optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

        elif method_l == "rac" and ep in rac_reset_epochs:
            net.merge_and_reinject(new_seed=seed + 2000 + ep)  # type: ignore[attr-defined]
            optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=learningRate)

        _t0 = time.time()

        tr_loss, tr_acc = _train_one_epoch(net, task.train_loader, loss_fn, optimizer, task_type, dev)
        val_loss, val_acc = _evaluate(net, task.val_loader, loss_fn, task_type, dev)

        if has_test:
            test_loss, test_acc = _evaluate(net, task.test_loader, loss_fn, task_type, dev)  # type: ignore[arg-type]

        if not has_test:
            print(
                f"epoch {ep}/{epochs}- "
                f"train_loss:{tr_loss} train_acc:{tr_acc} "
                f"val_loss:{val_loss} val_acc:{val_acc}"
            )
        else:
            print(
                f"epoch {ep}/{epochs}- "
                f"train_loss:{tr_loss} train_acc:{tr_acc} "
                f"val_loss:{val_loss} val_acc:{val_acc} "
                f"test_loss:{test_loss} test_acc:{test_acc}"
            )

        _ = time.time() - _t0  # parity (not printed)

    print("run complete")


if __name__ == "__main__":
    Run()