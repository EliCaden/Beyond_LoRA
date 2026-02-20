# data/e2e.py
# E2E NLG (Novikova et al., 2017) — restaurant domain, ~42k/4.6k/4.6k train/val/test.
# Each input (x) is a sequence of slot–value pairs; each x can have multiple natural-language
# references (y). License: Creative Commons BY-NC-SA 4.0.
#
# This DataModule:
#   • Canonicalizes MR as a sorted "slot = value" sequence.
#   • Supports multiple references per MR:
#       - ref_strategy="all" (default): expand to one training example per reference.
#       - ref_strategy="first": use the first reference only.
#       - ref_strategy="random": pick one reference deterministically per item (seeded).
#   • Builds causal-LM examples with prompt = serialized MR and target = reference text.
#     Labels for the prompt tokens are -100 (ignored); the target side (incl. EOS) is learned.

from __future__ import annotations

import os
import random
import re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .base import register_data, BaseDataModule


# ----------------------
# Utilities
# ----------------------

def _seed_worker(worker_id: int):
    # Reproducible dataloader workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_e2e() -> DatasetDict:
    """
    Prefer common HF IDs; keep robust fallbacks.
    Note: E2E needs trust_remote_code to build splits reliably.
    """
    tried: List[Tuple[str, str]] = []
    for did in ["GEM/e2e_nlg", "e2e_nlg", "e2e_nlg_cleaned"]:
        try:
            ds = load_dataset(did, trust_remote_code=True)
            return DatasetDict(ds) if not isinstance(ds, DatasetDict) else ds
        except Exception as e:
            tried.append((did, str(e)))
    raise RuntimeError(f"Could not load E2E dataset from any known ID: {tried}")

def _detect_fields(columns: List[str]) -> Dict[str, str]:
    """
    Map dataset columns to MR and references.
    Supports both single-reference ("reference") and multi-reference ("references") layouts.
    """
    cset = set(columns)
    mr_keys  = ["meaning_representation", "mr", "source", "input", "meaning", "semantics"]
    ref1_key = next((k for k in ["human_reference", "reference", "target", "text", "realization", "output"] if k in cset), None)
    refs_key = "references" if "references" in cset else None
    mr_key   = next((k for k in mr_keys if k in cset), None)

    if mr_key is None or (ref1_key is None and refs_key is None):
        raise ValueError(f"[E2E] Unrecognized columns: {columns}")
    return {"mr": mr_key, "ref": ref1_key, "refs": refs_key}


# Patterns like: slot[value]  (common in E2E releases)
_SLOTVAL = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9-]*)\s*\[\s*([^\]]+?)\s*\]\s*")

def _parse_mr_to_pairs(mr: Any) -> List[Tuple[str, str]]:
    """
    Convert an MR into a list of (slot, value) pairs.
    Accepts string in 'slot[value], slot[value], ...' format, dicts, or pre-tokenized lists.
    """
    # Dict-like
    if isinstance(mr, dict):
        return [(str(k), str(v)) for k, v in mr.items()]

    # List-like of "slot[value]" or of (slot, value)
    if isinstance(mr, (list, tuple)):
        pairs: List[Tuple[str, str]] = []
        for item in mr:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                pairs.append((str(k), str(v)))
            elif isinstance(item, str):
                m = _SLOTVAL.fullmatch(item.strip())
                if m:
                    pairs.append((m.group(1), m.group(2)))
                else:
                    # fallback: treat whole string as one value with unknown slot
                    pairs.append(("mr", item.strip()))
            else:
                pairs.append(("mr", str(item)))
        return pairs

    # String: try to find repeated slot[value] patterns
    if isinstance(mr, str):
        s = mr.strip()
        pairs = _SLOTVAL.findall(s)
        if pairs:
            return [(str(k), str(v)) for (k, v) in pairs]
        # Fallback: split on commas/semicolons into name=value-ish tokens
        toks = re.split(r"[;,]\s*", s)
        out: List[Tuple[str, str]] = []
        for t in toks:
            if not t:
                continue
            if "[" in t and "]" in t:
                m = _SLOTVAL.search(t)
                if m:
                    out.append((m.group(1), m.group(2)))
                else:
                    out.append(("mr", t))
            else:
                # try name=value or name: value
                m2 = re.match(r"\s*([^:=]+)\s*[:=]\s*(.+)\s*$", t)
                if m2:
                    out.append((m2.group(1).strip(), m2.group(2).strip()))
                else:
                    out.append(("mr", t.strip()))
        return out

    # Anything else
    return [("mr", str(mr))]


def _serialize_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    Canonical text form used in prompts: 'slot = value ; slot2 = value2 ; ...'
    Slots are lower-cased; pairs are sorted by slot for stability.
    """
    pairs = [(k.strip().lower(), v.strip()) for (k, v) in pairs]
    pairs.sort(key=lambda kv: kv[0])
    return " ; ".join(f"{k} = {v}" for (k, v) in pairs)


def _prompt_and_target_from_mr_ref(mr: Any, ref: str) -> Tuple[str, str]:
    pr = _serialize_pairs(_parse_mr_to_pairs(mr))
    prompt = f"MR: {pr}\nTEXT:"
    target = " " + ref.strip()
    return prompt, target


def _explode_multi_refs(ds: Dataset, fields: Dict[str, str], strategy: str, seed: int) -> Dataset:
    """
    Normalize dataset to one (MR, single reference) per row.
    - Prefer 'references' only when non-empty for that row; otherwise fall back to single 'ref' column.
    - Accept GEM-style dict refs like {'text': '...'} by unwrapping to the text.
    """
    from datasets import Dataset as HFDataset

    mr_key, ref1_key, refs_key = fields["mr"], fields["ref"], fields["refs"]
    rng = random.Random(seed)

    # Column-wise access
    mrs = ds[mr_key]
    refs_col = ds[refs_key] if (refs_key and refs_key in ds.column_names) else None
    ref1_col = ds[ref1_key] if (ref1_key and ref1_key in ds.column_names) else None

    def _as_text(x):
        # unwrap GEM dicts like {'text': '...'}; otherwise coerce to str
        if isinstance(x, dict):
            return x.get("text") or x.get("reference") or x.get("human_reference") or str(x)
        return str(x)

    rows_mr, rows_ref = [], []

    n = len(ds)
    for i in range(n):
        mr = mrs[i]
        picked = False

        # Try multi-ref first if present AND non-empty for this row
        if refs_col is not None:
            refs = refs_col[i]
            if isinstance(refs, (list, tuple)) and len(refs) > 0:
                if strategy == "all":
                    for r in refs:
                        rows_mr.append(mr)
                        rows_ref.append(_as_text(r))
                elif strategy == "first":
                    rows_mr.append(mr)
                    rows_ref.append(_as_text(refs[0]))
                elif strategy == "random":
                    rows_mr.append(mr)
                    rows_ref.append(_as_text(rng.choice(list(refs))))
                else:
                    raise ValueError(f"Unknown ref_strategy: {strategy}")
                picked = True

        # Fallback to single-reference column when needed
        if (not picked) and (ref1_col is not None):
            r = ref1_col[i]
            # some mirrors have empty strings; skip only truly missing
            if r is not None:
                rows_mr.append(mr)
                rows_ref.append(_as_text(r))
                picked = True

        # If neither gave us a reference, just skip the row

    # Final sanity: avoid returning empty dataset silently
    if len(rows_mr) == 0:
        raise RuntimeError(
            f"[E2E] no usable references found — check split schema. "
            f"columns={ds.column_names}, fields={fields}"
        )

    return HFDataset.from_dict({mr_key: rows_mr, "reference": rows_ref})



class _E2ECollator:
    """
    Pads 'input_ids', 'attention_mask', and 'labels' to the longest in the batch.
    Assumes tokenizer.pad_token_id is set (we set it to eos if missing).
    """
    def __init__(self, tok: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = 8):
        self.tok = tok
        self.pad_id = tok.pad_token_id
        self.label_pad = -100
        self.multiple = pad_to_multiple_of

    def _pad_to_multiple(self, length: int) -> int:
        if not self.multiple or self.multiple <= 1:
            return length
        rem = length % self.multiple
        return length if rem == 0 else (length + (self.multiple - rem))

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        max_len = self._pad_to_multiple(max_len)
        input_ids, attn, labels = [], [], []
        for ex in batch:
            ids = ex["input_ids"]
            lab = ex["labels"]
            pad = max(0, max_len - len(ids))
            input_ids.append(ids + [self.pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [self.label_pad] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ----------------------
# DataModule
# ----------------------

@register_data("e2e_nlg")
class E2EDataModule(BaseDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_length: int = 256,
        seed: int = 42,
        num_workers: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 8,
        drop_last: bool = False,
        ref_strategy: str = "all",   # "all" | "first" | "random"
        fallback_test_from_val: bool = True,
        val_test_split: float = 0.5,
        dataset: Optional[DatasetDict] = None,
    ):
        super().__init__(batch_size=batch_size, max_length=max_length, seed=seed)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.num_workers = (
            int(num_workers) if num_workers is not None
            else max(1, min(8, (os.cpu_count() or 4) // 2))
        )
        self.pad_to_multiple_of = pad_to_multiple_of
        self.drop_last = bool(drop_last)
        self.ref_strategy = str(ref_strategy).lower()
        self.fallback_test_from_val = bool(fallback_test_from_val)
        self.val_test_split = float(val_test_split)
        self._dataset_override = dataset

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.collator = _E2ECollator(self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

    # ---- plumbing ----
    def setup(self):
        raw = self._dataset_override if self._dataset_override is not None else _load_e2e()
        # Helper: check split exists and is non-empty (when length is defined)
        def _pick(name: str):
            ds = raw.get(name)
            if ds is None:
                return None
            try:
                if len(ds) == 0:
                    return None
            except TypeError:
                # iterable dataset => accept
                pass
            return ds

        # Use only canonical splits; avoid challenge_* for training
        train_split = _pick("train")
        if train_split is None:
            raise RuntimeError(f"[E2E] non-empty 'train' split not found. Available: {list(raw.keys())}")

        val_split  = _pick("validation") or _pick("dev")
        test_split = _pick("test") or val_split

        # Detect field names independently for each split (schemas differ!)
        fields_tr = _detect_fields(train_split.column_names)
        fields_va = _detect_fields(val_split.column_names) if val_split is not None else fields_tr
        fields_te = _detect_fields(test_split.column_names) if test_split is not None else fields_va

        # Normalize to {<mr>, "reference"} per split
        tr = _explode_multi_refs(train_split, fields_tr, self.ref_strategy, self.seed)
        va = _explode_multi_refs(val_split,  fields_va, self.ref_strategy, self.seed) if val_split  is not None else None
        te = _explode_multi_refs(test_split, fields_te, self.ref_strategy, self.seed) if test_split is not None else None

        # Sanity check — training must not be empty
        if hasattr(tr, "__len__") and len(tr) == 0:
            raise RuntimeError(
                f"[E2E] train split expanded to zero rows. "
                f"train columns={train_split.column_names}, fields_tr={fields_tr}"
            )

        mr_tr, mr_va, mr_te = fields_tr["mr"], fields_va["mr"], fields_te["mr"]

        def encode_row(ex, mr_key: str):
            mr = ex[mr_key]
            ref = ex["reference"]
            prompt, target = _prompt_and_target_from_mr_ref(mr, ref)

            pr_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            tg_text = target + (self.tokenizer.eos_token or "")
            tg_ids = self.tokenizer(tg_text, add_special_tokens=False)["input_ids"]

            ids = (pr_ids + tg_ids)
            if len(ids) > self.max_length:
                cut = len(pr_ids + tg_ids) - self.max_length
                ids = ids[-self.max_length:]
                pr_len = max(0, len(pr_ids) - cut)
            else:
                pr_len = len(pr_ids)

            labels = ([-100] * pr_len) + ids[pr_len:]
            return {"input_ids": ids, "labels": labels}

        # Use fn_kwargs to pass each split's mr_key
        tr_tok = tr.map(encode_row, fn_kwargs={"mr_key": mr_tr},
                        remove_columns=[c for c in tr.column_names if c not in [mr_tr, "reference"]])
        va_tok = (va.map(encode_row, fn_kwargs={"mr_key": mr_va},
                         remove_columns=[c for c in va.column_names if c not in [mr_va, "reference"]])
                  if va is not None else None)
        te_tok = (te.map(encode_row, fn_kwargs={"mr_key": mr_te},
                         remove_columns=[c for c in te.column_names if c not in [mr_te, "reference"]])
                  if te is not None else None)

        self.train_ds = tr_tok
        self.val_ds   = va_tok if va_tok is not None else tr_tok.select(range(min(1484, len(tr_tok))))
        self.test_ds  = te_tok if te_tok is not None else self.val_ds

    def _loader(self, ds, train=False) -> DataLoader:
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=train,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            drop_last=self.drop_last if train else False,
            collate_fn=self.collator,
            worker_init_fn=_seed_worker,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(ds, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)
    
    def on_epoch_start(self, epoch: int):
        pass

