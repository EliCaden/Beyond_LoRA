# data/glue.py
from __future__ import annotations

import os
import random
import inspect
from collections import Counter
from typing import Optional, Literal

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader, Sampler
from transformers import DataCollatorWithPadding

from .base import register_data, BaseDataModule

try:
    from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler
    _HAS_LGS = True
except Exception:
    _HAS_LGS = False

# Known GLUE field layouts
_SINGLE_SENTENCE = {"cola", "sst2"}
_PAIR_FIELDS = {
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"),
    "wnli": ("sentence1", "sentence2"),
}
_REGRESSION = {"stsb"}  # label is float


def _seed_worker(worker_id: int) -> None:
    # Derive per-worker seed from torch's worker seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---- Deterministic length samplers (fallbacks) ----

class _EpochSeededLengthSampler(Sampler[int]):
    """
    Non-DDP deterministic length bucketing sampler that reshuffles *per epoch*.

    - sorts by length
    - groups into batches
    - shuffles batch order + within-batch order using seed+epoch
    """

    def __init__(self, *, lengths, batch_size: int, seed: int, drop_last: bool):
        self.lengths = [int(l) for l in lengths]
        self.N = len(self.lengths)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_order(self) -> list[int]:
        idx = list(range(self.N))
        idx.sort(key=lambda i: self.lengths[i])

        B = self.batch_size
        chunks = [idx[i : i + B] for i in range(0, len(idx), B)]
        if self.drop_last and chunks and len(chunks[-1]) < B:
            chunks = chunks[:-1]

        r = random.Random(int(self.seed + self.epoch))
        r.shuffle(chunks)
        for c in chunks:
            r.shuffle(c)

        return [j for c in chunks for j in c]

    def __iter__(self):
        return iter(self._build_order())

    def __len__(self):
        if self.drop_last:
            return (self.N // self.batch_size) * self.batch_size
        return self.N


class _EpochSeededDistributedLengthSampler(Sampler[int]):
    """
    Deterministic DDP variant (reshuffles per epoch):

    - form global mega-batches of size batch_size*world_size by length
    - shuffle mega-batches deterministically (seed+epoch)
    - split each mega-batch into world_size chunks of size batch_size
    - each rank yields its chunk indices
    """

    def __init__(self, *, lengths, batch_size: int, world_size: int, rank: int, seed: int, drop_last: bool):
        self.lengths = [int(l) for l in lengths]
        self.N = len(self.lengths)
        self.B = int(batch_size)
        self.W = int(world_size)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_order_rank(self) -> list[int]:
        idx = list(range(len(self.lengths)))
        idx.sort(key=lambda i: self.lengths[i])

        mega = self.B * self.W
        megas = [idx[i : i + mega] for i in range(0, len(idx), mega)]
        if self.drop_last and megas and len(megas[-1]) < mega:
            megas = megas[:-1]

        r = random.Random(int(self.seed + self.epoch))
        r.shuffle(megas)
        for m in megas:
            r.shuffle(m)

        my_indices: list[int] = []
        for m in megas:
            if len(m) < mega and not self.drop_last:
                pad = mega - len(m)
                m = m + m[:pad]

            start = self.rank * self.B
            stop = start + self.B
            my_indices.extend(m[start:stop])
        return my_indices

    def __iter__(self):
        return iter(self._build_order_rank())

    def __len__(self):
        # number of samples *this rank* yields
        mega = self.B * self.W
        full = (self.N // mega) * self.B
        if self.drop_last:
            return full
        # padded last mega-batch contributes B samples as well
        rem = self.N % mega
        return full + (self.B if rem else 0)


@register_data("glue")
class GLUEDataModule(BaseDataModule):
    """
    Generic GLUE DataModule.

    Speed-friendly defaults:
     - dynamic padding (DataCollatorWithPadding)
     - pad_to_multiple_of=8 (tensor core friendly)
     - pinned memory, workers, persistent workers, prefetch
     - optional length bucketing (deterministic fallbacks)
     - optional multi-proc tokenization via tokenize_num_proc
    """

    def __init__(
        self,
        task_name: str,
        tokenizer,
        batch_size: int,
        max_length: int,
        seed: int = 42,
        num_workers: int | None = None,
        use_length_bucketing: bool = True,
        pad_to_multiple_of: int | None = 8,
        drop_last: bool = False,
        fallback_test_from_val: bool = False,
        val_test_split: float = 0.5,
        stratify_val_split: bool = True,
        force_val_as_test: bool = False,
        label_presence_threshold: float = 0.8,
        tokenize_num_proc: int | None = None,
        mnli_eval: Literal["matched", "mismatched", "both"] = "matched",
    ):
        super().__init__(task_name=task_name, batch_size=batch_size, max_length=max_length, seed=seed, mnli_eval=mnli_eval)
        self.task_name = str(task_name).lower().strip()
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.seed = int(seed)

        # dataloader perf knobs
        self.num_workers = (
            int(num_workers)
            if num_workers is not None
            else max(1, min(4, (os.cpu_count() or 4) // 2))
        )
        self.use_length_bucketing = bool(use_length_bucketing)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.drop_last = bool(drop_last)

        self.fallback_test_from_val = bool(fallback_test_from_val)
        self.val_test_split = float(val_test_split)
        self.stratify_val_split = bool(stratify_val_split)
        self.force_val_as_test = bool(force_val_as_test)
        self.label_presence_threshold = float(label_presence_threshold)
        self.tokenize_num_proc = int(tokenize_num_proc) if tokenize_num_proc is not None else None

        self.mnli_eval: Literal["matched", "mismatched", "both"] = mnli_eval

        # set at setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._train_lengths = None

        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # One generator used for (a) sampler creation seeds and (b) DataLoader generator.
        self._rng = torch.Generator()
        self._rng.manual_seed(self.seed)
        self._sampler_epoch = 0

        # Keep references so we can set_epoch even if the loader is reused.
        self._train_sampler = None
        self._train_loader = None

    def setup(self):
        ds = load_dataset("glue", self.task_name)

        # --- resolve input fields per task ---
        if self.task_name in _SINGLE_SENTENCE:
            mode = "single"
            fields = ("sentence",)
        elif self.task_name in _PAIR_FIELDS:
            mode = "pair"
            fields = _PAIR_FIELDS[self.task_name]
        else:
            cand_pairs = [
                ("sentence1", "sentence2"),
                ("premise", "hypothesis"),
                ("question1", "question2"),
                ("question", "sentence"),
            ]
            ok = None
            for a, b in cand_pairs:
                if a in ds["train"].column_names and b in ds["train"].column_names:
                    ok = (a, b)
                    break
            if ok:
                mode, fields = "pair", ok
            elif "sentence" in ds["train"].column_names:
                mode, fields = "single", ("sentence",)
            else:
                raise ValueError(f"Unrecognized GLUE columns for task={self.task_name}: {ds['train'].column_names}")

        def tokenize_fn(examples):
            if mode == "single":
                (s,) = fields
                return self.tokenizer(examples[s], truncation=True, max_length=self.max_length)
            a, b = fields
            return self.tokenizer(examples[a], examples[b], truncation=True, max_length=self.max_length)

        remove_cols = [c for c in ds["train"].column_names if c not in ["label"]]
        tokenized = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols,
            num_proc=self.tokenize_num_proc,
            load_from_cache_file=True,
        )

        # rename "label" -> "labels" wherever it exists
        def _maybe_rename(split):
            return split.rename_column("label", "labels") if "label" in split.column_names else split

        tokenized = DatasetDict({k: _maybe_rename(v) for k, v in tokenized.items()})

        # --- handle MNLI splits ---
        if self.task_name == "mnli":
            val_mat = tokenized.get("validation_matched")
            val_mis = tokenized.get("validation_mismatched")
            tst_mat = tokenized.get("test_matched")
            tst_mis = tokenized.get("test_mismatched")

            choice = self.mnli_eval
            if choice == "both":
                self.val_ds = concatenate_datasets([val_mat, val_mis])
                if (tst_mat is not None) and (tst_mis is not None):
                    self.test_ds = concatenate_datasets([tst_mat, tst_mis])
                else:
                    self.test_ds = tst_mat or tst_mis
            elif choice == "mismatched":
                self.val_ds = val_mis
                self.test_ds = tst_mis
            else:
                self.val_ds = val_mat
                self.test_ds = tst_mat
        else:
            self.val_ds = tokenized.get("validation")
            self.test_ds = tokenized.get("test")

        self.train_ds = tokenized["train"]

        # optional: synthesize labeled test from val if official test lacks labels
        self._maybe_make_test_from_val()

        # Precompute per-example lengths for bucketing
        if self.use_length_bucketing:
            try:
                self._train_lengths = [len(ids) for ids in self.train_ds["input_ids"]]
            except Exception:
                self._train_lengths = None

        # Reset cached loader if setup reruns.
        self._train_loader = None
        self._train_sampler = None

    # ---- helpers ----

    def _labels_sample(self, dset, sample_size=2048):
        n = min(sample_size, len(dset)) if dset is not None else 0
        return dset[:n]["labels"] if n > 0 and "labels" in dset.column_names else []

    def _collect_allowed_labels_cls(self):
        allowed = set()
        for split in (self.train_ds, self.val_ds):
            if split is None:
                continue
            try:
                vals = self._labels_sample(split)
                for v in vals:
                    if v is None:
                        continue
                    iv = int(v)
                    if iv >= 0:
                        allowed.add(iv)
            except Exception:
                pass
        return allowed

    def _has_labels(self, dset):
        try:
            if dset is None or "labels" not in dset.features:
                return False

            test_vals = self._labels_sample(dset)
            if not test_vals:
                return False

            if self.task_name in _REGRESSION:
                import math

                pool = []
                for split in (self.train_ds, self.val_ds):
                    if split is None:
                        continue
                    pool.extend(self._labels_sample(split))
                pool = [float(v) for v in pool if v is not None and not math.isnan(float(v))]
                if not pool:
                    return False
                lo, hi = min(pool), max(pool)
                eps = 1e-6

                def in_range(v):
                    try:
                        fv = float(v)
                        return (fv == fv) and (lo - eps <= fv <= hi + eps)
                    except Exception:
                        return False

                total = 0
                good = 0
                for v in test_vals:
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv != fv:
                        continue
                    total += 1
                    good += int(in_range(fv))
                frac = (good / total) if total > 0 else 0.0
                return frac >= self.label_presence_threshold

            ints = []
            for v in test_vals:
                try:
                    ints.append(int(v))
                except Exception:
                    continue
            if not ints:
                return False

            counts = Counter(ints)
            total = sum(counts.values())
            dom_label, dom_count = counts.most_common(1)[0]
            dom_frac = dom_count / max(1, total)

            # If almost all labels are the same, treat as suspicious unless that label is
            # the only label ever seen in train/val (rare).
            if dom_frac >= self.label_presence_threshold:
                allowed = self._collect_allowed_labels_cls()
                if len(allowed) == 1 and dom_label in allowed:
                    return True
                return False

            return True
        except Exception:
            return False

    def _maybe_make_test_from_val(self):
        if not self.fallback_test_from_val and not self.force_val_as_test:
            return

        needs = self.force_val_as_test or (self.test_ds is None) or (not self._has_labels(self.test_ds))
        if not needs:
            return
        if (self.val_ds is None) or (not self._has_labels(self.val_ds)):
            return

        stratify = None
        if self.task_name not in _REGRESSION and self.stratify_val_split:
            try:
                ys = list(self.val_ds["labels"])
                uniq = {int(y) for y in ys}
                if 1 < len(uniq) <= 10:
                    stratify = "labels"
            except Exception:
                stratify = None

        split = self.val_ds.train_test_split(
            test_size=self.val_test_split,
            seed=self.seed,
            stratify_by_column=stratify,
        )
        self.val_ds = split["train"]
        self.test_ds = split["test"]

    def _maybe_length_sampler(self, lengths):
        if not self.use_length_bucketing or lengths is None:
            return None, True

        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

        if _HAS_LGS:
            # Try HF samplers first; if they exist and support set_epoch, we'll use that.
            try:
                if ddp:
                    world_size = torch.distributed.get_world_size()
                    rank = torch.distributed.get_rank()
                    sig = inspect.signature(DistributedLengthGroupedSampler.__init__)
                    if "generator" in sig.parameters:
                        sampler = DistributedLengthGroupedSampler(
                            batch_size=self.batch_size,
                            lengths=lengths,
                            world_size=world_size,
                            rank=rank,
                            drop_last=self.drop_last,
                            generator=self._rng,
                        )
                    elif "seed" in sig.parameters:
                        sampler = DistributedLengthGroupedSampler(
                            batch_size=self.batch_size,
                            lengths=lengths,
                            world_size=world_size,
                            rank=rank,
                            drop_last=self.drop_last,
                            seed=int(self.seed + self._sampler_epoch),
                        )
                    else:
                        sampler = None
                else:
                    sig = inspect.signature(LengthGroupedSampler.__init__)
                    kwargs = dict(batch_size=self.batch_size, lengths=lengths)
                    if "drop_last" in sig.parameters:
                        kwargs["drop_last"] = self.drop_last
                    if "generator" in sig.parameters:
                        kwargs["generator"] = self._rng
                    elif "seed" in sig.parameters:
                        kwargs["seed"] = int(self.seed + self._sampler_epoch)
                    sampler = LengthGroupedSampler(**kwargs)

                if sampler is not None:
                    if hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(int(self._sampler_epoch))
                    return sampler, False
            except Exception:
                pass

        # fallback: our epoch-aware deterministic samplers
        if ddp:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            sampler = _EpochSeededDistributedLengthSampler(
                lengths=lengths,
                batch_size=self.batch_size,
                world_size=world_size,
                rank=rank,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            sampler.set_epoch(int(self._sampler_epoch))
            return sampler, False

        sampler = _EpochSeededLengthSampler(
            lengths=lengths,
            batch_size=self.batch_size,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        sampler.set_epoch(int(self._sampler_epoch))
        return sampler, False

    def _build_loader(self, dataset, *, lengths=None, train=False):
        sampler, shuffle = (None, True)
        if train:
            sampler, shuffle = self._maybe_length_sampler(lengths)

        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            drop_last=self.drop_last if train else False,
            generator=self._rng,
            worker_init_fn=_seed_worker,
        )

        # Prefer direct pinning to CUDA device when supported
        try:
            sig = inspect.signature(DataLoader.__init__)
            if torch.cuda.is_available() and "pin_memory_device" in sig.parameters:
                kwargs["pin_memory_device"] = "cuda"
        except Exception:
            pass

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2

        return DataLoader(dataset, **kwargs), sampler

    def on_epoch_start(self, epoch: int):
        self._sampler_epoch = int(epoch)
        # Reset generator seed for epoch-consistent worker/base seeding.
        self._rng.manual_seed(int(self.seed + epoch))

        # If we cached a train sampler, update its epoch too.
        if self._train_sampler is not None and hasattr(self._train_sampler, "set_epoch"):
            try:
                self._train_sampler.set_epoch(int(epoch))
            except Exception:
                pass

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is None:
            loader, sampler = self._build_loader(self.train_ds, lengths=self._train_lengths, train=True)
            self._train_loader = loader
            self._train_sampler = sampler
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        loader, _ = self._build_loader(self.val_ds, train=False)
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise ValueError("GLUE task has no test labels")
        loader, _ = self._build_loader(self.test_ds, train=False)
        return loader