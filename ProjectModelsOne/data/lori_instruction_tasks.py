# data/lori_instruction_tasks.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from .base import BaseDataModule, register_data


_TASK_ALIASES = {
    "boolq": "boolq",
    "piqa": "piqa",
    "social_i_qa": "social_i_qa",
    "siqa": "social_i_qa",
    "arc-challenge": "arc-challenge",
    "arc_c": "arc-challenge",
    "arc-easy": "arc-easy",
    "arc_e": "arc-easy",
    "openbookqa": "openbookqa",
    "obqa": "openbookqa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
}


def _canonical_task_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _TASK_ALIASES:
        raise ValueError(f"Unknown LoRI task alias: {name!r}. Known aliases: {sorted(_TASK_ALIASES)}")
    return _TASK_ALIASES[key]


def resolve_lori_task_dir(task_name: str, data_root: str = "datasets/lori") -> Path:
    """
    Resolve a LoRI task alias to a local dataset folder under data_root.

    Expected local layout:
        datasets/lori/
            boolq/train.json
            boolq/test.json
            piqa/train.json
            piqa/test.json
            social_i_qa/train.json
            social_i_qa/test.json
            arc-challenge/train.json
            arc-challenge/test.json
            arc-easy/train.json
            arc-easy/test.json
            openbookqa/train.json
            openbookqa/test.json
            hellaswag/train.json
            hellaswag/test.json
            winogrande/train.json
            winogrande/test.json
    """
    canonical = _canonical_task_name(task_name)
    task_dir = Path(data_root) / canonical

    train_path = task_dir / "train.json"
    test_path = task_dir / "test.json"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing LoRI train file for task={task_name!r}: {train_path}"
        )
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Missing LoRI test file for task={task_name!r}: {test_path}"
        )

    return task_dir


def build_lori_prompt(instruction: str, input_text: str | None = None) -> str:
    instruction = str(instruction)
    input_text = None if input_text is None else str(input_text)

    if input_text is not None and input_text.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


class LoRITaskJSONDataModule(BaseDataModule):
    """
    Local-JSON SFT loader for LoRI-style instruction tasks.

    Reads local files from:
        <task_root>/<canonical_task_name>/train.json
        <task_root>/<canonical_task_name>/test.json

    Example:
        datasets/lori/openbookqa/train.json
        datasets/lori/openbookqa/test.json

    Each JSON file should contain a list of rows with:
        - instruction: str
        - output: str
        - input: optional str
    """

    TASK_NAME: str | None = None

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
        max_length: int = 512,
        max_prompt_length: int = 256,
        num_workers: int = 0,
        task_root: str = "datasets/lori",
        val_fraction: float = 0.1,
        split_seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if self.TASK_NAME is None:
            raise ValueError(f"{self.__class__.__name__} must define TASK_NAME")

        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_prompt_length = int(max_prompt_length)
        self.num_workers = int(num_workers)
        self.task_root = Path(task_root)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)

        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.task_dir = resolve_lori_task_dir(self.task_name, data_root=str(self.task_root))

    @property
    def task_name(self) -> str:
        return _canonical_task_name(self.TASK_NAME)

    def _read_json_rows(self, split: str) -> List[Dict[str, Any]]:
        path = self.task_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"LoRI task file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(rows).__name__}")

        return [dict(r) for r in rows]

    def _split_train_val(self, rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = list(rows)
        if len(rows) <= 1:
            return rows, rows

        g = random.Random(self.split_seed)
        g.shuffle(rows)

        n_val = int(round(len(rows) * self.val_fraction))
        if self.val_fraction > 0 and n_val <= 0:
            n_val = 1
        n_val = min(max(1, n_val), len(rows) - 1)

        val_rows = rows[:n_val]
        train_rows = rows[n_val:]
        return train_rows, val_rows

    def _tokenize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_lori_prompt(row["instruction"], row.get("input"))
        target = str(row["output"])

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[: self.max_prompt_length]

        answer_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != eos_id):
            answer_ids = answer_ids + [eos_id]

        max_answer_len = self.max_length - len(prompt_ids)
        if max_answer_len <= 0:
            return {}

        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _rows_to_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        ds = Dataset.from_list(rows)
        ds = ds.map(self._tokenize_row, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x.get("input_ids", [])) > 0)
        return ds

    def setup(self) -> None:
        train_rows = self._read_json_rows("train")
        test_rows = self._read_json_rows("test")
        train_rows, val_rows = self._split_train_val(train_rows)

        self.train_ds = self._rows_to_dataset(train_rows)
        self.val_ds = self._rows_to_dataset(val_rows)
        self.test_ds = self._rows_to_dataset(test_rows)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )


@register_data("boolq_lori")
class BoolQLoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "boolq"


@register_data("piqa_lori")
class PIQALoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "piqa"


@register_data("siqa_lori")
class SIQALoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "social_i_qa"


@register_data("arc_c_lori")
class ARCChallengeLoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "arc-challenge"


@register_data("arc_e_lori")
class ARCEasyLoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "arc-easy"


@register_data("obqa_lori")
class OpenBookQALoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "openbookqa"


@register_data("hellaswag_lori")
class HellaSwagLoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "hellaswag"


@register_data("winogrande_lori")
class WinoGrandeLoRIDataModule(LoRITaskJSONDataModule):
    TASK_NAME = "winogrande"