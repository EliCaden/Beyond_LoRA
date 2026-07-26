# data/mcq_nlu.py
from __future__ import annotations

from typing import Dict, Any
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from .base import BaseDataModule, register_data

class MCQDataModule(BaseDataModule):
    """
    MCQ DataModule for LoRA fine-tuning (decoder-only models).
    Converts each example into:
        prompt: str
        answer: str
    Masks prompt tokens during training.
    """
    HF_NAME: str | None = None
    HF_SUBSET: str | None = None
    SPLIT_TRAIN: str = "train"
    SPLIT_VAL: str = "validation"
    SPLIT_TEST: str = "test"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
        max_length: int = 512,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        # Collator handles variable-length labels for batching
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self):
        ds = load_dataset(self.HF_NAME, self.HF_SUBSET) if self.HF_SUBSET else load_dataset(self.HF_NAME)

        def tokenize_fn(example):
            prompt, answer = self.normalize(example)
            if prompt is None:
                return {}

            # IMPORTANT: tokenize separately to avoid boundary artifacts
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            # include a leading space so " Yes"/" No" are scored correctly
            answer_ids = self.tokenizer(
                " " + answer,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids

            # keep the end (preserves answer tokens) if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length:]
                labels = labels[-self.max_length:]

            return {"input_ids": input_ids, "labels": labels}

        # Map datasets
        self.train_ds = ds[self.SPLIT_TRAIN].map(tokenize_fn, remove_columns=ds[self.SPLIT_TRAIN].column_names)
        self.train_ds = self.train_ds.filter(lambda x: len(x["input_ids"]) > 0) # filter out None training ex.

        self.val_ds = ds[self.SPLIT_VAL].map(tokenize_fn, remove_columns=ds[self.SPLIT_VAL].column_names)
        if self.SPLIT_TEST in ds:
          self.test_ds = ds[self.SPLIT_TEST].map(tokenize_fn, remove_columns=ds[self.SPLIT_TEST].column_names)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.test_ds is not None:
          return DataLoader(
              self.test_ds,
              batch_size=self.batch_size,
              shuffle=False,
              collate_fn=self.collator,
              num_workers=self.num_workers,
          )
        else:
          print(f"[WARNING] No test split found for {self.__class__.__name__}, falling back to validation set.")
          return self.val_dataloader()

    def normalize(self, example: Dict[str, Any]) -> tuple[str, str]:
        """Return prompt, answer strings. Subclasses override this."""
        raise NotImplementedError


# ---------------------------
# BoolQ
# ---------------------------
@register_data("boolq")
class BoolQDataModule(MCQDataModule):
    HF_NAME = "google/boolq"
    SPLIT_TRAIN = "train"
    SPLIT_VAL = "validation"

    def normalize(self, ex):
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        answer = "Yes" if ex["answer"] else "No"
        return prompt, answer


# ---------------------------
# PIQA
# ---------------------------
@register_data("piqa")
class PIQADataModule(MCQDataModule):
    HF_NAME = "baber/piqa" # NOTE: original ybisk/piqa not supported anymore, this one seems to be a fixed re-upload by someone else

    def normalize(self, ex):
        prompt = f"Goal: {ex['goal']}\nChoices:\nA) {ex['sol1']}\nB) {ex['sol2']}\nAnswer:"
        answer = "A" if ex["label"] == 0 else "B"
        return prompt, answer


# ---------------------------
# SIQA
# ---------------------------
# NOTE the dataset wasn't updated on HF side "Dataset scripts are no longer supported, but found social_i_qa.py"
# @register_data("siqa")
# class SIQADataModule(MCQDataModule):
#     HF_NAME = "allenai/social_i_qa"

#     def normalize(self, ex):
#         choices = ex["choices"]["text"]
#         letters = ["A", "B", "C"]
#         prompt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nChoices:\n"
#         for l, c in zip(letters, choices):
#             prompt += f"{l}) {c}\n"
#         prompt += "Answer:"
#         answer = letters[ex["label"]]
#         return prompt, answer


# ---------------------------
# ARC Challenge
# ---------------------------
@register_data("arc_c")
class ARCChallengeDataModule(MCQDataModule):
    HF_NAME = "allenai/ai2_arc"
    HF_SUBSET = "ARC-Challenge"

    def normalize(self, ex):
        choices = ex["choices"]["text"]
        letters = ["A", "B", "C", "D"][:len(choices)]
        prompt = f"Question: {ex['question']}\nChoices:\n"
        for l, c in zip(letters, choices):
            prompt += f"{l}) {c}\n"
        prompt += "Answer:"
        answer = ex["answerKey"]
        return prompt, answer


# ---------------------------
# ARC Easy
# ---------------------------
@register_data("arc_e")
class ARCEasyDataModule(MCQDataModule):
    HF_NAME = "allenai/ai2_arc"
    HF_SUBSET = "ARC-Easy"

    def normalize(self, ex):
        choices = ex["choices"]["text"]
        letters = ["A", "B", "C", "D"][:len(choices)]
        prompt = f"Question: {ex['question']}\nChoices:\n"
        for l, c in zip(letters, choices):
            prompt += f"{l}) {c}\n"
        prompt += "Answer:"
        answer = ex["answerKey"]
        return prompt, answer


# ---------------------------
# OpenBookQA
# ---------------------------
@register_data("obqa")
class OBQADataModule(MCQDataModule):
    HF_NAME = "allenai/openbookqa"
    HF_SUBSET = "main"

    def normalize(self, ex):
        choices = ex["choices"]["text"]
        letters = ["A", "B", "C", "D"][:len(choices)]
        prompt = f"Question: {ex['question_stem']}\nChoices:\n"
        for l, c in zip(letters, choices):
            prompt += f"{l}) {c}\n"
        prompt += "Answer:"
        answer = ex["answerKey"]
        return prompt, answer


# ---------------------------
# HellaSWAG
# ---------------------------
@register_data("hellaswag")
class HellaSwagDataModule(MCQDataModule):
    HF_NAME = "rowan/hellaswag"

    def normalize(self, ex):
      choices = ex["endings"]
      letters = ["A", "B", "C", "D"]
      prompt = f"{ex['ctx']}\nChoices:\n"
      for l, c in zip(letters, choices):
          prompt += f"{l}) {c}\n"
      prompt += "Answer:"
      if not ex["label"].strip():
          return None, None
      answer = letters[int(ex["label"])]
      return prompt, answer


# ---------------------------
# WinoGrande
# ---------------------------
@register_data("winogrande")
class WinoGrandeDataModule(MCQDataModule):
    HF_NAME = "allenai/winogrande"
    HF_SUBSET = "winogrande_xl"

    def normalize(self, ex):
      prompt = (
          f"{ex['sentence'].replace('_', '____')}\n"
          f"A) {ex['option1']}\n"
          f"B) {ex['option2']}\n"
          f"Answer:"
      )
      answer = "A" if ex["answer"] == "1" else "B"
      return prompt, answer