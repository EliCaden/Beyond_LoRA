# data/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, List

from torch.utils.data import DataLoader

_DATASETS: Dict[str, Type["BaseDataModule"]] = {}


def _norm_key(name: str) -> str:
    return str(name).strip().lower()


def register_data(name: str):
    """
    Decorator to register a DataModule under a string key.

    Important: keys are normalized to lowercase, and duplicate registrations
    raise to avoid silent overrides (which are painful in sweep orchestration).
    """
    key = _norm_key(name)

    def decorator(cls: Type[BaseDataModule]):
        prev = _DATASETS.get(key)
        if prev is not None and prev is not cls:
            raise KeyError(
                f"Dataset key {key!r} already registered by {prev.__module__}.{prev.__name__}; "
                f"refusing to overwrite with {cls.__module__}.{cls.__name__}."
            )
        _DATASETS[key] = cls
        return cls

    return decorator


def list_data() -> List[str]:
    """Return sorted list of registered dataset keys."""
    return sorted(_DATASETS.keys())


def get_data(name: str, **kwargs: Any):
    """
    Factory method: return an instance of the registered DataModule.
    """
    key = _norm_key(name)
    if key not in _DATASETS:
        avail = list_data()
        raise ValueError(
            f"Unknown dataset module: {name!r} (normalized: {key!r}). "
            f"Available: {avail}"
        )
    return _DATASETS[key](**kwargs)


class BaseDataModule(ABC):
    """
    Abstract base for all data modules.
    Implementors should:
     - load and preprocess data in setup()
     - provide PyTorch DataLoaders
    """

    def __init__(self, **cfg: Any):
        self.cfg = cfg

    def on_epoch_start(self, epoch: int) -> None:
        """Optional hook called by the trainer once per epoch."""
        return

    @abstractmethod
    def setup(self) -> None:
        """Load dataset, set self.train_ds, self.val_ds, self.test_ds."""
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError