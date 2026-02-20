# data/__init__.py
"""
Data modules + registry.

Importing this package will eagerly import dataset submodules so their
@register_data(...) decorators run.

Environment toggles:
  - DATA_EAGER_REGISTER=0   -> disable eager importing
  - DATA_STRICT_IMPORT=1    -> raise on ImportError during eager importing
"""

from __future__ import annotations

from .base import BaseDataModule, register_data, get_data


def _eager_register() -> None:
    import os
    import importlib
    import pkgutil

    eager = os.environ.get("DATA_EAGER_REGISTER", "1").lower() not in {"0", "false", "no"}
    if not eager:
        return

    strict = os.environ.get("DATA_STRICT_IMPORT", "0").lower() in {"1", "true", "yes"}

    # Discover immediate submodules in this package and import them.
    # This ensures @register_data decorators run.
    pkg = __name__  # e.g., "data" or "myproj.data"
    skip = {
        "__init__",
        "base",
        "vtab_helpers",
        "vision_common",
    }

    for mod in pkgutil.iter_modules(__path__):
        name = mod.name
        if name.startswith("_") or name in skip:
            continue
        try:
            importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            # Allow optional dependencies / optional datasets.
            if strict:
                raise


_eager_register()


# Optional: expose classes if imports succeed (kept safe so importing `data` doesn't crash)
try:
    from .glue import GLUEDataModule
except Exception:
    GLUEDataModule = None

try:
    from .officehome import OfficeHomeDataModule
except Exception:
    OfficeHomeDataModule = None

try:
    from .terraincognita import TerraIncognitaDataModule
except Exception:
    TerraIncognitaDataModule = None

try:
    from .cifar100 import CIFAR100DataModule
except Exception:
    CIFAR100DataModule = None

try:
    from .cifar10 import CIFAR10DataModule
except Exception:
    CIFAR10DataModule = None

try:
    from .imagenet1k import ImageNet1KDataModule
except Exception:
    ImageNet1KDataModule = None

try:
    from .imagenet21k import ImageNet21KDataModule
except Exception:
    ImageNet21KDataModule = None

try:
    from .caltech101 import Caltech101DataModule
except Exception:
    Caltech101DataModule = None

try:
    from .flowers102 import Flowers102DataModule
except Exception:
    Flowers102DataModule = None

try:
    from .pet import OxfordPetDataModule
except Exception:
    OxfordPetDataModule = None

try:
    from .svhn import SVHNDataModule
except Exception:
    SVHNDataModule = None

# If you later add data/e2e.py back, this will start exporting automatically.
try:
    from .e2e import E2EDataModule  # type: ignore
except Exception:
    E2EDataModule = None


__all__ = [
    "BaseDataModule",
    "register_data",
    "get_data",
    "GLUEDataModule",
    "OfficeHomeDataModule",
    "TerraIncognitaDataModule",
    "CIFAR100DataModule",
    "CIFAR10DataModule",
    "ImageNet1KDataModule",
    "ImageNet21KDataModule",
    "Caltech101DataModule",
    "Flowers102DataModule",
    "OxfordPetDataModule",
    "SVHNDataModule",
    "E2EDataModule",
]
