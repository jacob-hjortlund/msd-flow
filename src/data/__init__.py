# src/data/__init__.py

from .split import assign_splits
from .download_tng import download_tng_data
from .pipeline import resolve_dataset

__all__ = [
    "download_tng_data",
    "assign_splits",
    "resolve_dataset",
]
