# src/data/__init__.py

from .split import assign_splits
from .download_tng import download_tng_data

__all__ = [
    "download_tng_data",
    "assign_splits",
]
