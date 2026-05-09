import torch
import random

import numpy as np

from collections.abc import Iterable
from torchvision.transforms import Compose
from torch.utils.data import get_worker_info


class WorkerSeededTransform:
    """Mixin for transforms that own a NumPy Generator."""

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def set_rng_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)

    def _get_rng(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = np.random.default_rng()
        return self.rng


def _collect_seedable_transforms(transform) -> list[WorkerSeededTransform]:
    seedable = []

    def visit(obj):
        if obj is None:
            return

        if hasattr(obj, "set_rng_seed"):
            seedable.append(obj)

        # Only recurse through actual Compose containers.
        if isinstance(obj, Compose):
            for child in obj.transforms:
                visit(child)

    visit(transform)
    return seedable


def seed_transform_tree(transform, seed: int) -> None:
    """Seed every stochastic transform in a composed transform tree."""
    seedable = _collect_seedable_transforms(transform)
    if not seedable:
        return

    seq = np.random.SeedSequence(int(seed))
    child_seqs = seq.spawn(len(seedable))

    for transform_obj, child_seq in zip(seedable, child_seqs, strict=True):
        child_seed = int(child_seq.generate_state(1, dtype=np.uint32)[0])
        transform_obj.set_rng_seed(child_seed)


def seed_worker(worker_id: int) -> None:
    """
    PyTorch worker init function.

    - Seeds Python random + NumPy from torch's worker seed.
    - Re-seeds stochastic transforms attached to the dataset copy
      owned by this worker.
    """
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    info = get_worker_info()
    if info is None:
        return

    dataset = info.dataset

    image_transform = getattr(dataset, "image_transform", None)
    if image_transform is not None:
        seed_transform_tree(image_transform, worker_seed)

    metadata_transform = getattr(dataset, "metadata_transform", None)
    if metadata_transform is not None:
        seed_transform_tree(metadata_transform, worker_seed + 1)
