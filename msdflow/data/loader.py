import torch

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from msdflow.data.random import seed_transform_tree, seed_worker


def build_dataloader(loader_cfg, *, seed: int) -> DataLoader:
    dataset = instantiate(loader_cfg.dataset)(seed=seed)
    num_workers = int(loader_cfg.num_workers)

    # Ensure deterministic stochastic transforms even when num_workers == 0.
    if num_workers == 0:
        image_transform = getattr(dataset, "image_transform", None)
        if image_transform is not None:
            seed_transform_tree(image_transform, seed)

        metadata_transform = getattr(dataset, "metadata_transform", None)
        if metadata_transform is not None:
            seed_transform_tree(metadata_transform, seed + 1)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs = dict(
        dataset=dataset,
        batch_size=int(loader_cfg.batch_size),
        shuffle=bool(loader_cfg.shuffle),
        drop_last=bool(loader_cfg.drop_last),
        num_workers=num_workers,
        generator=generator,
    )

    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["prefetch_factor"] = int(loader_cfg.prefetch_factor)
        kwargs["persistent_workers"] = bool(loader_cfg.persistent_workers)
        kwargs["multiprocessing_context"] = loader_cfg.multiprocessing_context

    return DataLoader(**kwargs)
