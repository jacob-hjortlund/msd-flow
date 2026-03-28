import os
import hydra
import logging

import numpy as np
import pandas as pd
import jax.random as jr
import src.data as data

from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from src.utils import register_all_resolvers

register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # 0. TODO: clearML setup

    # 1. Variables and logging
    seed = cfg.seed
    rng_key = jr.PRNGKey(seed)

    log.info("Starting full training pipeline...")

    # 2. Download data
    log.info("--- Step 2: Data Download ---")
    download_cfg = cfg.data.download
    call(download_cfg)

    # 3. Assign splits
    log.info("--- Step 3: Split Assignment ---")
    split_cfg = cfg.data.split
    data.assign_splits(
        processed_dir=split_cfg.processed_dir,
        seed=split_cfg.seed,
        ratios=dict(split_cfg.ratios),
    )

    # 4. Build dataloaders
    log.info("--- Step 4: Dataloader Initialization ---")

    train_loader = instantiate(cfg.data.dataloader.train)
    val_loader = instantiate(cfg.data.dataloader.val)
    test_loader = instantiate(cfg.data.dataloader.test)

    log.info(f"Initialized train loader with {len(train_loader)} batches.")

    # 5. Build model
    log.info("--- Step 5: Model Initialization ---")
    model_key, rng_key = jr.split(rng_key)
    model = instantiate(cfg.model)(key=model_key)

    # Train model
    log.info("--- Step 6: Model Training ---")
    train_key, rng_key = jr.split(rng_key)
    trained_model = call(cfg.train)(
        key=train_key,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
    )


if __name__ == "__main__":
    main()
