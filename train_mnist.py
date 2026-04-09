"""Training entry point for MNIST flow matching.

Mirrors ``train_model.py`` but omits TNG-specific infrastructure
(dataset download, ClearML dataset management, runtime config injection).
Run from the ``msd-flow/`` directory:

    python train_mnist.py
    python train_mnist.py train.num_epochs=50 data.dataloader.batch_size=256
"""

import logging

import hydra
import jax.random as jr

from hydra.utils import instantiate, call
from omegaconf import DictConfig

from msdflow.utils import register_all_resolvers


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config_mnist")
def main(cfg: DictConfig):

    # 1. Build dataloaders
    log.info("--- Step 1: Dataloader Initialization ---")
    train_loader = instantiate(cfg.data.dataloader.train)
    val_loader = instantiate(cfg.data.dataloader.val)

    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Seed
    log.info("--- Step 2: Seeding ---")
    rng_key = jr.PRNGKey(cfg.seed)

    # 3. Build model
    log.info("--- Step 3: Model Initialization ---")
    model_key, rng_key = jr.split(rng_key)
    model = instantiate(cfg.model)(key=model_key)

    # 4. Train
    log.info("--- Step 4: Training ---")
    train_key, rng_key = jr.split(rng_key)
    call(cfg.train)(
        key=train_key,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        clearml_task=None,
    )


if __name__ == "__main__":
    main()
