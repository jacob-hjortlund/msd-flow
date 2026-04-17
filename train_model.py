import logging

import hydra
import jax.random as jr

from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, open_dict

from msdflow.tracking import setup_task
from msdflow.utils import register_all_resolvers
from msdflow.data.pipeline import resolve_dataset


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # 0. ClearML setup
    task = setup_task(cfg.clearml)

    # 1. Dataset resolution — download / re-split / reuse as needed
    log.info("--- Step 1: Dataset Resolution ---")
    dataset_cfg = cfg.data.dataset
    dataset_path = resolve_dataset(
        task=task,
        dataset_name=dataset_cfg.dataset_name,
        data_dir=dataset_cfg.data_dir,
        seed=dataset_cfg.seed,
        ratios=OmegaConf.to_container(dataset_cfg.ratios, resolve=True),
        download_cfg=cfg.data.download,
        use_dataset=cfg.clearml.use_dataset,
    )

    # 2. Inject resolved path into dataloader config
    log.info("--- Step 2: Config Injection ---")
    with open_dict(cfg):
        cfg.data.dataloader.cache_dir = cfg.data.dataloader.data_dir
        cfg.data.dataloader.data_dir = dataset_path

    # 3. Build dataloaders
    log.info("--- Step 3: Dataloader Initialization ---")
    train_loader = instantiate(cfg.data.dataloader.train)
    val_loader = instantiate(cfg.data.dataloader.val)
    test_loader = instantiate(cfg.data.dataloader.test)

    log.info(f"Initialized train loader with {len(train_loader)} batches.")

    # 4. Seed
    log.info("--- Step 4: Seeding ---")
    seed = cfg.seed
    rng_key = jr.PRNGKey(seed)

    # 5. Build model
    log.info("--- Step 5: Model Initialization ---")
    model_key, rng_key = jr.split(rng_key)
    model = instantiate(cfg.model)(key=model_key)

    # 6. Train model
    log.info("--- Step 6: Model Training ---")
    train_key, rng_key = jr.split(rng_key)
    trained_model = call(cfg.train)(
        key=train_key,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        clearml_task=task,
    )

    # 7. Test model
    # TODO: implement test loop and call here


if __name__ == "__main__":
    main()
