"""Train an MSD flow model from a Hydra configuration."""

import logging

import hydra
import jax.random as jr

from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, open_dict

from msdflow.tracking import setup_task
from msdflow.data.loader import build_dataloader
from msdflow.data.pipeline import resolve_dataset
from msdflow.train.checkpointing import (
    checkpoint_run_dir,
    compute_config_hash,
    discover_latest_checkpoint,
)
from msdflow.utils import register_all_resolvers, seed_everything

register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """Run dataset resolution, model construction, and training.

    Args:
        cfg: Hydra configuration for the training entrypoint.
    """
    hash_exclude = list(cfg.train.resume.hash_exclude)
    if cfg.train.resume.hash is None:
        checkpoint_hash, hash_payload = compute_config_hash(
            cfg,
            exclude_paths=hash_exclude,
        )
    else:
        checkpoint_hash = str(cfg.train.resume.hash)
        _, hash_payload = compute_config_hash(
            cfg,
            exclude_paths=hash_exclude,
        )

    run_checkpoint_dir = checkpoint_run_dir(
        cfg.train.checkpoint_dir,
        checkpoint_hash,
    )
    resume_metadata = None
    if cfg.train.resume.auto:
        resume_metadata = discover_latest_checkpoint(
            run_checkpoint_dir,
            latest_filename=str(cfg.train.resume.latest_filename),
            restart=bool(cfg.train.resume.restart),
        )

    # 0. ClearML setup
    resume_task_id = (
        resume_metadata["clearml_task_id"] if resume_metadata is not None else None
    )
    task = setup_task(cfg.clearml, resume_task_id=resume_task_id)

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
        cfg.train.checkpoint_dir = run_checkpoint_dir
        cfg.train.checkpoint_hash = checkpoint_hash
        cfg.train.hash_payload = hash_payload
        cfg.train.latest_filename = str(cfg.train.resume.latest_filename)
        cfg.train.save_on_sigterm = bool(cfg.train.resume.save_on_sigterm)

    # 3. Seed
    log.info("--- Step 3: Seeding ---")
    seed = cfg.seed
    rng_key = seed_everything(seed)

    # 4. Build dataloaders
    log.info("--- Step 4: Dataloader Initialization ---")
    train_loader = build_dataloader(cfg.data.dataloader.train, seed=seed)
    val_loader = build_dataloader(cfg.data.dataloader.val, seed=seed + 1)
    test_loader = build_dataloader(cfg.data.dataloader.test, seed=seed + 2)

    log.info(f"Initialized train loader with {len(train_loader)} batches.")

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
        resume_checkpoint_path=(
            resume_metadata["payload_path"] if resume_metadata is not None else None
        ),
        resume_metadata=resume_metadata if resume_metadata is not None else None,
        source_checkpoint_path=(
            resume_metadata["payload_path"] if resume_metadata is not None else None
        ),
    )

    # 7. Test model
    # TODO: implement test loop and call here


if __name__ == "__main__":
    main()
