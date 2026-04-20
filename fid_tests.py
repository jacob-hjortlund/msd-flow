import logging

import hydra
import numpy as np
import pandas as pd

from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from msdflow.tracking import setup_task
from msdflow.utils import register_all_resolvers
from msdflow.data.pipeline import resolve_dataset


register_all_resolvers()
log = logging.getLogger(__name__)

LINE_WIDTH = 88
HEAVY = "═" * LINE_WIDTH
LIGHT = "─" * LINE_WIDTH


def configure_terminal_logging(level: int = logging.INFO) -> None:
    """Force a clean, readable terminal logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def banner(title: str) -> str:
    return f"\n{HEAVY}\n{title.upper():^{LINE_WIDTH}}\n{HEAVY}"


def section(title: str) -> str:
    return f"\n{LIGHT}\n{title:<{LINE_WIDTH}}\n{LIGHT}"


def kv(key: str, value, indent: int = 2) -> str:
    return f"{' ' * indent}{key:<24}: {value}"


def format_group_summary(values, groups) -> str:
    lines = [section("Group summary")]
    for name, idx in groups.items():
        vals = values[idx]
        lines.extend(
            [
                f"  [{name.upper()}]",
                kv("count", len(idx), indent=4),
                kv("range", f"[{vals.min():.3g}, {vals.max():.3g}]", indent=4),
                kv("mean", f"{vals.mean():.3g}", indent=4),
                kv("median", f"{np.median(vals):.3g}", indent=4),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def format_pairwise_block(
    g1,
    g2,
    zb_dist,
    in_dist,
    zb_dist_ref=None,
    in_dist_ref=None,
) -> str:
    lines = [f"  {g1.upper()} vs {g2.upper()}"]
    lines.append(kv("Zoobot FID", f"{zb_dist:.4g}", indent=4))
    lines.append(kv("InceptionV3 FID", f"{in_dist:.4g}", indent=4))

    if zb_dist_ref is not None:
        delta_zb = zb_dist - zb_dist_ref
        frac_zb = delta_zb / zb_dist_ref
        lines.append(kv("Zoobot ΔFID", f"{delta_zb:.4g}", indent=4))
        lines.append(kv("Zoobot frac. Δ", f"{100 * frac_zb:.3g}%", indent=4))

    if in_dist_ref is not None:
        delta_in = in_dist - in_dist_ref
        frac_in = delta_in / in_dist_ref
        lines.append(kv("InceptionV3 ΔFID", f"{delta_in:.4g}", indent=4))
        lines.append(kv("InceptionV3 frac. Δ", f"{100 * frac_in:.3g}%", indent=4))

    if (zb_dist_ref is not None) and (in_dist_ref is not None):
        rel_zb = zb_dist / zb_dist_ref
        rel_in = in_dist / in_dist_ref
        lines.append(kv("Zoobot vs ref", f"{rel_zb:.3f}x", indent=4))
        lines.append(kv("InceptionV3 vs ref", f"{rel_in:.3f}x", indent=4))

    return "\n".join(lines)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    configure_terminal_logging()
    log.info(banner("FID metric comparison"))

    # 0. ClearML setup (may fork a background monitor subprocess)
    log.info(section("Step 0 | ClearML setup"))
    task = setup_task(cfg.clearml)

    import jax
    import jax.numpy as jnp

    from msdflow.model.convnext import build_zoobot_nano
    from msdflow.model.inceptionv3 import build_headless_inceptionv3
    from msdflow.train.metrics import (
        FIDAccumulator,
        _frechet_distance,
        morphology_metrics,
    )

    def get_mu_sig(acc, imgs, batch_size=128, desc="Embedding batches"):
        acc.reset()
        total = (len(imgs) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(imgs), batch_size),
            total=total,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        ):
            acc.update(jnp.asarray(imgs[i : i + batch_size]))
        mu, sig, _ = acc.statistics()
        acc.reset()
        return mu, sig

    def split_into_three_groups(values):
        """Split indices into low / medium / high groups with near-equal counts."""
        values = np.asarray(values)
        order = np.argsort(values)
        idx_low, idx_med, idx_high = np.array_split(order, 3)
        return {
            "low": idx_low,
            "medium": idx_med,
            "high": idx_high,
        }

    def compare_metric_groups(
        images,
        metrics,
        metric,
        zb_acc,
        in_acc,
        zb_dist_ref=None,
        in_dist_ref=None,
    ):
        values = np.asarray(metrics[metric].values)
        groups = split_into_three_groups(values)

        lines = [banner(f"FID comparison | {metric}")]
        lines.append(format_group_summary(values, groups))
        lines.append(section("Pairwise FID comparisons"))

        # Precompute Gaussian stats for each group for both encoders
        zb_stats = {}
        in_stats = {}

        for name, idx in groups.items():
            imgs = images[idx]
            zb_stats[name] = get_mu_sig(
                zb_acc,
                imgs,
                desc=f"Zoobot stats ({metric} | {name})",
            )
            in_stats[name] = get_mu_sig(
                in_acc,
                imgs,
                desc=f"Inception stats ({metric} | {name})",
            )

        # Pairwise comparisons
        pairs = [("low", "medium"), ("medium", "high"), ("low", "high")]

        for g1, g2 in pairs:
            zb_mu_1, zb_sig_1 = zb_stats[g1]
            zb_mu_2, zb_sig_2 = zb_stats[g2]
            in_mu_1, in_sig_1 = in_stats[g1]
            in_mu_2, in_sig_2 = in_stats[g2]

            zb_dist = _frechet_distance(zb_mu_1, zb_sig_1, zb_mu_2, zb_sig_2)
            in_dist = _frechet_distance(in_mu_1, in_sig_1, in_mu_2, in_sig_2)

            lines.append(
                format_pairwise_block(
                    g1=g1,
                    g2=g2,
                    zb_dist=zb_dist,
                    in_dist=in_dist,
                    zb_dist_ref=zb_dist_ref,
                    in_dist_ref=in_dist_ref,
                )
            )
            lines.append("")

        log.info("\n".join(lines).rstrip())

    # 1. Dataset resolution — download / re-split / reuse as needed
    log.info(section("Step 1 | Dataset resolution"))
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
    log.info(kv("Resolved dataset path", dataset_path))

    # 2. Inject resolved path into dataloader config
    log.info(section("Step 2 | Config injection"))
    with open_dict(cfg):
        cfg.data.dataloader.cache_dir = cfg.data.dataloader.data_dir
        cfg.data.dataloader.data_dir = dataset_path
    log.info(kv("cache_dir", cfg.data.dataloader.cache_dir))
    log.info(kv("data_dir", cfg.data.dataloader.data_dir))

    # 3. Build dataloaders
    log.info(section("Step 3 | Dataloader initialization"))
    train_loader = instantiate(cfg.data.dataloader.train)
    val_loader = instantiate(cfg.data.dataloader.val)
    log.info(kv("Train batches", len(train_loader)))
    log.info(kv("Val batches", len(val_loader)))

    metrics_fn = jax.jit(jax.vmap(morphology_metrics))
    dataframes = []
    batches = []

    log.info(section("Step 4 | Extract data"))
    n_batches = len(val_loader)

    for batch, _ in tqdm(
        val_loader,
        total=n_batches,
        desc="Validation batches",
        dynamic_ncols=True,
    ):
        batch = np.array(batch.numpy(), copy=True)
        batch_metrics = metrics_fn(batch)
        dataframes.append(pd.DataFrame(batch_metrics))
        batches.append(batch)

    images = np.concatenate(batches)
    metrics_df = pd.concat(dataframes, ignore_index=True)
    log.info(kv("Collected images", len(images)))
    log.info(kv("Metrics rows", len(metrics_df)))

    log.info(section("Step 5 | Build encoders and accumulators"))
    zoobot = build_zoobot_nano()
    inception = build_headless_inceptionv3()

    zb_acc = FIDAccumulator(zoobot)
    in_acc = FIDAccumulator(inception)
    log.info(kv("Zoobot accumulator", "ready"))
    log.info(kv("Inception accumulator", "ready"))

    log.info(banner("Reference FIDs"))

    # Use random 50/50 split of val dataset to calculate FIDs
    mask = np.random.binomial(n=1, p=0.5, size=len(metrics_df)).astype(bool)
    split_1 = images[mask]
    split_2 = images[~mask]

    log.info(kv("Split 1 size", len(split_1)))
    log.info(kv("Split 2 size", len(split_2)))

    zb_mu_1, zb_sig_1 = get_mu_sig(zb_acc, split_1, desc="Zoobot ref split 1")
    in_mu_1, in_sig_1 = get_mu_sig(in_acc, split_1, desc="Inception ref split 1")

    zb_mu_2, zb_sig_2 = get_mu_sig(zb_acc, split_2, desc="Zoobot ref split 2")
    in_mu_2, in_sig_2 = get_mu_sig(in_acc, split_2, desc="Inception ref split 2")

    zb_dist_ref = _frechet_distance(zb_mu_1, zb_sig_1, zb_mu_2, zb_sig_2)
    in_dist_ref = _frechet_distance(in_mu_1, in_sig_1, in_mu_2, in_sig_2)

    log.info(kv("Zoobot ref. FID", f"{zb_dist_ref:.3g}"))
    log.info(kv("InceptionV3 ref. FID", f"{in_dist_ref:.3g}"))

    metrics_to_check = ["axis_ratio", "concentration", "asymmetry"]
    log.info(section("Step 6 | Metric comparisons"))
    log.info(kv("Metrics", ", ".join(metrics_to_check)))

    for metric in metrics_to_check:
        compare_metric_groups(
            images=images,
            metrics=metrics_df,
            metric=metric,
            zb_acc=zb_acc,
            in_acc=in_acc,
            zb_dist_ref=zb_dist_ref,
            in_dist_ref=in_dist_ref,
        )

    log.info(banner("Done"))


if __name__ == "__main__":
    main()
