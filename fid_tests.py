import os
import logging

import hydra
import numpy as np
import pandas as pd

from tqdm import tqdm
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, open_dict

from msdflow.tracking import setup_task
from msdflow.utils import register_all_resolvers
from msdflow.data.pipeline import resolve_dataset


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # 0. ClearML setup (may fork a background monitor subprocess)
    task = setup_task(cfg.clearml)

    import jax
    import jax.random as jr
    import jax.numpy as jnp

    from msdflow.model.convnext import build_zoobot_nano
    from msdflow.model.inceptionv3 import build_headless_inceptionv3
    from msdflow.train.metrics import (
        FIDAccumulator,
        _frechet_distance,
        morphology_metrics,
    )

    def get_mu_sig(acc, imgs, batch_size=128):
        acc.reset()
        total = (len(imgs) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(imgs), batch_size), total=total):
            acc.update(jnp.asarray(imgs[i : i + batch_size]))
        mu, sig, _ = acc.statistics()
        acc.reset()
        return mu, sig

    def split_into_three_groups(values):
        """
        Split indices into low / medium / high groups with near-equal counts.
        """
        values = np.asarray(values)
        order = np.argsort(values)
        idx_low, idx_med, idx_high = np.array_split(order, 3)

        groups = {
            "low": idx_low,
            "medium": idx_med,
            "high": idx_high,
        }
        return groups

    def summarize_groups(values, groups):
        print_string = "\n"
        for name, idx in groups.items():
            vals = values[idx]
            print_string += (
                f"\n{name:>6}: n={len(idx):5d}, "
                f"\nrange=[{vals.min():.3g}, {vals.max():.3g}], "
                f"\nmean={vals.mean():.3g}, median={np.median(vals):.3g}\n"
            )
        return print_string

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

        print_string = f"\nMetric: {metric}"
        print_string += summarize_groups(values, groups)

        # Precompute Gaussian stats for each group for both encoders
        zb_stats = {}
        in_stats = {}

        for name, idx in groups.items():
            imgs = images[idx]
            zb_stats[name] = get_mu_sig(zb_acc, imgs)
            in_stats[name] = get_mu_sig(in_acc, imgs)

        # Pairwise comparisons
        pairs = [("low", "medium"), ("medium", "high"), ("low", "high")]

        print_string += "\nPairwise FID comparisons:"
        for g1, g2 in pairs:
            zb_mu_1, zb_sig_1 = zb_stats[g1]
            zb_mu_2, zb_sig_2 = zb_stats[g2]
            in_mu_1, in_sig_1 = in_stats[g1]
            in_mu_2, in_sig_2 = in_stats[g2]

            zb_dist = _frechet_distance(zb_mu_1, zb_sig_1, zb_mu_2, zb_sig_2)
            in_dist = _frechet_distance(in_mu_1, in_sig_1, in_mu_2, in_sig_2)

            print_string += (
                f"\n{g1.upper()} vs {g2.upper()}"
                + f"\n  Zoobot      FID: {zb_dist:.4g}"
                + f"\n  InceptionV3 FID: {in_dist:.4g}"
            )

            if zb_dist_ref is not None:
                delta_zb = zb_dist - zb_dist_ref
                frac_zb = delta_zb / zb_dist_ref
                print_string += (
                    f"\n  Zoobot      ΔFID: {delta_zb:.4g}, "
                    f"\nFrac. Δ: {100 * frac_zb:.3g}%"
                )

            if in_dist_ref is not None:
                delta_in = in_dist - in_dist_ref
                frac_in = delta_in / in_dist_ref
                print_string += (
                    f"\n  InceptionV3 ΔFID: {delta_in:.4g}, "
                    f"\nFrac. Δ: {100 * frac_in:.3g}%"
                )

            if (zb_dist_ref is not None) and (in_dist_ref is not None):
                rel_zb = zb_dist / zb_dist_ref
                rel_in = in_dist / in_dist_ref
                print_string += (
                    f"\n  Relative separation vs ref: "
                    f"\nZoobot={rel_zb:.3f}x, InceptionV3={rel_in:.3f}x"
                )

        log.info(print_string)

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

    metrics_fn = jax.jit(jax.vmap(morphology_metrics))
    dataframes = []
    batches = []

    log.info("--- Step 4: Extract Data ---")
    n_batches = len(val_loader)
    i = 0
    for batch, _ in tqdm(val_loader, total=n_batches):
        batch = np.array(batch.numpy(), copy=True)
        batch_metrics = metrics_fn(batch)
        dataframes.append(pd.DataFrame(batch_metrics))
        batches.append(batch)
        i += 1
        if i == 2:
            break

    images = np.concatenate(batches)

    print("first-5 sample means:", [float(images[i].mean()) for i in range(5)])
    print("first-5 sample stds: ", [float(images[i].std()) for i in range(5)])
    print(
        "inter-sample std of pixel means:",
        float(images.reshape(images.shape[0], -1).mean(axis=1).std()),
    )
    print("range:", float(images.min()), float(images.max()))

    metrics_df = pd.concat(dataframes, ignore_index=True)

    zoobot = build_zoobot_nano()
    inception = build_headless_inceptionv3()

    zb_acc = FIDAccumulator(zoobot)
    in_acc = FIDAccumulator(inception)

    # ---------------------------------------------------------------------------- #
    #                                Reference FIDs                                #
    # ---------------------------------------------------------------------------- #

    # Use random 50/50 split of val dataset to caclulate FIDs
    mask = np.random.binomial(n=1, p=0.5, size=len(metrics_df)).astype(bool)

    split_1 = images[mask]
    split_2 = images[~mask]

    log.info("--- Calculate Reference FIDs ---")
    zb_mu_1, zb_sig_1 = get_mu_sig(zb_acc, split_1)
    in_mu_1, in_sig_1 = get_mu_sig(in_acc, split_1)

    zb_mu_2, zb_sig_2 = get_mu_sig(zb_acc, split_2)
    in_mu_2, in_sig_2 = get_mu_sig(in_acc, split_2)

    zb_dist_ref = _frechet_distance(zb_mu_1, zb_sig_1, zb_mu_2, zb_sig_2)
    in_dist_ref = _frechet_distance(in_mu_1, in_sig_1, in_mu_2, in_sig_2)

    log.info(f"Zoobot Ref. FID: {zb_dist_ref:.3g}")
    log.info(f"InceptionV3 Ref. FID: {in_dist_ref:.3g}")

    metrics_to_check = ["axis_ratio", "concentration", "asymmetry"]
    for metric in metrics_to_check:

        log.info(
            f"\n----------------------- FID Comparison for {metric} -----------------------\n"
        )
        compare_metric_groups(
            images=images,
            metrics=metrics_df,
            metric=metric,
            zb_acc=zb_acc,
            in_acc=in_acc,
            zb_dist_ref=zb_dist_ref,
            in_dist_ref=in_dist_ref,
        )
        print("\n")


if __name__ == "__main__":
    main()
