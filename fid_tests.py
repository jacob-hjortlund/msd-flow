import logging

import jax
import hydra
import numpy as np
import pandas as pd
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm
from hydra.utils import instantiate, call
from jax.scipy.ndimage import map_coordinates
from omegaconf import DictConfig, OmegaConf, open_dict

from msdflow.tracking import setup_task
from msdflow.utils import register_all_resolvers
from msdflow.data.pipeline import resolve_dataset
from msdflow.model.convnext import build_zoobot_nano
from msdflow.model.inceptionv3 import build_headless_inceptionv3
from msdflow.train.metrics import FIDAccumulator, _frechet_distance

register_all_resolvers()
log = logging.getLogger(__name__)


def _safe_divide(num, den, eps=1e-12):
    return num / jnp.maximum(den, eps)


def _prepare_image(img):
    """
    img: array of shape (1, N, N)
    Returns:
        J: non-negative 2D image of shape (N, N)
    """
    assert img.ndim == 3 and img.shape[0] == 1, "Expected image of shape (1, N, N)"
    # J = jnp.maximum(img[0], 0.0)
    J = (img[0] + 1.0) / 2.0
    return J


def _coordinate_grids(N):
    """
    Returns X, Y coordinate grids of shape (N, N).
    X is horizontal (column index), Y is vertical (row index).
    """
    y = jnp.arange(N)
    x = jnp.arange(N)
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    return X, Y


def centroid(img, eps=1e-12):
    """
    Intensity-weighted centroid.

    img: shape (1, N, N)
    Returns:
        xc, yc
    """
    # J = _prepare_image(img)
    # N = J.shape[0]
    # X, Y = _coordinate_grids(N)
    # total = jnp.sum(J)
    # xc = _safe_divide(jnp.sum(J * X), total, eps)
    # yc = _safe_divide(jnp.sum(J * Y), total, eps)
    # return xc, yc
    return (256, 256)


def _radii_and_sorted_intensity(J, xc, yc):
    """
    Flattened radii and intensities sorted by radius.
    """
    N = J.shape[0]
    X, Y = _coordinate_grids(N)
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    r_flat = r.reshape(-1)
    J_flat = J.reshape(-1)

    order = jnp.argsort(r_flat)
    r_sorted = r_flat[order]
    J_sorted = J_flat[order]
    return r_sorted, J_sorted


def radius_at_fraction(img, frac, eps=1e-12):
    """
    Radius enclosing a given fraction of total non-negative intensity.

    img: shape (1, N, N)
    frac: scalar in [0, 1]
    """
    J = _prepare_image(img)
    # xc, yc = centroid(img, eps=eps)
    xc, yc = (256, 256)
    r_sorted, J_sorted = _radii_and_sorted_intensity(J, xc, yc)

    cumulative = jnp.cumsum(J_sorted)
    total = jnp.sum(J_sorted)
    target = frac * total

    idx = jnp.searchsorted(cumulative, target, side="left")
    idx = jnp.clip(idx, 0, r_sorted.shape[0] - 1)
    return r_sorted[idx]


def half_light_radius(img, eps=1e-12):
    return radius_at_fraction(img, 0.5, eps=eps)


def concentration(img, eps=1e-12):
    """
    C = 5 log10(r80 / r20)
    """
    r20 = radius_at_fraction(img, 0.2, eps=eps)
    r80 = radius_at_fraction(img, 0.8, eps=eps)
    C = 5.0 * jnp.log10(_safe_divide(r80, r20, eps))
    return C, r20, r80


def second_moments(img, eps=1e-12):
    """
    Intensity-weighted second central moments.

    Returns:
        Mxx, Myy, Mxy
    """
    J = _prepare_image(img)
    N = J.shape[0]
    X, Y = _coordinate_grids(N)
    # xc, yc = centroid(img, eps=eps)
    xc, yc = (256, 256)

    total = jnp.sum(J)

    dx = X - xc
    dy = Y - yc

    Mxx = _safe_divide(jnp.sum(J * dx * dx), total, eps)
    Myy = _safe_divide(jnp.sum(J * dy * dy), total, eps)
    Mxy = _safe_divide(jnp.sum(J * dx * dy), total, eps)

    return Mxx, Myy, Mxy


def shape_metrics(img, eps=1e-12):
    """
    Axis ratio, ellipticity, and position angle from second moments.

    Returns:
        q : axis ratio b/a
        e : ellipticity = 1 - q
        theta : position angle in radians
        a, b : sqrt(eigenvalues)
    """
    Mxx, Myy, Mxy = second_moments(img, eps=eps)

    cov = jnp.array([[Mxx, Mxy], [Mxy, Myy]])
    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # eigh returns ascending eigenvalues
    lam2, lam1 = eigvals[0], eigvals[1]  # lam1 >= lam2
    a = jnp.sqrt(jnp.maximum(lam1, 0.0))
    b = jnp.sqrt(jnp.maximum(lam2, 0.0))

    q = _safe_divide(b, a, eps)
    e = 1.0 - q

    # Position angle of major axis
    # Equivalent formula:
    # theta = 0.5 * arctan2(2 Mxy, Mxx - Myy)
    theta = 0.5 * jnp.arctan2(2.0 * Mxy, Mxx - Myy)

    return q, e, theta, a, b


def _rotate_180_about_center(J, xc, yc, order=1):
    """
    Rotate a 2D image J by 180 degrees about (xc, yc), using interpolation.

    Returns rotated image of same shape.
    """
    N = J.shape[0]
    X, Y = _coordinate_grids(N)

    # 180-degree rotation about (xc, yc):
    # x' = 2 xc - x
    # y' = 2 yc - y
    X_src = 2.0 * xc - X
    Y_src = 2.0 * yc - Y

    coords = jnp.stack([Y_src, X_src], axis=0)  # map_coordinates expects (row, col)
    J_rot = map_coordinates(J, coords, order=order, mode="constant", cval=0.0)
    return J_rot


def asymmetry(img, eps=1e-12):
    """
    A = sum |J - J_180| / sum |J|

    Uses 180-degree rotation about the intensity-weighted centroid.
    """
    J = _prepare_image(img)
    # xc, yc = centroid(img, eps=eps)
    xc, yc = (256, 256)
    J_rot = _rotate_180_about_center(J, xc, yc, order=1)

    num = jnp.sum(jnp.abs(J - J_rot))
    den = jnp.sum(jnp.abs(J))
    A = _safe_divide(num, den, eps)
    return A


def morphology_metrics(img, eps=1e-12):
    """
    Compute a set of morphology metrics for a single image of shape (1, N, N).

    Returns a dict of JAX scalars.
    """
    # xc, yc = centroid(img, eps=eps)
    xc, yc = (256, 256)
    r50 = half_light_radius(img, eps=eps)
    C, r20, r80 = concentration(img, eps=eps)
    q, e, theta, a, b = shape_metrics(img, eps=eps)
    A = asymmetry(img, eps=eps)

    return {
        "xc": xc,
        "yc": yc,
        "r20": r20,
        "r50": r50,
        "r80": r80,
        "concentration": C,
        "axis_ratio": q,
        "ellipticity": e,
        "position_angle": theta,
        "a": a,
        "b": b,
        "asymmetry": A,
    }


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
    print(images.shape)

    print("first-5 sample means:", [float(images[i].mean()) for i in range(5)])
    print("first-5 sample stds: ", [float(images[i].std()) for i in range(5)])
    print(
        "inter-sample std of pixel means:",
        float(images.reshape(images.shape[0], -1).mean(axis=1).std()),
    )
    print("range:", float(images.min()), float(images.max()))

    metrics_df = pd.concat(dataframes, ignore_index=True)

    flat = images.reshape(images.shape[0], -1)

    print("n images:", len(images))
    print("pixel-mean std across samples:", flat.mean(axis=1).std())

    # Compare several pairs, not just one
    pairs = [(0, 1), (0, 10), (0, 100), (5, 105), (10, 110)]
    for a, b in pairs:
        print(a, b, np.abs(images[a] - images[b]).mean())

    # Check whether many images are literally identical
    diffs = np.abs(images[:, None] - images[None, :]).mean(axis=(2, 3, 4))
    print(
        "min off-diagonal mean abs diff:",
        diffs[np.triu_indices(len(images), k=1)].min(),
    )

    zoobot = build_zoobot_nano()
    inception = build_headless_inceptionv3()

    z0, z1 = zoobot(x0), zoobot(x1)
    i0, i1 = inception(x0), inception(x1)

    print("zoobot   diff:", float(jnp.abs(z0 - z1).mean()), "z0[:5]:", z0[:5])
    print("incep    diff:", float(jnp.abs(i0 - i1).mean()), "i0[:5]:", i0[:5])

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
