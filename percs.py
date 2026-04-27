import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from msdflow.tracking import setup_task
from sklearn.cluster import KMeans

import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from msdflow.data.pipeline import resolve_dataset


from msdflow.data.preprocess import (
    SurfaceBrightnessToNanomaggies,
    ClipAndPad,
    Downsample,
)


# These will be initialized once per worker process
mag2flux = None
cap = None
dwn = None


def init_worker():
    """
    Runs once in each worker process.
    This avoids recreating the transform objects for every image.
    """
    global mag2flux, cap, dwn

    mag2flux = SurfaceBrightnessToNanomaggies()
    cap = ClipAndPad(n=512)
    dwn = Downsample(target_size=256)


def process_file(filename):
    """
    Process a single .npy file and return its 99.9th percentile.
    """
    img = np.load(filename)

    img = mag2flux(img)
    img = cap(img)
    img = dwn(img)

    return np.percentile(img, 99.9)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    task = setup_task(cfg.clearml)

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

    csv_path = os.path.join(dataset_path, "metadata.csv")
    metadata = pd.read_csv(csv_path)

    metadata = metadata[metadata["split"] == "train"]
    filenames = metadata["filename"].tolist()
    n = len(filenames)

    # Choose a sensible number for your node.
    # For example, 8 or 16 if running on the shared A100 node.
    num_workers = min(8, os.cpu_count())

    # "spawn" is safer if your broader codebase uses JAX / CUDA / fork-sensitive libraries.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
    ) as executor:
        percs = list(
            tqdm(
                executor.map(process_file, filenames, chunksize=16),
                total=n,
            )
        )

    percs = np.asarray(percs)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(percs[:, None])
    clip = np.sum(kmeans.cluster_centers_) / 2

    fig, ax = plt.subplots()
    ax.hist(percs, bins="doane")
    ax.axvline(clip, c="k")

    cl_logger = task.get_logger()
    cl_logger.report_matplotlib_figure(
        title="flux dist",
        series="flux dist",
        iteration=0,
        figure=fig,
        report_image=False,
        report_interactive=False,
    )


if __name__ == "__main__":
    main()
