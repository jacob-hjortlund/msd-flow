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
    _sample_filenames,
)


# These will be initialized once per worker process
mag2flux = None
cap = None
dwn = None
worker_dataset_path = None


def init_worker(dataset_path):
    """
    Runs once in each worker process.
    This avoids recreating the transform objects for every image.
    """
    global mag2flux, cap, dwn, worker_dataset_path

    worker_dataset_path = dataset_path

    mag2flux = SurfaceBrightnessToNanomaggies()
    cap = ClipAndPad(n=512)
    dwn = Downsample(target_size=256)


def process_file(filename):
    """
    Process a single relative .npy filename and return its 99.9th percentile.
    """
    path = os.path.join(worker_dataset_path, filename)

    img = np.load(path)

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
    filenames = _sample_filenames(filenames, 0.25, 42)
    n = len(filenames)

    num_workers = min(32, os.cpu_count())
    ctx = mp.get_context("spawn")

    dataset_path = os.path.abspath(dataset_path)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(dataset_path,),
    ) as executor:
        percs = list(
            tqdm(
                executor.map(process_file, filenames, chunksize=16),
                total=n,
            )
        )

    percs = np.asarray(percs)
    eps = 1e-8
    z = np.log10(percs + eps)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(z.reshape(-1, 1))
    log_boundary = float(np.sum(kmeans.cluster_centers_) / 2)
    boundary = 10**log_boundary - eps

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].hist(percs, bins="doane")
    ax[0].axvline(boundary, c="k")
    ax[1].hist(z, bins="doane")
    ax[1].axvline(log_boundary, c="k")
    fig.tight_layout()

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
