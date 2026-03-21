"""Download TNG50 galaxy FITS images from the IllustrisTNG API.

Traverses the TNG50-1 SKIRT image endpoints, extracts per-subhalo FITS
URLs, and downloads them in parallel with exponential-backoff retry.
"""

import os
import time
import hydra
import logging
import requests
import itertools

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_snapshot_ids", lambda start, count: [start + i for i in range(count)]
)


def extract_tng_urls(
    version_ids: list[int],
    snap_ids: list[int],
    headers: dict,
    N: int = 0,
) -> list[str]:
    """Traverse the TNG50-1 API to extract idealized SKIRT image URLs.

    Args:
        version_ids: Version integers (e.g. ``[0, 1, 2, 3]``).
        snap_ids: Snapshot integers (e.g. ``[72, 73, 74]``).
        headers: HTTP headers; must contain a valid ``'api-key'`` entry.
        N: Maximum URLs to extract per combination. ``0`` extracts all.

    Returns:
        Flat list of URLs pointing to ``.fits`` files.
    """

    url_list = []
    combos = list(itertools.product(version_ids, snap_ids))

    for vId, snap in tqdm(combos):
        endpoint_url = f"http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v{vId}_{snap}/"

        try:
            response = requests.get(endpoint_url, headers=headers)
            response.raise_for_status()
            urls = response.json()["files"]
            selected_urls = urls[:N] if N > 0 else urls
            url_list.extend(selected_urls)

        except requests.exceptions.HTTPError as err:
            log.warning(
                f"Skipping version {vId}, snapID {snap} - API returned {err.response.status_code}"
            )
        except requests.exceptions.RequestException as err:
            log.error(f"Network error on version {vId}, snapID {snap}: {err}")

    return url_list


def download_tng_fits_file(
    url: str, save_dir: str, headers: dict, max_retries: int = 4, timeout_base: int = 3
) -> str:
    """Download a single TNG FITS file with exponential-backoff retry.

    Args:
        url: URL of the FITS file.
        save_dir: Directory to save the downloaded file.
        headers: HTTP headers; must contain a valid ``'api-key'`` entry.
        max_retries: Maximum retry attempts on transient errors.
        timeout_base: Base for exponential back-off (sleep = ``base ** attempt``).

    Returns:
        Status string indicating success or failure.
    """

    url_parts = url.split("/")
    snap_id = url_parts[6]
    subhalo_id = url_parts[8]
    original_filename = url_parts[-1]

    unique_filename = f"snap_{snap_id}_subhalo_{subhalo_id}_{original_filename}"
    save_path = os.path.join(save_dir, unique_filename)

    if os.path.exists(save_path):
        return f"Already exists: {unique_filename}"

    for attempt in range(max_retries):
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(15, 120)
            ) as response:

                if response.status_code in [502, 503, 504]:
                    sleep_time = timeout_base**attempt
                    log.warning(
                        f"Server busy ({response.status_code}) for {unique_filename}. Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()

                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            return f"Successfully downloaded: {unique_filename}"

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            sleep_time = timeout_base**attempt
            log.warning(
                f"Connection dropped for {unique_filename}. Retrying in {sleep_time}s..."
            )
            time.sleep(sleep_time)
        except requests.exceptions.RequestException as e:
            return f"Failed to download {unique_filename} (Attempt {attempt+1}): {e}"

    return f"Failed completely after {max_retries} attempts: {unique_filename}"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point: extract TNG URLs and download FITS files in parallel."""

    dl_cfg = cfg.data.download
    os.makedirs(dl_cfg.raw_dir, exist_ok=True)
    headers = {"api-key": dl_cfg.api_key}

    log.info("Starting extraction of TNG URLs...")
    urls = extract_tng_urls(
        version_ids=dl_cfg.version_ids,
        snap_ids=dl_cfg.snapshots,
        headers=headers,
        N=dl_cfg.num_files_per_view,
    )

    log.info(f"Extracted {len(urls)} URLs. Beginning download via thread pool...")
    with ThreadPoolExecutor(max_workers=dl_cfg.max_workers) as executor:
        future_to_url = {
            executor.submit(download_tng_fits_file, url, dl_cfg.raw_dir, headers): url
            for url in urls
        }

        progress_bar = tqdm(
            as_completed(future_to_url), total=len(urls), desc="Downloading FITS files"
        )

        for future in progress_bar:
            try:
                result = future.result()

                if "Failed" in result:
                    tqdm.write(f"ERROR: {result}")
                    log.error(result)

            except Exception as exc:
                tqdm.write(f"EXCEPTION: {exc}")
                log.error(f"Generated an exception: {exc}")

    log.info("Download process complete.")


if __name__ == "__main__":
    main()
