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
    """
    Traverses the TNG50-1 API to extract idealized SKIRT images.

    Parameters
    ----------
    - versiond_ids: list[int]
        List of integers representing the versions (e.g., [0, 1, 2, 3])
    - snap_ids: list[int]
        List of integers representing the snapshots (e.g., [72, 73, 74])
    - headers: dict
        http headers to pass to requests. Must contain valid 'api-key' entry.
    - N: int, optional
        Maximum number of URLs to extract per combination. If 0, extracts all.
        Defaults to 0.

    Returns
    ----------
    list[str]
        A flat list of string URLs pointing to the .fits files.
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
    """
    Downloads a single TNG image FITS file.

    Parameters
    ----------
    - url: str
        Url of the FITS file to be downloaded
    - save_dir: str
        Directory to save the downloaded FITS file in
    - headers: dict
        http headers to pass to requests. Must contain valid 'api-key' entry.
    - max_retries: int, optional
        Maximum number of retries in case of timeouts. Defaults to 4.
    - timeout_base: int, optional
        Base time out. For the i'th attempt, the timeout is set to timeout_base**i.
        Defaults to 3.

    Returns
    ----------
    str
        Status string on whether download succeded or not.
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
