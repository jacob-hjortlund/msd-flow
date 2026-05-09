"""Weight loading helpers for InceptionV3."""

import os
import tempfile

import requests
from tqdm import tqdm

__all__ = ["download", "get"]


def download(url, ckpt_dir="data"):
    """Download an InceptionV3 checkpoint when it is not already cached.

    Args:
        url: URL for the checkpoint file.
        ckpt_dir: Directory where the checkpoint should be cached. Uses the
            system temporary directory when ``None``.

    Returns:
        Path to the cached checkpoint file.
    """
    name = url[url.rfind("/") + 1 : url.rfind("?")]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: "{url[:url.rfind("?")]}" to {ckpt_file}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + ".temp")
        with open(ckpt_file_temp, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("An error occured while downloading, please try again.")
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


def get(dictionary, key):
    """Return a nested parameter dictionary entry when it exists.

    Args:
        dictionary: Parameter dictionary or ``None``.
        key: Entry name to read.

    Returns:
        The dictionary entry for ``key``, or ``None`` when absent.
    """
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]
