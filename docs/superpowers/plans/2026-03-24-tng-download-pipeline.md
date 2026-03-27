# TNG50 Download and Processing Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `download_tng.py` into a batched download-extract-cleanup pipeline with resumption, and add a PyTorch `TNG50Dataset` for data loading.

**Architecture:** Batch loop in `main()` orchestrates download → extract → save_metadata → cleanup per batch. Individual `.npy` files + `metadata.csv` on disk. `TNG50Dataset` reads the metadata index for random access.

**Tech Stack:** Python, NumPy, pandas, astropy, PyTorch, Hydra/OmegaConf, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-tng-download-pipeline-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/data/download_tng.py` | Add `fits_name_from_url`, `load_fits`, `download_batch`, `extract_batch`, `cleanup_batch`, `save_metadata`; refactor `download_tng_fits_file` return type; rewrite `main()` |
| Modify | `src/data/preprocess.py` | Remove `load_fits` |
| Create | `src/data/dataset.py` | `TNG50Dataset` |
| Modify | `configs/data/download.yaml` | Add `bands`, `batch_size`, `processed_dir` |
| Modify | `tests/data/test_download_tng.py` | Update existing tests, add tests for new functions |
| Modify | `tests/data/test_preprocess.py` | Remove `load_fits` import (it was never tested here, but verify) |
| Create | `tests/data/test_dataset.py` | Tests for `TNG50Dataset` |

---

### Task 1: Extract `fits_name_from_url` and refactor `download_tng_fits_file`

**Files:**
- Modify: `src/data/download_tng.py:66-125`
- Modify: `tests/data/test_download_tng.py`

- [ ] **Step 1: Write failing tests for `fits_name_from_url`**

In `tests/data/test_download_tng.py`, add:

```python
from src.data.download_tng import fits_name_from_url


def test_fits_name_from_url_extracts_identifier():
    """Verify FITS name is extracted from a standard TNG API URL."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"
    result = fits_name_from_url(url)
    assert result == "snap_skirt_images_hsc_idealized_v0_72_subhalo_12345_image"


def test_fits_name_from_url_strips_extension():
    """Verify .fits extension is stripped from the result."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v2_80/subhalos/999/galaxy.fits"
    result = fits_name_from_url(url)
    assert result == "snap_skirt_images_hsc_idealized_v2_80_subhalo_999_galaxy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py::test_fits_name_from_url_extracts_identifier tests/data/test_download_tng.py::test_fits_name_from_url_strips_extension -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `fits_name_from_url` and refactor `download_tng_fits_file`**

In `src/data/download_tng.py`, add the new import and function before `download_tng_fits_file`:

```python
def fits_name_from_url(url: str) -> str:
    """Extract a unique FITS identifier from a TNG API URL.

    Parses the URL path to build a name from the snapshot ID, subhalo ID,
    and original filename (without extension). This is the single source
    of truth for URL-to-filename mapping.

    Args:
        url: Full TNG API URL pointing to a FITS file.

    Returns:
        Identifier string, e.g. ``'snap_skirt_..._subhalo_12345_image'``.
    """
    url_parts = url.split("/")
    snap_id = url_parts[6]
    subhalo_id = url_parts[8]
    original_filename = url_parts[-1]
    name = f"snap_{snap_id}_subhalo_{subhalo_id}_{original_filename}"
    return os.path.splitext(name)[0]
```

Then refactor `download_tng_fits_file` to use it and return `str | None`:

```python
def download_tng_fits_file(
    url: str, save_dir: str, headers: dict, max_retries: int = 4, timeout_base: int = 3
) -> str | None:
    """Download a single TNG FITS file with exponential-backoff retry.

    Args:
        url: URL of the FITS file.
        save_dir: Directory to save the downloaded file.
        headers: HTTP headers; must contain a valid ``'api-key'`` entry.
        max_retries: Maximum retry attempts on transient errors.
        timeout_base: Base for exponential back-off (sleep = ``base ** attempt``).

    Returns:
        Path to the downloaded file on success, or ``None`` on failure.
    """
    fits_name = fits_name_from_url(url)
    unique_filename = fits_name + ".fits"
    save_path = os.path.join(save_dir, unique_filename)

    if os.path.exists(save_path):
        return save_path

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

            return save_path

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            sleep_time = timeout_base**attempt
            log.warning(
                f"Connection dropped for {unique_filename}. Retrying in {sleep_time}s..."
            )
            time.sleep(sleep_time)
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to download {unique_filename} (Attempt {attempt+1}): {e}")
            return None

    log.error(f"Failed completely after {max_retries} attempts: {unique_filename}")
    return None
```

- [ ] **Step 4: Update existing tests for the new return type**

In `tests/data/test_download_tng.py`, update:

```python
def test_download_skips_existing_file(tmp_path):
    """Verify download returns path when the file already exists."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"
    expected_name = "snap_skirt_images_hsc_idealized_v0_72_subhalo_12345_image.fits"
    (tmp_path / expected_name).touch()

    result = download_tng_fits_file(url, str(tmp_path), headers={"api-key": "test"})
    assert result == str(tmp_path / expected_name)


@patch("src.data.download_tng.requests.get")
def test_download_success(mock_get, tmp_path):
    """Verify a successful download returns the file path."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [b"fake fits data"]
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_response

    result = download_tng_fits_file(url, str(tmp_path), headers={"api-key": "test"})
    expected_name = "snap_skirt_images_hsc_idealized_v0_72_subhalo_12345_image.fits"
    assert result == str(tmp_path / expected_name)
```

- [ ] **Step 5: Run all download tests**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/download_tng.py tests/data/test_download_tng.py
git commit -m "refactor: extract fits_name_from_url, change download return type to path|None"
```

---

### Task 2: Move `load_fits` to `download_tng.py` and generalize to multi-band

**Files:**
- Modify: `src/data/download_tng.py`
- Modify: `src/data/preprocess.py:22-39`
- Modify: `tests/data/test_download_tng.py`

- [ ] **Step 1: Write failing tests for multi-band `load_fits`**

In `tests/data/test_download_tng.py`, add:

```python
import numpy as np
from astropy.io import fits as astropy_fits

from src.data.download_tng import load_fits


@pytest.fixture
def multi_band_fits(tmp_path):
    """Create a multi-extension FITS file with g, r, i bands."""
    hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU()])
    for band in ["g", "r", "i"]:
        data = np.random.default_rng(ord(band[0])).random((64, 64)).astype(np.float32)
        hdu = astropy_fits.ImageHDU(data=data, name=band)
        hdu.header["TESTKEY"] = f"value_{band}"
        hdul.append(hdu)
    path = tmp_path / "test.fits"
    hdul.writeto(path)
    return str(path)


def test_load_fits_single_band(multi_band_fits):
    """Verify loading a single band returns (1, H, W) array and header dict."""
    data, header = load_fits(multi_band_fits, ["g"])
    assert data.shape == (1, 64, 64)
    assert header["TESTKEY"] == "value_g"  # raw header dict, no hdr_ prefix


def test_load_fits_multi_band_stacks_in_order(multi_band_fits):
    """Verify multiple bands are stacked in the order given."""
    data, header = load_fits(multi_band_fits, ["i", "g"])
    assert data.shape == (2, 64, 64)
    # Header should come from the first band in the list (i)
    assert header["TESTKEY"] == "value_i"


def test_load_fits_missing_band_raises(multi_band_fits):
    """Verify ValueError is raised if a band is not found."""
    with pytest.raises(ValueError, match="Band 'z' not found"):
        load_fits(multi_band_fits, ["g", "z"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py::test_load_fits_single_band tests/data/test_download_tng.py::test_load_fits_multi_band_stacks_in_order tests/data/test_download_tng.py::test_load_fits_missing_band_raises -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement multi-band `load_fits` in `download_tng.py`**

Add at the top of `src/data/download_tng.py` imports:

```python
import numpy as np
from astropy.io import fits
```

Add the function after the existing imports section, before `fits_name_from_url`:

```python
def load_fits(filename: str, bands: list[str]) -> tuple[np.ndarray, dict]:
    """Load one or more bands from a multi-extension FITS file.

    Args:
        filename: Path to the FITS file.
        bands: ``EXTNAME`` values to extract, in stacking order.

    Returns:
        Tuple of the stacked image array with shape ``(C, H, W)`` and
        the FITS header dict from the first band's extension.

    Raises:
        ValueError: If any band in *bands* is not found in the file.
    """
    arrays = []
    header = None
    with fits.open(filename) as hdul:
        for band in bands:
            found = False
            for hdu in hdul:
                if hdu.header.get("EXTNAME") == band:
                    arrays.append(hdu.data)
                    if header is None:
                        header = dict(hdu.header)
                    found = True
                    break
            if not found:
                raise ValueError(f"Band '{band}' not found in {filename}")
    return np.stack(arrays, axis=0), header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py::test_load_fits_single_band tests/data/test_download_tng.py::test_load_fits_multi_band_stacks_in_order tests/data/test_download_tng.py::test_load_fits_missing_band_raises -v`
Expected: All PASS

- [ ] **Step 5: Remove `load_fits` from `preprocess.py`**

Delete lines 19–39 of `src/data/preprocess.py` (the `# --- I/O ---` comment block and the `load_fits` function). Also remove the `from astropy.io import fits` import on line 16 if it's no longer used by any other function in the file.

Check: `from astropy.io import fits` — the remaining functions in `preprocess.py` (`surface_brightness_to_nanomaggies`, `clip_and_pad`, `arcsinh_stretch`, `linear_normalize`, `preprocess_image`) use only `numpy`. The `astropy` imports on lines 10–13 and 16 can all be removed. However, only remove `from astropy.io import fits` (line 16) since the other astropy imports may be used elsewhere or planned for future use. Check if lines 10-14 (`import astropy`, `import astropy.units as u`, `import astropy.cosmology as ap_cosmo`) are used — they are not used by any function in the file, but since the spec says "no other changes", leave them and only remove the `fits` import and the `load_fits` function.

- [ ] **Step 6: Verify preprocess tests still pass**

Run: `cd msd-flow && python -m pytest tests/data/test_preprocess.py -v`
Expected: All PASS (no test in this file imports `load_fits`)

- [ ] **Step 7: Commit**

```bash
git add src/data/download_tng.py src/data/preprocess.py tests/data/test_download_tng.py
git commit -m "feat: move load_fits to download_tng.py, generalize to multi-band"
```

---

### Task 3: Implement `get_existing_ids` and `save_metadata`

**Files:**
- Modify: `src/data/download_tng.py`
- Modify: `tests/data/test_download_tng.py`

- [ ] **Step 1: Write failing tests**

In `tests/data/test_download_tng.py`, add:

```python
import pandas as pd

from src.data.download_tng import get_existing_ids, save_metadata


def test_get_existing_ids_empty_dir(tmp_path):
    """Verify empty set returned when no metadata.csv exists."""
    assert get_existing_ids(str(tmp_path)) == set()


def test_get_existing_ids_reads_fits_name_column(tmp_path):
    """Verify fits_name values are returned from existing metadata."""
    df = pd.DataFrame({"filename": ["g_00000.npy"], "fits_name": ["snap_v0_72_sub_1_img"]})
    df.to_csv(tmp_path / "metadata.csv", index=False)
    result = get_existing_ids(str(tmp_path))
    assert result == {"snap_v0_72_sub_1_img"}


def test_save_metadata_creates_new_csv(tmp_path):
    """Verify save_metadata creates metadata.csv with header row."""
    records = [{"filename": "galaxy_00000.npy", "fits_name": "snap_a", "band_map": "g"}]
    save_metadata(records, str(tmp_path))
    df = pd.read_csv(tmp_path / "metadata.csv")
    assert len(df) == 1
    assert df.iloc[0]["fits_name"] == "snap_a"


def test_save_metadata_appends_to_existing(tmp_path):
    """Verify save_metadata appends rows to an existing CSV."""
    records1 = [{"filename": "galaxy_00000.npy", "fits_name": "snap_a", "band_map": "g"}]
    save_metadata(records1, str(tmp_path))
    records2 = [{"filename": "galaxy_00001.npy", "fits_name": "snap_b", "band_map": "g"}]
    save_metadata(records2, str(tmp_path))
    df = pd.read_csv(tmp_path / "metadata.csv")
    assert len(df) == 2
    assert set(df["fits_name"]) == {"snap_a", "snap_b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py::test_get_existing_ids_empty_dir tests/data/test_download_tng.py::test_save_metadata_creates_new_csv -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `get_existing_ids` and `save_metadata`**

Add `import pandas as pd` and `import tempfile` to the imports in `src/data/download_tng.py`. Then add:

```python
def get_existing_ids(processed_dir: str) -> set[str]:
    """Read already-processed FITS identifiers from metadata.csv.

    Args:
        processed_dir: Directory containing ``metadata.csv``.

    Returns:
        Set of ``fits_name`` values, or empty set if no CSV exists.
    """
    csv_path = os.path.join(processed_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, usecols=["fits_name"], on_bad_lines="skip")
    return set(df["fits_name"])


def save_metadata(records: list[dict], processed_dir: str) -> None:
    """Atomically append metadata records to metadata.csv.

    Reads the existing CSV (if any), concatenates new rows, writes the
    full result to a temporary file, then atomically renames it.

    Args:
        records: List of metadata dicts (one per galaxy).
        processed_dir: Directory containing ``metadata.csv``.
    """
    csv_path = os.path.join(processed_dir, "metadata.csv")
    new_df = pd.DataFrame(records)
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, on_bad_lines="skip")
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    fd, tmp_path = tempfile.mkstemp(dir=processed_dir, suffix=".csv")
    os.close(fd)
    try:
        combined_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py::test_get_existing_ids_empty_dir tests/data/test_download_tng.py::test_get_existing_ids_reads_fits_name_column tests/data/test_download_tng.py::test_save_metadata_creates_new_csv tests/data/test_download_tng.py::test_save_metadata_appends_to_existing -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/download_tng.py tests/data/test_download_tng.py
git commit -m "feat: add get_existing_ids and save_metadata with atomic CSV writes"
```

---

### Task 4: Implement `download_batch`, `extract_batch`, and `cleanup_batch`

**Files:**
- Modify: `src/data/download_tng.py`
- Modify: `tests/data/test_download_tng.py`

- [ ] **Step 1: Write failing tests for `download_batch`**

```python
from src.data.download_tng import download_batch


@patch("src.data.download_tng.download_tng_fits_file")
def test_download_batch_filters_failures(mock_download, tmp_path):
    """Verify download_batch returns only successful paths."""
    mock_download.side_effect = [str(tmp_path / "a.fits"), None, str(tmp_path / "c.fits")]
    urls = ["url1", "url2", "url3"]
    result = download_batch(urls, str(tmp_path), {"api-key": "test"}, max_workers=2)
    assert len(result) == 2
```

- [ ] **Step 2: Write failing tests for `extract_batch`**

```python
from src.data.download_tng import extract_batch


def test_extract_batch_saves_npy_and_returns_metadata(tmp_path, multi_band_fits):
    """Verify extract_batch saves .npy files and returns metadata dicts.

    Note: depends on ``multi_band_fits`` fixture defined in Task 2.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    records = extract_batch([multi_band_fits], ["g", "r"], str(processed), start_idx=0)
    assert len(records) == 1
    assert records[0]["filename"] == "galaxy_00000.npy"
    assert records[0]["band_map"] == "g,r"
    # FITS header keys are prefixed with hdr_ to avoid collisions
    assert "hdr_TESTKEY" in records[0]
    saved = np.load(processed / "galaxy_00000.npy")
    assert saved.shape == (2, 64, 64)


def test_extract_batch_skips_failed_and_continues(tmp_path, multi_band_fits):
    """Verify failed extractions are skipped without gaps in numbering.

    Note: depends on ``multi_band_fits`` fixture defined in Task 2.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    bad_path = str(tmp_path / "nonexistent.fits")
    records = extract_batch([bad_path, multi_band_fits], ["g"], str(processed), start_idx=0)
    assert len(records) == 1
    assert records[0]["filename"] == "galaxy_00000.npy"
```

- [ ] **Step 3: Write failing test for `cleanup_batch`**

```python
from src.data.download_tng import cleanup_batch


def test_cleanup_batch_deletes_files(tmp_path):
    """Verify all listed files are deleted."""
    paths = []
    for name in ["a.fits", "b.fits"]:
        p = tmp_path / name
        p.touch()
        paths.append(str(p))
    cleanup_batch(paths)
    assert not any(os.path.exists(p) for p in paths)


def test_cleanup_batch_ignores_missing_files(tmp_path):
    """Verify cleanup does not error on already-deleted files."""
    cleanup_batch([str(tmp_path / "nonexistent.fits")])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py -k "download_batch or extract_batch or cleanup_batch" -v`
Expected: FAIL with ImportError

- [ ] **Step 5: Implement the three functions**

In `src/data/download_tng.py`:

```python
def download_batch(
    urls: list[str], raw_dir: str, headers: dict, max_workers: int
) -> list[str]:
    """Download a batch of FITS files in parallel.

    Args:
        urls: URLs to download.
        raw_dir: Directory for downloaded FITS files.
        headers: HTTP headers with API key.
        max_workers: Thread pool size.

    Returns:
        Paths of successfully downloaded files.
    """
    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_tng_fits_file, url, raw_dir, headers): url
            for url in urls
        }
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                if result is not None:
                    paths.append(result)
            except Exception as exc:
                log.error(f"Download exception: {exc}")
    return paths


def extract_batch(
    fits_paths: list[str],
    bands: list[str],
    processed_dir: str,
    start_idx: int,
) -> list[dict]:
    """Extract bands and metadata from FITS files, saving as .npy.

    Args:
        fits_paths: Paths to downloaded FITS files.
        bands: Band names to extract and stack.
        processed_dir: Output directory for ``.npy`` files.
        start_idx: Starting index for galaxy file numbering.

    Returns:
        List of metadata dicts for successfully extracted galaxies.
    """
    records = []
    success_count = 0
    for fits_path in fits_paths:
        try:
            data, header = load_fits(fits_path, bands)
        except Exception as exc:
            log.warning(f"Failed to extract {fits_path}: {exc}")
            continue

        idx = start_idx + success_count
        npy_name = f"galaxy_{idx:05d}.npy"
        np.save(os.path.join(processed_dir, npy_name), data)

        fits_name = os.path.splitext(os.path.basename(fits_path))[0]
        # Sanitize header: stringify values, replace newlines/commas,
        # drop empty keys, and prefix to avoid collisions with our fields.
        safe_header = {
            f"hdr_{k}": str(v).replace("\n", " ").replace(",", ";")
            for k, v in header.items()
            if k
        }
        record = {
            "filename": npy_name,
            "fits_name": fits_name,
            "band_map": ",".join(bands),
        }
        record.update(safe_header)
        records.append(record)
        success_count += 1

    return records


def cleanup_batch(fits_paths: list[str]) -> None:
    """Delete FITS files after extraction.

    Args:
        fits_paths: Paths to FITS files to delete.
    """
    for path in fits_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py -k "download_batch or extract_batch or cleanup_batch" -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/data/download_tng.py tests/data/test_download_tng.py
git commit -m "feat: add download_batch, extract_batch, cleanup_batch"
```

---

### Task 5: Rewrite `main()` with batch orchestration and resumption

**Files:**
- Modify: `src/data/download_tng.py:128-171`
- Modify: `configs/data/download.yaml`

- [ ] **Step 1: Update Hydra config**

Replace `configs/data/download.yaml` with:

```yaml
api_key: ${oc.env:TNG_API_KEY}
version_ids: [0,1,2,3]
snapshots: ${generate_snapshot_ids:72,20}
num_files_per_view: 50
max_workers: 5
raw_dir: "${hydra:runtime.cwd}/data/raw"
bands: ["g"]
batch_size: 100
processed_dir: "${hydra:runtime.cwd}/data/processed/g_band"
```

- [ ] **Step 2: Rewrite `main()`**

Replace the existing `main()` in `src/data/download_tng.py`:

```python
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point: download TNG FITS in batches, extract bands, and clean up."""

    dl_cfg = cfg.data.download
    os.makedirs(dl_cfg.raw_dir, exist_ok=True)
    os.makedirs(dl_cfg.processed_dir, exist_ok=True)
    headers = {"api-key": dl_cfg.api_key}

    log.info("Extracting TNG URLs...")
    urls = extract_tng_urls(
        version_ids=dl_cfg.version_ids,
        snap_ids=dl_cfg.snapshots,
        headers=headers,
        N=dl_cfg.num_files_per_view,
    )
    log.info(f"Found {len(urls)} total URLs.")

    existing_ids = get_existing_ids(dl_cfg.processed_dir)
    log.info(f"Resuming: {len(existing_ids)} galaxies already processed.")

    remaining_urls = [
        url for url in urls if fits_name_from_url(url) not in existing_ids
    ]
    log.info(f"{len(remaining_urls)} URLs remaining after resumption filter.")

    batch_size = dl_cfg.batch_size
    num_batches = (len(remaining_urls) + batch_size - 1) // batch_size
    start_idx = len(existing_ids)

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_urls = remaining_urls[batch_start : batch_start + batch_size]
        log.info(f"Batch {batch_num + 1}/{num_batches}: downloading {len(batch_urls)} files...")

        paths = download_batch(batch_urls, dl_cfg.raw_dir, headers, dl_cfg.max_workers)
        log.info(f"Downloaded {len(paths)}/{len(batch_urls)} files.")

        records = extract_batch(paths, list(dl_cfg.bands), dl_cfg.processed_dir, start_idx)
        log.info(f"Extracted {len(records)} galaxies.")

        if records:
            save_metadata(records, dl_cfg.processed_dir)

        cleanup_batch(paths)
        start_idx += len(records)

    log.info("Pipeline complete.")
```

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `cd msd-flow && python -m pytest tests/data/test_download_tng.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/data/download_tng.py configs/data/download.yaml
git commit -m "feat: rewrite main() with batched download-extract-cleanup loop and resumption"
```

---

### Task 6: Create `TNG50Dataset`

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_dataset.py`:

```python
"""Tests for src.data.dataset."""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import TNG50Dataset


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal processed directory with .npy files and metadata."""
    records = []
    for i in range(5):
        name = f"galaxy_{i:05d}.npy"
        data = np.random.default_rng(i).random((1, 64, 64)).astype(np.float32)
        np.save(tmp_path / name, data)
        records.append({"filename": name, "fits_name": f"snap_{i}", "band_map": "g"})
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


def test_dataset_length(sample_dataset):
    """Verify __len__ matches number of entries in metadata."""
    ds = TNG50Dataset(sample_dataset)
    assert len(ds) == 5


def test_dataset_getitem_returns_tensor(sample_dataset):
    """Verify __getitem__ returns a float tensor with correct shape."""
    ds = TNG50Dataset(sample_dataset)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (1, 64, 64)
    assert item.dtype == torch.float32


def test_dataset_transform_applied(sample_dataset):
    """Verify transform is called on the tensor."""
    transform = lambda x: x * 2
    ds = TNG50Dataset(sample_dataset, transform=transform)
    raw_ds = TNG50Dataset(sample_dataset)
    torch.testing.assert_close(ds[0], raw_ds[0] * 2)


def test_dataset_metadata_accessible(sample_dataset):
    """Verify metadata DataFrame is accessible."""
    ds = TNG50Dataset(sample_dataset)
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 5
    assert "fits_name" in ds.metadata.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd msd-flow && python -m pytest tests/data/test_dataset.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `TNG50Dataset`**

Create `src/data/dataset.py`:

```python
"""PyTorch Dataset for processed TNG50 galaxy images."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TNG50Dataset(Dataset):
    """Random-access dataset over extracted TNG50 galaxy ``.npy`` files.

    Args:
        processed_dir: Path to directory containing ``metadata.csv`` and
            ``.npy`` image files.
        transform: Optional callable applied to each image tensor.
    """

    def __init__(self, processed_dir: str, transform=None):
        self.processed_dir = processed_dir
        self.transform = transform
        csv_path = os.path.join(processed_dir, "metadata.csv")
        self.metadata = pd.read_csv(csv_path)
        self.filenames = self.metadata["filename"].tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.processed_dir, self.filenames[idx])
        data = np.load(path)
        tensor = torch.from_numpy(data).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd msd-flow && python -m pytest tests/data/test_dataset.py -v`
Expected: All PASS

- [ ] **Step 5: Verify DataLoader compatibility**

Add a quick integration test to `tests/data/test_dataset.py`:

```python
from torch.utils.data import DataLoader


def test_dataset_works_with_dataloader(sample_dataset):
    """Verify dataset integrates with PyTorch DataLoader."""
    ds = TNG50Dataset(sample_dataset)
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch.shape == (2, 1, 64, 64)
```

- [ ] **Step 6: Run all dataset tests**

Run: `cd msd-flow && python -m pytest tests/data/test_dataset.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/data/dataset.py tests/data/test_dataset.py
git commit -m "feat: add TNG50Dataset for PyTorch DataLoader integration"
```

---

### Task 7: Full test suite pass and cleanup

**Files:**
- All modified/created files

- [ ] **Step 1: Run the full test suite**

Run: `cd msd-flow && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify preprocess.py no longer has load_fits**

Run: `cd msd-flow && python -c "from src.data.preprocess import load_fits"` — should raise `ImportError`.

- [ ] **Step 3: Verify imports are clean**

Run: `cd msd-flow && python -c "from src.data.download_tng import fits_name_from_url, load_fits, get_existing_ids, save_metadata, download_batch, extract_batch, cleanup_batch; print('All imports OK')"` — should print "All imports OK".

- [ ] **Step 4: Commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: final cleanup after pipeline implementation"
```
