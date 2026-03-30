"""Tests for msdflow.data.download_tng."""

import os

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from astropy.io import fits as astropy_fits

from msdflow.data.download_tng import (
    cleanup_batch,
    download_batch,
    download_tng_fits_file,
    extract_batch,
    extract_tng_urls,
    fits_name_from_url,
    get_existing_ids,
    load_fits,
    save_metadata,
)


# ------------------------------ extract_tng_urls ----------------------------- #


@patch("msdflow.data.download_tng.requests.get")
def test_extract_tng_urls_returns_list(mock_get):
    """Verify URL list is returned from a successful API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["url1", "url2", "url3"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0], [72], headers={"api-key": "test"})
    assert urls == ["url1", "url2", "url3"]


@patch("msdflow.data.download_tng.requests.get")
def test_extract_tng_urls_n_limits_results(mock_get):
    """Verify N parameter limits the number of URLs per combination."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["a", "b", "c", "d"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0], [72], headers={"api-key": "test"}, N=2)
    assert urls == ["a", "b"]


@patch("msdflow.data.download_tng.requests.get")
def test_extract_tng_urls_http_error_skipped(mock_get):
    """Verify HTTP errors are logged and skipped without raising."""
    import requests as req

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = req.exceptions.HTTPError(
        response=MagicMock(status_code=404)
    )
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0], [72], headers={"api-key": "test"})
    assert urls == []


@patch("msdflow.data.download_tng.requests.get")
def test_extract_tng_urls_multiple_combos(mock_get):
    """Verify URLs are collected across all version-snapshot combinations."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["url1"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0, 1], [72, 73], headers={"api-key": "test"})
    assert len(urls) == 4  # 2 versions x 2 snapshots x 1 url each


# ------------------------------ fits_name_from_url ----------------------------- #


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


# --------------------------- download_tng_fits_file -------------------------- #


def test_download_skips_existing_file(tmp_path):
    """Verify download returns path when the file already exists."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"
    expected_name = "snap_skirt_images_hsc_idealized_v0_72_subhalo_12345_image.fits"
    (tmp_path / expected_name).touch()

    result = download_tng_fits_file(url, str(tmp_path), headers={"api-key": "test"})
    assert result == str(tmp_path / expected_name)


@patch("msdflow.data.download_tng.requests.get")
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


# --------------------------------- load_fits -------------------------------- #


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
    assert header["TESTKEY"] == "value_g"


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


# ----------------------------- get_existing_ids ----------------------------- #


def test_get_existing_ids_empty_dir(tmp_path):
    """Verify empty set returned when no metadata.csv exists."""
    assert get_existing_ids(str(tmp_path)) == set()


def test_get_existing_ids_reads_fits_name_column(tmp_path):
    """Verify fits_name values are returned from existing metadata."""
    df = pd.DataFrame({"filename": ["g_00000.npy"], "fits_name": ["snap_v0_72_sub_1_img"]})
    df.to_csv(tmp_path / "metadata.csv", index=False)
    result = get_existing_ids(str(tmp_path))
    assert result == {"snap_v0_72_sub_1_img"}


# ------------------------------- save_metadata ------------------------------ #


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


# ------------------------------ download_batch ------------------------------ #


@patch("msdflow.data.download_tng.download_tng_fits_file")
def test_download_batch_filters_failures(mock_download, tmp_path):
    """Verify download_batch returns only successful paths."""
    mock_download.side_effect = [str(tmp_path / "a.fits"), None, str(tmp_path / "c.fits")]
    urls = ["url1", "url2", "url3"]
    result = download_batch(urls, str(tmp_path), {"api-key": "test"}, max_workers=2)
    assert len(result) == 2


# ------------------------------ extract_batch ------------------------------- #


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


# ------------------------------ cleanup_batch ------------------------------- #


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
