"""Tests for src.data.download_tng."""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.data.download_tng import download_tng_fits_file, extract_tng_urls


# ------------------------------ extract_tng_urls ----------------------------- #


@patch("src.data.download_tng.requests.get")
def test_extract_tng_urls_returns_list(mock_get):
    """Verify URL list is returned from a successful API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["url1", "url2", "url3"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0], [72], headers={"api-key": "test"})
    assert urls == ["url1", "url2", "url3"]


@patch("src.data.download_tng.requests.get")
def test_extract_tng_urls_n_limits_results(mock_get):
    """Verify N parameter limits the number of URLs per combination."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["a", "b", "c", "d"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0], [72], headers={"api-key": "test"}, N=2)
    assert urls == ["a", "b"]


@patch("src.data.download_tng.requests.get")
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


@patch("src.data.download_tng.requests.get")
def test_extract_tng_urls_multiple_combos(mock_get):
    """Verify URLs are collected across all version-snapshot combinations."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"files": ["url1"]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    urls = extract_tng_urls([0, 1], [72, 73], headers={"api-key": "test"})
    assert len(urls) == 4  # 2 versions x 2 snapshots x 1 url each


# --------------------------- download_tng_fits_file -------------------------- #


def test_download_skips_existing_file(tmp_path):
    """Verify download is skipped when the file already exists."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"
    # Pre-create the expected file
    expected_name = "snap_skirt_images_hsc_idealized_v0_72_subhalo_12345_image.fits"
    (tmp_path / expected_name).touch()

    result = download_tng_fits_file(url, str(tmp_path), headers={"api-key": "test"})
    assert "Already exists" in result


@patch("src.data.download_tng.requests.get")
def test_download_success(mock_get, tmp_path):
    """Verify a successful download writes the file and returns success."""
    url = "http://www.tng-project.org/api/TNG50-1/files/skirt_images_hsc_idealized_v0_72/subhalos/12345/image.fits"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [b"fake fits data"]
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_response

    result = download_tng_fits_file(url, str(tmp_path), headers={"api-key": "test"})
    assert "Successfully downloaded" in result
