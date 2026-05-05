"""Tests for simulator static data resolver and downloader helpers."""

import os

import pytest

from opticalib.simulator._API import simdata


class TestSimDataHelpers:
    """Test suite for simulator data helper functions."""

    def test_available_files_include_m4_data(self):
        """Known data files should include the M4 heavy dataset."""
        assert "m4_data.h5" in simdata.available_simdata_files()

    def test_get_simdata_file_resolves_cached_file(self, monkeypatch, tmp_path):
        """Cached simulator files should resolve without download."""
        monkeypatch.setattr(simdata._root, "CONFIGURATION_FOLDER", str(tmp_path))
        monkeypatch.setattr(simdata, "_validate_if_known", lambda *args: None)
        cache_dir = tmp_path / "SimData"
        cache_dir.mkdir(parents=True)
        cached_file = cache_dir / "dp_cmdmat.fits"
        cached_file.write_bytes(b"test")

        path = simdata.get_simdata_file("dp_cmdmat.fits", auto_download=False)

        assert os.path.exists(path)
        assert path.endswith("dp_cmdmat.fits")

    def test_missing_file_without_download_raises(self, monkeypatch, tmp_path):
        """Missing files should raise a clear error when download is disabled."""
        monkeypatch.setattr(simdata._root, "CONFIGURATION_FOLDER", str(tmp_path))
        with pytest.raises(FileNotFoundError) as exc:
            simdata.get_simdata_file("m4_data.h5", auto_download=False)

        assert "OPTICALIB_SIMDATA_BASE_URL" in str(exc.value)

    def test_base_url_can_be_overridden(self, monkeypatch):
        """Environment variable should override default base URL."""
        monkeypatch.setenv("OPTICALIB_SIMDATA_BASE_URL", "https://example.org/sim")

        url = simdata._resolve_download_url("m4_data.h5")

        assert url == "https://example.org/sim/m4_data.h5"
