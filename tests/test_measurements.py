"""
Tests for opticalib.measurements module.
"""

import pytest
import os
import numpy as np
import numpy.ma as ma
from unittest.mock import MagicMock, Mock, patch
from opticalib.measurements import Measurements


class _FakeInterferometer:
    """Minimal interferometer satisfying the InterferometerDevice protocol."""

    def acquire_map(self, nframes=1, delay=0, rebin=1):
        """Return a fake masked image."""
        data = np.random.randn(50, 50).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        return ma.masked_array(data, mask=mask)

    def acquireFullFrame(self, **kwargs):
        """Return a fake full-frame image."""
        data = np.random.randn(50, 50).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        return ma.masked_array(data, mask=mask)

    def capture(self, numberOfFrames=1, folder_name=None):
        """Return a fake tracking number."""
        return "20240101_120000"

    def produce(self, tn):
        """Produce data for a tracking number."""
        pass


class _FakeCamera:
    """Minimal camera satisfying the CameraDevice protocol."""

    def acquire_frames(self, **kwargs):
        """Return a fake image."""
        data = np.random.randn(50, 50).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        return ma.masked_array(data, mask=mask)

    def set_exptime(self, exptime):
        """Set the exposure time."""
        self._exptime = exptime

    def get_exptime(self):
        """Get the exposure time."""
        return getattr(self, "_exptime", 1.0)


class TestMeasurementsInit:
    """Test Measurements class initialization."""

    def test_init_with_interferometer(self):
        """Test initialization with an interferometer device."""
        interf = _FakeInterferometer()
        m = Measurements(interf)

        assert m._camera is interf

    def test_init_with_camera(self):
        """Test initialization with a camera device."""
        cam = _FakeCamera()
        m = Measurements(cam)

        assert m._camera is cam

    def test_init_with_devices(self):
        """Test initialization with additional devices."""
        interf = _FakeInterferometer()
        device = MagicMock()
        m = Measurements(interf, devices=device)

        assert m._camera is interf
        assert m._devices is device

    def test_init_with_unsupported_device_raises(self):
        """Test that unsupported camera type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported camera device type"):
            Measurements(object())

    def test_acquire_func_set_for_interferometer(self):
        """Test that acquire function is set correctly for interferometer."""
        interf = _FakeInterferometer()
        m = Measurements(interf)

        assert m._acquire_func == interf.acquire_map

    def test_acquire_func_set_for_camera(self):
        """Test that acquire function is set correctly for camera."""
        cam = _FakeCamera()
        m = Measurements(cam)

        assert m._acquire_func == cam.acquire_frames


class TestAcquireTimeSeries:
    """Test Measurements.acquire_time_series method."""

    def test_acquire_time_series_returns_tn(self, temp_dir, monkeypatch):
        """Test that acquire_time_series returns a tracking number string."""
        interf = _FakeInterferometer()
        m = Measurements(interf)

        # Patch save_fits and create_data_folder to avoid actual file I/O
        opd_folder = os.path.join(temp_dir, "OPDSeries")
        data_subfolder = os.path.join(opd_folder, "20260403_080000")
        os.makedirs(data_subfolder, exist_ok=True)

        monkeypatch.setattr("opticalib.measurements._cdf", lambda path: data_subfolder)
        monkeypatch.setattr("opticalib.measurements._save_fits", lambda path, data: None)

        result = m.acquire_time_series(nframes=2)

        assert isinstance(result, str)

    def test_acquire_time_series_calls_acquire_map(self, temp_dir, monkeypatch):
        """Test acquire_time_series calls acquire_map nframes times."""
        interf = _FakeInterferometer()
        call_count = 0
        original_acquire = interf.acquire_map

        def counting_acquire(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_acquire(*args, **kwargs)

        interf.acquire_map = counting_acquire
        m = Measurements(interf)

        data_subfolder = os.path.join(temp_dir, "sub")
        os.makedirs(data_subfolder, exist_ok=True)
        monkeypatch.setattr("opticalib.measurements._cdf", lambda path: data_subfolder)
        monkeypatch.setattr("opticalib.measurements._save_fits", lambda path, data: None)

        nframes = 3
        m.acquire_time_series(nframes=nframes)

        assert call_count == nframes

    def test_acquire_time_series_calls_acquire_frames(self, temp_dir, monkeypatch):
        """Test acquire_time_series calls acquire_frames nframes times for camera."""
        cam = _FakeCamera()
        call_count = 0
        original_acquire = cam.acquire_frames

        def counting_acquire(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_acquire(*args, **kwargs)

        cam.acquire_frames = counting_acquire
        m = Measurements(cam)

        data_subfolder = os.path.join(temp_dir, "sub")
        os.makedirs(data_subfolder, exist_ok=True)
        monkeypatch.setattr("opticalib.measurements._cdf", lambda path: data_subfolder)
        monkeypatch.setattr("opticalib.measurements._save_fits", lambda path, data: None)

        nframes = 2
        m.acquire_time_series(nframes=nframes)

        assert call_count == nframes

