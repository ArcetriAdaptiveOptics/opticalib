"""Tests for opticalib.devices.interferometer static helper methods."""

from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma

from opticalib.devices.interferometer import _4DInterferometer


class Test4DInterferometerStatics:
    """Tests for static methods in _4DInterferometer."""

    def test_get_camera_settings_uses_new_width_method(self, monkeypatch):
        """Test camera settings retrieval with the renamed width accessor."""

        class ReaderStub:
            """Reader stub exposing only the new width API."""

            def get_image_width_in_pixels(self):
                return 2000

            def get_image_height_in_pixels(self):
                return 1500

            def get_offset_x(self):
                return 11

            def get_offset_y(self):
                return 22

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )

        settings = _4DInterferometer.get_camera_settings()

        assert settings == [2000, 1500, 11, 22]

    def test_get_frame_rate_reads_from_reader(self, monkeypatch):
        """Test frame-rate retrieval through the configured reader."""

        class ReaderStub:
            """Reader stub for frame-rate retrieval."""

            def get_frame_rate(self):
                return 77.5

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )

        frame_rate = _4DInterferometer.get_frame_rate()

        assert frame_rate == 77.5

    def test_into_full_frame_uses_camera_offsets_when_offset_is_none(
        self, monkeypatch
    ):
        """Test that into_full_frame works with default optional offset."""
        monkeypatch.setattr(
            "opticalib.devices.interferometer._4DInterferometer.get_camera_settings",
            staticmethod(lambda _tn=None: [2048, 2048, 4, 7]),
        )

        img = ma.masked_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

        fullimg = _4DInterferometer.into_full_frame(img)

        assert fullimg.shape == (2048, 2048)
        np.testing.assert_array_equal(fullimg[7:9, 4:6], img.data)

    def test_into_full_frame_offset_argument_overrides_settings(self, monkeypatch):
        """Test that explicit offsets have precedence over configuration offsets."""
        monkeypatch.setattr(
            "opticalib.devices.interferometer._4DInterferometer.get_camera_settings",
            staticmethod(lambda _tn=None: [2048, 2048, 100, 100]),
        )

        img = ma.masked_array(np.array([[9.0]]))

        fullimg = _4DInterferometer.into_full_frame(img, offset=(0, 0))

        assert fullimg[0, 0] == 9.0

    def test_into_full_frame_reads_offsets_from_config_path(self, monkeypatch):
        """Test config_path-based offset loading for full-frame insertion."""
        reader = MagicMock()
        reader.get_offset_x.return_value = 5
        reader.get_offset_y.return_value = 6
        monkeypatch.setattr(
            "opticalib.devices.interferometer._fn.ConfSettingReader4D",
            lambda _path: reader,
        )

        img = ma.masked_array(np.array([[3.0, 1.0]]))

        fullimg = _4DInterferometer.into_full_frame(
            img,
            config_path="/tmp/fake_4d_settings.ini",
            offset=(0, 0),
        )

        np.testing.assert_array_equal(fullimg[0:1, 0:2], img.data)
