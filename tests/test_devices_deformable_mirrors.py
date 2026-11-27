"""Tests for deformable mirror device interfaces."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opticalib.devices.deformable_mirrors import DP


class TestDP:
    """Test the deformable platform interface."""

    @patch("opticalib.devices.deformable_mirrors._rc.get_iff_config", return_value=MagicMock())
    def test_read_buffer_organizes_data_by_diagnostic_key(self, mock_getiff):
        """Test that each buffer key contains a sample-by-actuator matrix."""
        sample_count = 4
        diagnostic_count = 17
        actuator_count = 111
        mock_getiff.return_value.get.return_value = {"frequency": 1.0}

        dm = DP.__new__(DP)
        dm._cmd_history = np.zeros((actuator_count, 2))
        dm._logger = MagicMock()
        dm._aoClient = MagicMock()

        subsystem = dm._aoClient.aoSystem.aoSubSystem0
        subsystem.sysConf.gen.cntFreq = 1000.0
        subsystem.support.diagBuf.read.return_value = {
            f"ch{actuator:04d}": (
                np.arange(sample_count * diagnostic_count).reshape(
                    sample_count, diagnostic_count
                )
                + actuator * 1000
            )
            for actuator in range(actuator_count)
        }

        with dm.read_buffer(segment=0, npoints_per_cmd=10) as result:
            assert result == {}

        assert len(result) == diagnostic_count
        assert result["actPos"].shape == (sample_count, actuator_count)
        np.testing.assert_array_equal(
            result["actPos"][:, 0],
            subsystem.support.diagBuf.read.return_value["ch0000"][:, 4],
        )
        np.testing.assert_array_equal(
            result["actPos"][:, -1],
            subsystem.support.diagBuf.read.return_value["ch0110"][:, 4],
        )
        np.testing.assert_array_equal(dm.bufferData["actPos"], result["actPos"])

class TestNacoDmConstruction:
    """Regression: NacoDm used to crash on construction (read-only property)."""

    @pytest.fixture
    def naco(self, monkeypatch):
        from opticalib.core.exceptions import DeviceNotFoundError
        from opticalib.devices.deformable_mirrors import NacoDm

        def _raise(*a, **k):
            raise DeviceNotFoundError("no config")

        monkeypatch.setattr("opticalib.core.config.get_device_config", _raise)
        # Patch through NacoDm's own base: other tests reload opticalib modules,
        # so a freshly imported NACOMirror may not be the class NacoDm inherits.
        monkeypatch.setattr(
            NacoDm.__bases__[0], "_mirror_command", lambda self, cmd: None
        )
        return NacoDm(host="127.0.0.1", port=45678)

    def test_construction(self, naco):
        assert naco.cmd_history is None
        assert naco.act_coord.shape == (185, 2)
        assert naco.n_segments == 1
        assert naco.n_acts_per_segment == 185
        assert naco.slave_ids == [] and naco.border_ids == []

    def test_upload_cmd_history(self, naco):
        mat = np.random.default_rng(0).normal(size=(185, 3))
        naco.upload_cmd_history(mat)
        np.testing.assert_array_equal(naco.cmd_history, mat)
