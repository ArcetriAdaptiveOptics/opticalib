"""Tests for deformable mirror device interfaces."""

from unittest.mock import MagicMock, patch

import numpy as np

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
        dm.cmdHistory = np.zeros((actuator_count, 2))
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