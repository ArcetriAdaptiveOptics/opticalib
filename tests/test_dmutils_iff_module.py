"""
Tests for opticalib.dmutils.iff_module module.
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from opticalib.dmutils import iff_module


class TestIffDataAcquisition:
    """Test iffDataAcquisition function."""

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    @patch("opticalib.dmutils.iff_module._osu.save_fits")
    def test_iff_data_acquisition_basic(
        self,
        mock_save_fits,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test basic IFF data acquisition."""
        from opticalib.core.root import folders

        # Setup mocks
        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 10),
            "modesVector": np.array([1, 2, 3]),
            "regActs": np.array([1, 2]),
            "ampVector": np.array([0.1, 0.2, 0.3]),
            "indexList": np.array([0, 1, 2]),
            "template": np.array([1, -1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        # Mock folder paths
        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        # Run function
        tn = iff_module.iffDataAcquisition(mock_dm, mock_interferometer)

        # Verify
        assert tn == "20240101_120000"
        mock_prep.createTimedCmdHistory.assert_called_once()
        mock_dm.uploadCmdHistory.assert_called_once()
        mock_dm.runCmdHistory.assert_called_once()

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    def test_iff_data_acquisition_with_modes(
        self,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test IFF data acquisition with custom modes."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 5),
            "modesVector": np.array([1, 2, 3, 4, 5]),
            "regActs": np.array([]),
            "ampVector": np.array([0.1] * 5),
            "indexList": np.array([0, 1, 2, 3, 4]),
            "template": np.array([1, -1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        modes = [1, 2, 3, 4, 5]
        amplitude = 0.1
        tn = iff_module.iffDataAcquisition(
            mock_dm, mock_interferometer, modesList=modes, amplitude=amplitude
        )

        assert tn == "20240101_120000"
        # Verify modes and amplitude were passed
        mock_prep.createTimedCmdHistory.assert_called_once()
        call_args = mock_prep.createTimedCmdHistory.call_args
        assert np.array_equal(call_args.kwargs['modesList'], modes) or call_args.kwargs['modesList'] == modes

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    def test_iff_data_acquisition_with_shuffle(
        self,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test IFF data acquisition with shuffle enabled."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        info_dict = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 10),
            "modesVector": np.array([1, 2, 3]),
            "regActs": np.array([]),
            "ampVector": np.array([0.1, 0.2, 0.3]),
            "indexList": np.array([0, 1, 2]),
            "template": np.array([1, -1]),
            "shuffle": 1,
        }
        mock_prep.getInfoToSave.return_value = info_dict
        mock_iff_prep.return_value = mock_prep

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        tn = iff_module.iffDataAcquisition(mock_dm, mock_interferometer, shuffle=True)

        assert tn == "20240101_120000"
        call_args = mock_prep.createTimedCmdHistory.call_args
        assert call_args.kwargs.get('shuffle', False) is True

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    def test_iff_data_acquisition_differential(
        self,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test IFF data acquisition with differential mode."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 10),
            "modesVector": np.array([1, 2, 3]),
            "regActs": np.array([]),
            "ampVector": np.array([0.1, 0.2, 0.3]),
            "indexList": np.array([0, 1, 2]),
            "template": np.array([1, -1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        tn = iff_module.iffDataAcquisition(
            mock_dm, mock_interferometer, differential=True
        )

        assert tn == "20240101_120000"
        # Verify differential was passed to runCmdHistory
        mock_dm.runCmdHistory.assert_called_once()
        call_args = mock_dm.runCmdHistory.call_args
        assert call_args[1]["differential"] is True


class TestAcquirePistonData:
    """Test acquirePistonData function."""

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    @patch("opticalib.dmutils.iff_module._osu.save_fits")
    def test_acquire_piston_data_basic(
        self,
        mock_save_fits,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test basic piston data acquisition."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.cmdMatHistory = np.random.randn(100, 50).astype(np.float32)
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 50),
            "modesVector": np.arange(50),
            "regActs": np.array([]),
            "ampVector": np.random.randn(50).astype(np.float32),
            "indexList": np.arange(50),
            "template": np.array([1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        template = [1, -1, 1]
        tn = iff_module.acquirePistonData(
            mock_dm, mock_interferometer, template=template, nstep=10, stepamp=70e-9
        )

        assert tn == "20240101_120000"
        mock_prep.createTimedCmdHistory.assert_called_once()
        mock_dm.uploadCmdHistory.assert_called_once()
        mock_dm.runCmdHistory.assert_called_once()

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    @patch("opticalib.dmutils.iff_module._osu.save_fits")
    def test_acquire_piston_data_with_buffer(
        self,
        mock_save_fits,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test piston data acquisition with buffer reading."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.cmdMatHistory = np.random.randn(100, 50).astype(np.float32)
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 50),
            "modesVector": np.arange(50),
            "regActs": np.array([]),
            "ampVector": np.random.randn(50).astype(np.float32),
            "indexList": np.arange(50),
            "template": np.array([1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        # Mock buffer context manager
        mock_buffer_data = {"actuator_1": np.random.randn(100)}
        mock_dm.read_buffer = MagicMock()
        mock_dm.read_buffer.return_value.__enter__ = MagicMock(
            return_value=mock_buffer_data
        )
        mock_dm.read_buffer.return_value.__exit__ = MagicMock(return_value=None)

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        template = [1, -1, 1]
        tn = iff_module.acquirePistonData(
            mock_dm,
            mock_interferometer,
            template=template,
            nstep=10,
            stepamp=70e-9,
            read_buffer=True,
        )

        assert tn == "20240101_120000"
        mock_dm.read_buffer.assert_called_once()
        mock_save_fits.assert_called()

    @patch("opticalib.dmutils.iff_module._ifa.IFFCapturePreparation")
    @patch("opticalib.dmutils.iff_module._osu.newtn")
    @patch("opticalib.dmutils.iff_module._rif")
    def test_acquire_piston_data_reverse(
        self,
        mock_read_config,
        mock_newtn,
        mock_iff_prep,
        mock_dm,
        mock_interferometer,
        temp_dir,
        monkeypatch,
    ):
        """Test piston data acquisition with reverse option."""
        from opticalib.core.root import folders

        mock_newtn.return_value = "20240101_120000"
        mock_prep = MagicMock()
        mock_prep.cmdMatHistory = np.random.randn(100, 50).astype(np.float32)
        mock_prep.createTimedCmdHistory.return_value = np.random.randn(100, 50)
        mock_prep.getInfoToSave.return_value = {
            "timedCmdHistory": np.random.randn(100, 50),
            "cmdMatrix": np.random.randn(100, 50),
            "modesVector": np.arange(50),
            "regActs": np.array([]),
            "ampVector": np.random.randn(50).astype(np.float32),
            "indexList": np.arange(50),
            "template": np.array([1]),
            "shuffle": 0,
        }
        mock_iff_prep.return_value = mock_prep

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        template = [1, -1, 1]
        tn = iff_module.acquirePistonData(
            mock_dm,
            mock_interferometer,
            template=template,
            nstep=5,
            stepamp=70e-9,
            reverse=True,
        )

        assert tn == "20240101_120000"


class TestSaveBufferData:
    """Test saveBufferData function."""

    @patch("opticalib.dmutils.iff_module._osu.is_tn")
    @patch("opticalib.dmutils.iff_module._osu.save_dict")
    def test_save_buffer_data_with_tn(
        self, mock_save_dict, mock_is_tn, mock_dm, temp_dir, monkeypatch
    ):
        """Test saving buffer data with tracking number."""
        from opticalib.core.root import folders

        mock_is_tn.return_value = True
        mock_dm.bufferData = {
            "actuator_1": np.random.randn(100).astype(np.float32),
            "actuator_2": np.random.randn(100).astype(np.float32),
        }

        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)

        tn = "20240101_120000"
        iff_module.saveBufferData(mock_dm, tn)

        # Verify save_dict was called with correct path
        mock_save_dict.assert_called_once()
        call_args = mock_save_dict.call_args
        assert tn in call_args[0][1]  # Check path contains tn
        assert call_args[0][0] == mock_dm.bufferData  # Check data

    @patch("opticalib.dmutils.iff_module._osu.is_tn")
    @patch("opticalib.dmutils.iff_module._osu.save_dict")
    def test_save_buffer_data_with_path(
        self, mock_save_dict, mock_is_tn, mock_dm, temp_dir
    ):
        """Test saving buffer data with full path."""
        mock_is_tn.return_value = False
        mock_dm.bufferData = {
            "actuator_1": np.random.randn(100).astype(np.float32),
        }

        # Use directory path since the code checks os.path.exists
        # and save_dict will append .h5 to create the file
        filepath = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        iff_module.saveBufferData(mock_dm, filepath)

        mock_save_dict.assert_called_once()
        call_args = mock_save_dict.call_args
        expected_path = os.path.join(filepath, 'buffer_data.h5')
        assert call_args[0][1] == expected_path

    def test_save_buffer_data_no_read_buffer(self):
        """Test that BufferError is raised when DM has no read_buffer."""
        from opticalib.core import exceptions as _oe

        # Create a mock DM without read_buffer attribute
        # Use a simple Mock instead of MagicMock to avoid hasattr always returning True
        mock_dm = Mock(spec=[])  # Empty spec means no attributes by default
        mock_dm.bufferData = {"data": np.random.randn(10)}

        with pytest.raises(_oe.BufferError):
            iff_module.saveBufferData(mock_dm, "20240101_120000")

    @patch("opticalib.dmutils.iff_module._osu.is_tn")
    def test_save_buffer_data_invalid_path(self, mock_is_tn, mock_dm):
        """Test that PathError is raised for invalid path."""
        from opticalib.core import exceptions as _oe

        mock_is_tn.return_value = False
        mock_dm.bufferData = {"data": np.random.randn(10)}

        with pytest.raises(_oe.PathError):
            iff_module.saveBufferData(mock_dm, "/nonexistent/path")


class TestPrepareSteppingAmplitudes:
    """Test _prepareSteppingAmplitudes function."""

    def test_prepare_stepping_amplitudes_basic(self):
        """Test basic stepping amplitude preparation."""
        template = [1, -1, 1]
        nstep = 5
        stepamp = 70e-9

        result = iff_module._prepareSteppingAmplitudes(template, nstep, stepamp)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        # Check that amplitudes are positive (after removing initial 0)
        assert np.all(result >= 0)

    def test_prepare_stepping_amplitudes_reverse(self):
        """Test stepping amplitude preparation with reverse."""
        template = [1, -1, 1]
        nstep = 3
        stepamp = 70e-9

        result = iff_module._prepareSteppingAmplitudes(
            template, nstep, stepamp, reverse=True
        )

        assert isinstance(result, np.ndarray)
        # With reverse, the sequence should be longer
        assert len(result) > nstep * len(template)

    def test_prepare_stepping_amplitudes_even_template_error(self):
        """Test that ValueError is raised for even-length template."""
        template = [1, -1]  # Even length
        nstep = 5
        stepamp = 70e-9

        with pytest.raises(ValueError, match="Template must return to starting point"):
            iff_module._prepareSteppingAmplitudes(template, nstep, stepamp)

    def test_prepare_stepping_amplitudes_custom_stepamp(self):
        """Test stepping amplitude preparation with custom step amplitude."""
        template = [1, -1, 1]
        nstep = 4
        stepamp = 50e-9

        result = iff_module._prepareSteppingAmplitudes(template, nstep, stepamp)

        assert isinstance(result, np.ndarray)
        # Check that max amplitude is approximately nstep * stepamp
        assert np.max(result) <= (nstep * stepamp * 1.1)  # Allow small tolerance
