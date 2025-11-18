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

    @patch('opticalib.dmutils.iff_module._ifa.IFFCapturePreparation')
    @patch('opticalib.dmutils.iff_module._ts')
    @patch('opticalib.dmutils.iff_module._rif')
    @patch('opticalib.dmutils.iff_module._sf')
    def test_iff_data_acquisition_basic(
        self, mock_save_fits, mock_read_config, mock_newtn, mock_iff_prep, 
        mock_dm, mock_interferometer, temp_dir, monkeypatch
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
            "shuffle": 0
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

    @patch('opticalib.dmutils.iff_module._ifa.IFFCapturePreparation')
    @patch('opticalib.dmutils.iff_module._ts')
    @patch('opticalib.dmutils.iff_module._rif')
    def test_iff_data_acquisition_with_modes(
        self, mock_read_config, mock_newtn, mock_iff_prep,
        mock_dm, mock_interferometer, temp_dir, monkeypatch
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
            "shuffle": 0
        }
        mock_iff_prep.return_value = mock_prep
        
        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)
        
        modes = [1, 2, 3, 4, 5]
        amplitude = 0.1
        tn = iff_module.iffDataAcquisition(
            mock_dm, mock_interferometer,
            modesList=modes,
            amplitude=amplitude
        )
        
        assert tn == "20240101_120000"
        # Verify modes and amplitude were passed
        mock_prep.createTimedCmdHistory.assert_called_once()
        call_args = mock_prep.createTimedCmdHistory.call_args
        assert np.array_equal(call_args[0][0], modes) or call_args[0][0] == modes

    @patch('opticalib.dmutils.iff_module._ifa.IFFCapturePreparation')
    @patch('opticalib.dmutils.iff_module._ts')
    @patch('opticalib.dmutils.iff_module._rif')
    def test_iff_data_acquisition_with_shuffle(
        self, mock_read_config, mock_newtn, mock_iff_prep,
        mock_dm, mock_interferometer, temp_dir, monkeypatch
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
            "shuffle": 1
        }
        mock_prep.getInfoToSave.return_value = info_dict
        mock_iff_prep.return_value = mock_prep
        
        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)
        
        tn = iff_module.iffDataAcquisition(
            mock_dm, mock_interferometer,
            shuffle=True
        )
        
        assert tn == "20240101_120000"
        call_args = mock_prep.createTimedCmdHistory.call_args
        assert call_args[0][3] is True

    @patch('opticalib.dmutils.iff_module._ifa.IFFCapturePreparation')
    @patch('opticalib.dmutils.iff_module._ts')
    @patch('opticalib.dmutils.iff_module._rif')
    def test_iff_data_acquisition_differential(
        self, mock_read_config, mock_newtn, mock_iff_prep,
        mock_dm, mock_interferometer, temp_dir, monkeypatch
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
            "shuffle": 0
        }
        mock_iff_prep.return_value = mock_prep
        
        iff_folder = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_folder, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)
        
        tn = iff_module.iffDataAcquisition(
            mock_dm, mock_interferometer,
            differential=True
        )
        
        assert tn == "20240101_120000"
        # Verify differential was passed to runCmdHistory
        mock_dm.runCmdHistory.assert_called_once()
        call_args = mock_dm.runCmdHistory.call_args
        assert call_args[1]['differential'] is True

