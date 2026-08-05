"""
Tests for opticalib.dmutils.iff_preparation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from opticalib.dmutils import iff_preparation as ifa
from opticalib.core.exceptions import DeviceError


class TestIFFCapturePreparation:
    """Test IFFCapturePreparation class."""

    def test_init(self, mock_dm):
        """Test IFFCapturePreparation initialization."""
        prep = ifa.IFFCapturePreparation(mock_dm)

        assert prep._NActs == mock_dm.n_acts
        assert prep.mirrorModes is not None
        assert prep._modalBase is not None

    def test_init_invalid_device(self):
        """Test initialization with invalid device."""
        invalid_dm = "not_a_dm"

        with pytest.raises(DeviceError):
            ifa.IFFCapturePreparation(invalid_dm)

    def _iff_config(self):
        return {
            "timing": 2,
            "TRIGGER": {
                "trailing_zeros": 1,
                "modes_list": [10],
                "amplitude": 0.1,
                "template": [1],
                "modal_base": "mirror",
            },
            "REGISTRATION": {
                "trailing_zeros": 0,
                "modes_list": [1, 2],
                "amplitude": 0.1,
                "template": [1, -1],
                "modal_base": "zonal",
            },
            "IFFUNC": {
                "trailing_zeros": 0,
                "padding_zeros": 2,
                "modes_list": np.arange(20),
                "amplitude": 0.1,
                "template": [1, -1],
                "shuffle": False,
                "n_repetitions": 1,
                "modal_base": "hadamard",
            },
        }

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_timed_cmd_history_basic(self, mock_get_iff_config, mock_dm):
        """Test creating timed command history."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        tch = prep.create_timed_cmd_history()
        assert tch is not None
        assert isinstance(tch, np.ndarray)
        assert prep.timedCmdHistory is not None

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_timed_cmd_history_with_modes(self, mock_get_iff_config, mock_dm):
        """Test creating timed command history with custom modes."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        tch = prep.create_timed_cmd_history(modesList=[1, 2, 3, 4, 5])
        assert tch is not None
        assert prep._modesList is not None

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_timed_cmd_history_with_shuffle(self, mock_get_iff_config, mock_dm):
        """Test creating timed command history with shuffle."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        modes = np.arange(mock_dm.n_acts)
        tch = prep.create_timed_cmd_history(modesList=modes, shuffle=True)
        assert tch is not None
        assert prep._shuffle is True

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_get_info_to_save(self, mock_get_iff_config, mock_dm):
        """Test getting info to save."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep.create_timed_cmd_history()
        info = prep.get_info_to_save()
        assert isinstance(info, dict)
        assert "timed_cmd_history" in info
        assert "cmd_matrix" in info
        assert "modes_list" in info
        assert "template" in info
        assert "shuffle" in info

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_cmd_matrix_history(self, mock_get_iff_config, mock_dm):
        """Test creating command matrix history."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        cmd_hist = prep.create_cmd_matrix_history()
        assert cmd_hist is not None
        assert isinstance(cmd_hist, np.ndarray)
        assert prep.cmdMatHistory is not None

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_aux_cmd_history(self, mock_get_iff_config, mock_dm):
        """Test creating auxiliary command history."""
        mock_get_iff_config.return_value = self._iff_config()
        prep = ifa.IFFCapturePreparation(mock_dm)
        aux_hist = prep.create_aux_cmd_history()
        assert aux_hist is not None

    def test_create_zonal_mat(self, mock_dm):
        """Test creating zonal matrix."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        zonal = prep._create_zonal_mat()

        assert zonal is not None
        assert zonal.shape == (mock_dm.n_acts, mock_dm.n_acts)
        # Zonal should be identity matrix
        np.testing.assert_array_equal(zonal, np.eye(mock_dm.n_acts))

    def test_create_hadamard_mat(self, mock_dm):
        """Test creating Hadamard matrix."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        hadamard = prep._create_hadamard_mat()

        assert hadamard is not None
        assert hadamard.shape[0] == mock_dm.n_acts

    @patch("opticalib.dmutils.iff_preparation._osu.load_fits")
    def test_create_user_mat(self, mock_load_fits, mock_dm, temp_dir, monkeypatch):
        """Test creating user-defined modal base."""
        from opticalib.core.root import MODALBASE_ROOT_FOLDER
        import os

        modal_folder = os.path.join(temp_dir, "ModalBases")
        os.makedirs(modal_folder, exist_ok=True)
        monkeypatch.setattr("opticalib.core.root.MODALBASE_ROOT_FOLDER", modal_folder)

        # Create a test modal base file
        test_modal = np.random.randn(100, 50).astype(np.float32)
        from opticalib.ground import osutils

        modal_file = os.path.join(modal_folder, "test_modal.fits")
        osutils.save_fits(modal_file, test_modal, overwrite=True)

        # Mock load_fits to return the actual data
        mock_load_fits.return_value = test_modal

        prep = ifa.IFFCapturePreparation(mock_dm)
        user_mat = prep._create_user_mat("test_modal.fits")

        assert user_mat is not None
        assert user_mat.shape == (100, 50)

    def test_update_modal_base_mirror(self, mock_dm):
        """Test updating modal base to mirror."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._update_modal_base("mirror")

        assert prep.modalBaseId == "mirror"
        np.testing.assert_array_equal(prep._modalBase, mock_dm.mirrorModes)

    def test_update_modal_base_zonal(self, mock_dm):
        """Test updating modal base to zonal."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._update_modal_base("zonal")

        assert prep.modalBaseId == "zonal"
        assert prep._modalBase.shape == (mock_dm.n_acts, mock_dm.n_acts)

    def test_update_modal_base_hadamard(self, mock_dm):
        """Test updating modal base to Hadamard."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._update_modal_base("hadamard")

        assert prep.modalBaseId == "hadamard"
        assert prep._modalBase.shape[0] == mock_dm.n_acts

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_create_cmd_matrix_history_invalid_n_repetitions(
        self, mock_get_iff_config, mock_dm
    ):
        """Test that create_cmd_matrix_history raises ValueError for invalid n_repetitions."""
        mock_get_iff_config.return_value = self._iff_config()

        prep = ifa.IFFCapturePreparation(mock_dm)

        # Test n_repetitions = 0
        with pytest.raises(ValueError, match="n_repetitions must be >= 1"):
            prep.create_cmd_matrix_history(modesList=np.arange(5), n_repetitions=0)

        # Test n_repetitions = -1
        with pytest.raises(ValueError, match="n_repetitions must be >= 1"):
            prep.create_cmd_matrix_history(modesList=np.arange(5), n_repetitions=-1)

        # Test n_repetitions = -10
        with pytest.raises(ValueError, match="n_repetitions must be >= 1"):
            prep.create_cmd_matrix_history(modesList=np.arange(5), n_repetitions=-10)
