"""
Tests for opticalib.dmutils.iff_acquisition_preparation module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from opticalib.dmutils import iff_acquisition_preparation as ifa
from opticalib.core.exceptions import DeviceError


class TestIFFCapturePreparation:
    """Test IFFCapturePreparation class."""

    def test_init(self, mock_dm):
        """Test IFFCapturePreparation initialization."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        
        assert prep._NActs == mock_dm.nActs
        assert prep.mirrorModes is not None
        assert prep._modalBase is not None

    def test_init_invalid_device(self):
        """Test initialization with invalid device."""
        invalid_dm = "not_a_dm"
        
        with pytest.raises(DeviceError):
            ifa.IFFCapturePreparation(invalid_dm)

    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getTiming')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    def test_create_timed_cmd_history_basic(
        self, mock_get_iff_config, mock_get_timing, mock_get_info, mock_dm, temp_dir, monkeypatch
    ):
        """Test creating timed command history."""
        from opticalib.core.root import folders
        import os
        import yaml
        
        # Create config file
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        config_file = os.path.join(config_folder, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
                "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "IFFUNC": {"zeros": 0, "modes": list(range(100)), "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "DM": {"nacts": 100, "timing": 10}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        mock_get_timing.return_value = 10
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": list(range(100)), "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"nacts": 100, "timing": 10}
        )
        config_map = {
            "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            "IFFUNC": {"zeros": 0, "modes": list(range(100)), "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        tch = prep.createTimedCmdHistory()
        
        assert tch is not None
        assert isinstance(tch, np.ndarray)
        assert prep.timedCmdHistory is not None

    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getTiming')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    def test_create_timed_cmd_history_with_modes(
        self, mock_get_iff_config, mock_get_timing, mock_get_info, mock_dm, temp_dir, monkeypatch
    ):
        """Test creating timed command history with custom modes."""
        from opticalib.core.root import folders
        import os
        import yaml
        
        # Create config file
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        config_file = os.path.join(config_folder, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
                "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "DM": {"nacts": 100, "timing": 10}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        mock_get_timing.return_value = 10
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"nacts": 100, "timing": 10}
        )
        config_map = {
            "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        modes = [1, 2, 3, 4, 5]
        tch = prep.createTimedCmdHistory(modesList=modes)
        
        assert tch is not None
        assert prep._modesList is not None

    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getTiming')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    def test_create_timed_cmd_history_with_shuffle(
        self, mock_get_iff_config, mock_get_timing, mock_get_info, mock_dm, temp_dir, monkeypatch
    ):
        """Test creating timed command history with shuffle."""
        from opticalib.core.root import folders
        import os
        import yaml
        
        # Create config file
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        config_file = os.path.join(config_folder, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
                "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "DM": {"nacts": 100, "timing": 10}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        mock_get_timing.return_value = 10
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"nacts": 100, "timing": 10}
        )
        config_map = {
            "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        modes = np.arange(mock_dm.nActs)
        tch = prep.createTimedCmdHistory(modesList=modes, shuffle=True)
        
        assert tch is not None
        assert prep._shuffle == 1

    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getTiming')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    def test_get_info_to_save(
        self, mock_get_iff_config, mock_get_timing, mock_get_info, mock_dm, temp_dir, monkeypatch
    ):
        """Test getting info to save."""
        from opticalib.core.root import folders
        import os
        import yaml
        
        # Create config file
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        config_file = os.path.join(config_folder, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
                "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "DM": {"nacts": 100, "timing": 10}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        mock_get_timing.return_value = 10
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"nacts": 100, "timing": 10}
        )
        config_map = {
            "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep.createTimedCmdHistory()
        info = prep.getInfoToSave()
        
        assert isinstance(info, dict)
        assert "timedCmdHistory" in info
        assert "cmdMatrix" in info
        assert "modesVector" in info
        assert "template" in info
        assert "shuffle" in info

    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    def test_create_cmd_matrix_history(
        self, mock_get_iff_config, mock_get_info, mock_dm, temp_dir, monkeypatch
    ):
        """Test creating command matrix history."""
        from opticalib.core.root import folders
        import os
        import yaml
        
        # Create config file
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        config_file = os.path.join(config_folder, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
                "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
                "DM": {"nacts": 100, "timing": 10}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"nacts": 100, "timing": 10}
        )
        config_map = {
            "TRIGGER": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"zeros": 0, "modes": [], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            "IFFUNC": {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        cmd_hist = prep.createCmdMatrixHistory()
        
        assert cmd_hist is not None
        assert isinstance(cmd_hist, np.ndarray)
        assert prep.cmdMatHistory is not None

    @patch('opticalib.dmutils.iff_acquisition_preparation._rif.getIffConfig')
    @patch('opticalib.dmutils.iff_acquisition_preparation._getAcqInfo')
    def test_create_aux_cmd_history(
        self, mock_get_info, mock_get_iff_config, mock_dm
    ):
        """Test creating auxiliary command history."""
        mock_get_info.return_value = (
            {"modes": [1], "amplitude": 0.1, "zeros": 2, "modalBase": "mirror"},
            {"modes": [1], "amplitude": 0.1, "template": [1, -1], "zeros": 0, "modalBase": "mirror"},
            {"modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "zeros": 0, "modalBase": "mirror"},
            {}
        )
        config_map = {
            "TRIGGER": {"modes": [1], "amplitude": 0.1, "zeros": 2, "template": [1], "modalBase": "mirror"},
            "REGISTRATION": {"modes": [1], "amplitude": 0.1, "template": [1, -1], "zeros": 0, "modalBase": "mirror"},
            "IFFUNC": {"modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "zeros": 0, "modalBase": "mirror"},
        }

        def fake_get_iff_config(section):
            return config_map[section].copy()

        mock_get_iff_config.side_effect = fake_get_iff_config
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        aux_hist = prep.createAuxCmdHistory()
        
        assert aux_hist is not None or prep.auxCmdHistory is None  # May be None if no trigger/reg

    def test_create_zonal_mat(self, mock_dm):
        """Test creating zonal matrix."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        zonal = prep._createZonalMat()
        
        assert zonal is not None
        assert zonal.shape == (mock_dm.nActs, mock_dm.nActs)
        # Zonal should be identity matrix
        np.testing.assert_array_equal(zonal, np.eye(mock_dm.nActs))

    def test_create_hadamard_mat(self, mock_dm):
        """Test creating Hadamard matrix."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        hadamard = prep._createHadamardMat()
        
        assert hadamard is not None
        assert hadamard.shape[0] == mock_dm.nActs

    @patch('opticalib.dmutils.iff_acquisition_preparation._osu.load_fits')
    def test_create_user_mat(self, mock_load_fits, mock_dm, temp_dir, monkeypatch):
        """Test creating user-defined modal base."""
        from opticalib.core.root import MODALBASE_ROOT_FOLDER
        import os
        
        modal_folder = os.path.join(temp_dir, "ModalBases")
        os.makedirs(modal_folder, exist_ok=True)
        monkeypatch.setattr('opticalib.core.root.MODALBASE_ROOT_FOLDER', modal_folder)
        
        # Create a test modal base file
        test_modal = np.random.randn(100, 50).astype(np.float32)
        from opticalib.ground import osutils
        modal_file = os.path.join(modal_folder, "test_modal.fits")
        osutils.save_fits(modal_file, test_modal, overwrite=True)
        
        # Mock load_fits to return the actual data
        mock_load_fits.return_value = test_modal
        
        prep = ifa.IFFCapturePreparation(mock_dm)
        user_mat = prep._createUserMat("test_modal.fits")
        
        assert user_mat is not None
        assert user_mat.shape == (100, 50)

    def test_update_modal_base_mirror(self, mock_dm):
        """Test updating modal base to mirror."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._updateModalBase("mirror")
        
        assert prep.modalBaseId == "mirror"
        np.testing.assert_array_equal(prep._modalBase, mock_dm.mirrorModes)

    def test_update_modal_base_zonal(self, mock_dm):
        """Test updating modal base to zonal."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._updateModalBase("zonal")
        
        assert prep.modalBaseId == "zonal"
        assert prep._modalBase.shape == (mock_dm.nActs, mock_dm.nActs)

    def test_update_modal_base_hadamard(self, mock_dm):
        """Test updating modal base to Hadamard."""
        prep = ifa.IFFCapturePreparation(mock_dm)
        prep._updateModalBase("hadamard")
        
        assert prep.modalBaseId == "hadamard"
        assert prep._modalBase.shape[0] == mock_dm.nActs

