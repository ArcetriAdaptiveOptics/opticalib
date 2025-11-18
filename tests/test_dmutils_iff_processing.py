"""
Tests for opticalib.dmutils.iff_processing module.
"""
import pytest
import os
import numpy as np
import numpy.ma as ma
from unittest.mock import Mock, patch, MagicMock
from opticalib.dmutils import iff_processing as ifp
from opticalib.ground import osutils


class TestSaveCube:
    """Test saveCube function."""

    def test_save_cube_basic(self, sample_iff_folder_structure, temp_dir, monkeypatch):
        """Test saving a cube."""
        from opticalib.core.root import folders
        
        tn, tn_folder = sample_iff_folder_structure
        
        int_folder = os.path.join(temp_dir, "INTMatrices")
        os.makedirs(int_folder, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", int_folder)
        monkeypatch.setattr(ifp, "_intMatFold", int_folder)
        iff_folder = os.path.dirname(tn_folder)

        def fake_get_file_list(tn_arg=None, fold=None, key=None):
            folder = os.path.join(iff_folder, tn)
            files = sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if key is None or key in f
            )
            return files
        monkeypatch.setattr(osutils, "getFileList", fake_get_file_list)
        
        cube = ifp.saveCube(tn, rebin=1)
        
        assert cube is not None
        assert isinstance(cube, ma.MaskedArray)
        cube_path = os.path.join(int_folder, tn, "IMCube.fits")
        assert os.path.exists(cube_path)

    def test_save_cube_with_rebin(self, sample_iff_folder_structure, temp_dir, monkeypatch):
        """Test saving a cube with rebinning."""
        from opticalib.core.root import folders
        
        tn, tn_folder = sample_iff_folder_structure
        
        int_folder = os.path.join(temp_dir, "INTMatrices")
        os.makedirs(int_folder, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", int_folder)
        monkeypatch.setattr(ifp, "_intMatFold", int_folder)
        iff_folder = os.path.dirname(tn_folder)

        def fake_get_file_list(tn_arg=None, fold=None, key=None):
            folder = os.path.join(iff_folder, tn)
            files = sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if key is None or key in f
            )
            return files
        monkeypatch.setattr(osutils, "getFileList", fake_get_file_list)
        
        cube = ifp.saveCube(tn, rebin=2)
        
        assert cube is not None
        # Rebinning should reduce size
        assert cube.shape[0] < 50

    def test_save_cube_with_header(self, sample_iff_folder_structure, temp_dir, monkeypatch):
        """Test saving a cube with custom header."""
        from opticalib.core.root import folders
        from opticalib.ground import osutils
        
        tn, tn_folder = sample_iff_folder_structure
        
        int_folder = os.path.join(temp_dir, "INTMatrices")
        os.makedirs(int_folder, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", int_folder)
        monkeypatch.setattr(ifp, "_intMatFold", int_folder)
        iff_folder = os.path.dirname(tn_folder)

        def fake_get_file_list(tn_arg=None, fold=None, key=None):
            folder = os.path.join(iff_folder, tn)
            files = sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if key is None or key in f
            )
            return files
        
        custom_header = {"TESTKEY": "testvalue"}
        monkeypatch.setattr(osutils, "getFileList", fake_get_file_list)
        cube = ifp.saveCube(tn, rebin=1, cube_header=custom_header)
        
        assert cube is not None
        # Verify header was saved
        cube_path = os.path.join(int_folder, tn, "IMCube.fits")
        from astropy.io import fits
        with fits.open(cube_path) as hdul:
            assert "TESTKEY" in hdul[0].header


class TestStackCubes:
    """Test stackCubes function."""

    def test_stack_cubes_basic(self, sample_int_matrix_folder, temp_dir, monkeypatch):
        """Test stacking cubes."""
        from opticalib.core.root import folders
        from opticalib.ground import osutils
        
        tn1, tn1_folder = sample_int_matrix_folder
        
        # Create second cube with same shape as first
        tn2 = "20240101_130000"
        int_folder = os.path.dirname(tn1_folder)
        tn2_folder = os.path.join(int_folder, tn2)
        os.makedirs(tn2_folder, exist_ok=True)
        
        # Create second cube with same number of modes as first (10)
        cube2 = ma.masked_array(
            np.random.randn(100, 100, 10).astype(np.float32),
            mask=np.ones((100, 100, 10), dtype=bool)
        )
        osutils.save_fits(
            os.path.join(tn2_folder, "IMCube.fits"),
            cube2,
            overwrite=True,
            header={"REBIN": 1}
        )
        
        cmd_mat2 = np.random.randn(100, 10).astype(np.float32)
        osutils.save_fits(
            os.path.join(tn2_folder, "cmdMatrix.fits"),
            cmd_mat2,
            overwrite=True
        )
        
        modes_vec2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        osutils.save_fits(
            os.path.join(tn2_folder, "modesVector.fits"),
            modes_vec2,
            overwrite=True
        )
        
        tnlist = [tn1, tn2]
        monkeypatch.setattr(ifp, "_checkStackedCubes", lambda tn_list: 0)
        ifp.stackCubes(tnlist, cubeNames=None)
        
        # Verify stacked cube was created
        stacked_folders = [f for f in os.listdir(int_folder) if os.path.isdir(os.path.join(int_folder, f)) and f != tn1 and f != tn2]
        assert len(stacked_folders) > 0


class TestFilterZernikeCube:
    """Test filterZernikeCube function."""

    def test_filter_zernike_cube_basic(self, sample_int_matrix_folder):
        """Test filtering Zernike modes from cube."""
        import shutil
        tn, tn_folder = sample_int_matrix_folder
        
        # Clean up any existing folders that might conflict
        int_folder = os.path.dirname(tn_folder)
        for item in os.listdir(int_folder):
            if item != tn and os.path.isdir(os.path.join(int_folder, item)):
                try:
                    shutil.rmtree(os.path.join(int_folder, item))
                except:
                    pass
        
        ffcube, new_tn = ifp.filterZernikeCube(tn, zern_modes=[1, 2, 3], save=True)

        assert ffcube is not None
        assert isinstance(ffcube, ma.MaskedArray)
        assert new_tn is not None
        # Verify filtered cube was saved
        new_tn_folder = os.path.join(os.path.dirname(tn_folder), new_tn)
        assert os.path.exists(os.path.join(new_tn_folder, "IMCube.fits"))

    def test_filter_zernike_cube_custom_modes(self, sample_int_matrix_folder):
        """Test filtering custom Zernike modes."""
        import shutil
        tn, tn_folder = sample_int_matrix_folder
        
        # Clean up any existing folders
        int_folder = os.path.dirname(tn_folder)
        for item in os.listdir(int_folder):
            if item != tn and os.path.isdir(os.path.join(int_folder, item)):
                try:
                    shutil.rmtree(os.path.join(int_folder, item))
                except:
                    pass
        
        ffcube, new_tn = ifp.filterZernikeCube(tn, zern_modes=[1, 2, 3, 4], save=True)
        
        assert ffcube is not None
        assert new_tn is not None

    def test_filter_zernike_cube_no_save(self, sample_int_matrix_folder):
        """Test filtering without saving."""
        import shutil
        tn, tn_folder = sample_int_matrix_folder
        
        # Clean up any existing folders
        int_folder = os.path.dirname(tn_folder)
        for item in os.listdir(int_folder):
            if item != tn and os.path.isdir(os.path.join(int_folder, item)):
                try:
                    shutil.rmtree(os.path.join(int_folder, item))
                except:
                    pass
        
        ffcube, new_tn = ifp.filterZernikeCube(tn, zern_modes=[1, 2, 3], save=False)
        
        assert ffcube is not None
        assert new_tn is not None


class TestGetAcqPar:
    """Test _getAcqPar function."""

    def test_get_acq_par(self, sample_iff_folder_structure, temp_dir, monkeypatch):
        """Test getting acquisition parameters."""
        from opticalib.core.root import folders
        from opticalib.dmutils import iff_processing as ifp
        from opticalib.ground import osutils
        
        # Ensure the module-level _ifFold is patched
        iff_folder = os.path.join(osutils._OPTDATA, "IFFunctions")
        monkeypatch.setattr(ifp, "_ifFold", iff_folder)
        
        tn, tn_folder = sample_iff_folder_structure
        
        ampVector, modesVector, template, indexList, registrationActs, shuffle = ifp._getAcqPar(tn)
        
        assert ampVector is not None
        assert modesVector is not None
        assert template is not None
        assert indexList is not None
        assert registrationActs is not None
        assert isinstance(shuffle, int)


class TestGetAcqInfo:
    """Test _getAcqInfo function."""

    @patch('opticalib.dmutils.iff_processing._rif.getIffConfig')
    @patch('opticalib.dmutils.iff_processing._rif.getDmIffConfig')
    def test_get_acq_info(self, mock_dm_config, mock_iff_config, temp_dir, monkeypatch):
        """Test getting acquisition info."""
        from opticalib.core.root import folders
        
        # Mock config responses
        mock_iff_config.side_effect = [
            {"zeros": 0, "modes": [1], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"},
            {"zeros": 0, "modes": [1, 2, 3], "amplitude": 0.1, "template": [1, -1], "modalBase": "mirror"}
        ]
        mock_dm_config.return_value = {"nacts": 100, "timing": 10}
        
        config_folder = os.path.join(temp_dir, "SysConfig")
        os.makedirs(config_folder, exist_ok=True)
        monkeypatch.setattr(folders, "CONFIGURATION_FOLDER", config_folder)
        
        infoT, infoR, infoIF, infoDM = ifp._getAcqInfo()
        
        assert infoT is not None
        assert infoR is not None
        assert infoIF is not None
        assert infoDM is not None


class TestGetTriggerFrame:
    """Test getTriggerFrame function."""

    @patch('opticalib.dmutils.iff_processing._osu.getFileList')
    @patch('opticalib.dmutils.iff_processing._osu.read_phasemap')
    @patch('opticalib.dmutils.iff_processing._getAcqInfo')
    def test_get_trigger_frame_no_zeros(self, mock_get_info, mock_read_phasemap, mock_get_file_list):
        """Test getting trigger frame with no zeros."""
        # Mock setup
        mock_get_info.return_value = (
            {"zeros": 0, "modes": [], "amplitude": 0.1},
            {},
            {},
            {}
        )
        mock_get_file_list.return_value = ["file1.fits", "file2.fits"]
        
        trig_frame = ifp.getTriggerFrame("test_tn")
        
        assert trig_frame == 0

    @patch('opticalib.dmutils.iff_processing._zern.ZernikeFitter')
    @patch('opticalib.dmutils.iff_processing._osu.getFileList')
    @patch('opticalib.dmutils.iff_processing._osu.read_phasemap')
    @patch('opticalib.dmutils.iff_processing._getAcqInfo')
    def test_get_trigger_frame_with_trigger(
        self, mock_get_info, mock_read_phasemap, mock_get_file_list, mock_zernike
    ):
        """Test getting trigger frame with trigger detection."""
        import numpy.ma as ma
        
        # Mock setup
        mock_get_info.return_value = (
            {"zeros": 2, "modes": [1], "amplitude": 0.1},
            {},
            {},
            {}
        )
        mock_get_file_list.return_value = ["file0.fits", "file1.fits", "file2.fits", "file3.fits"]
        
        # Create mock images - first few with low std, then one with high std
        img0 = ma.masked_array(np.random.randn(50, 50) * 0.01)
        img1 = ma.masked_array(np.random.randn(50, 50) * 0.01)
        img2 = ma.masked_array(np.random.randn(50, 50) * 0.5)  # High std - trigger
        
        mock_read_phasemap.side_effect = [img0, img1, img2]
        
        # Mock Zernike fitter
        mock_fitter = MagicMock()
        mock_fitter.removeZernike = Mock(side_effect=lambda img, modes: img)
        mock_zernike.return_value = mock_fitter
        
        trig_frame = ifp.getTriggerFrame("test_tn", amplitude=0.1)
        
        assert trig_frame >= 0

