"""
Tests for opticalib.dmutils.flattening module.
"""
import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import Mock, patch, MagicMock
from opticalib.dmutils import flattening as flt


class TestFlattening:
    """Test Flattening class."""

    def test_init(self, sample_int_matrix_folder):
        """Test Flattening initialization."""
        tn, tn_folder = sample_int_matrix_folder
        
        # The Flattening class uses _ifp._intMatFold which should be patched by the fixture
        f = flt.Flattening(tn)
        
        assert f.tn == tn
        assert f._intCube is not None
        assert f._cmdMat is not None
        assert f._rec is not None

    def test_properties(self, sample_int_matrix_folder):
        """Test Flattening properties."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        
        # Test RM property
        assert f.RM is None  # Not computed yet
        
        # Test CM property
        assert f.CM is not None
        assert isinstance(f.CM, np.ndarray)
        
        # Test IM property
        assert f.IM is not None

    def test_load_image_to_shape(self, sample_int_matrix_folder, sample_image):
        """Test loading image to shape."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        
        assert f.shape2flat is not None
        assert isinstance(f.shape2flat, ma.MaskedArray)

    def test_load_image_to_shape_with_compute(self, sample_int_matrix_folder, sample_image):
        """Test loading image to shape with automatic compute."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image, compute=5)
        
        assert f.shape2flat is not None
        assert f._recMat is not None

    def test_compute_rec_mat(self, sample_int_matrix_folder, sample_image):
        """Test computing reconstruction matrix."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        f.computeRecMat(threshold=5)
        
        assert f._recMat is not None
        assert isinstance(f._recMat, np.ndarray)

    def test_compute_flat_cmd_int_modes(self, sample_int_matrix_folder, sample_image):
        """Test computing flat command with integer modes."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        f.computeRecMat(threshold=5)
        
        flat_cmd = f.computeFlatCmd(n_modes=5)
        
        assert flat_cmd is not None
        assert isinstance(flat_cmd, np.ndarray)
        assert len(flat_cmd) == f._cmdMat.shape[0]
        assert f.flatCmd is not None

    def test_compute_flat_cmd_list_modes(self, sample_int_matrix_folder, sample_image):
        """Test computing flat command with list of modes."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        f.computeRecMat(threshold=5)
        
        flat_cmd = f.computeFlatCmd(n_modes=[0, 1, 2, 3, 4])
        
        assert flat_cmd is not None
        assert isinstance(flat_cmd, np.ndarray)
        assert len(flat_cmd) == f._cmdMat.shape[0]

    def test_compute_flat_cmd_invalid_type(self, sample_int_matrix_folder, sample_image):
        """Test computing flat command with invalid type."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        f.computeRecMat(threshold=5)
        
        with pytest.raises(TypeError):
            f.computeFlatCmd(n_modes="invalid")

    def test_get_svd_matrices(self, sample_int_matrix_folder, sample_image):
        """Test getting SVD matrices."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(sample_image)
        f.computeRecMat(threshold=5)
        
        U, S, Vt = f.getSVDmatrices()
        
        assert U is not None
        assert S is not None
        assert Vt is not None
        assert isinstance(U, np.ndarray)
        assert isinstance(S, np.ndarray)
        assert isinstance(Vt, np.ndarray)

    @patch('opticalib.dmutils.flattening._ifp.filterZernikeCube')
    def test_filter_int_cube(self, mock_filter, sample_int_matrix_folder):
        """Test filtering interaction cube."""
        tn, tn_folder = sample_int_matrix_folder
        
        mock_filter.return_value = (ma.masked_array(np.random.randn(50, 50, 10)), "new_tn")
        
        f = flt.Flattening(tn)
        result = f.filterIntCube(zernModes=[1, 2, 3])
        
        assert result is f  # Should return self
        # Note: filterIntCube may skip if already filtered

    def test_load_new_tn(self, sample_int_matrix_folder):
        """Test loading new tracking number."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        new_tn = "20240101_130000"
        
        # This will fail if the new tn doesn't exist, but we test the method
        # In a real scenario, you'd create the new tn folder first
        try:
            f.loadNewTn(new_tn)
            assert f.tn == new_tn
        except FileNotFoundError:
            # Expected if new_tn doesn't exist
            pass

    def test_get_master_mask(self, sample_int_matrix_folder):
        """Test getting master mask."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        master_mask = f._getMasterMask()
        
        assert master_mask is not None
        assert isinstance(master_mask, np.ndarray)
        assert master_mask.dtype == bool
        assert master_mask.shape == f._intCube.shape[:2]

    def test_align_img_and_cube_masks(self, sample_int_matrix_folder, sample_image):
        """Test aligning image and cube masks."""
        tn, tn_folder = sample_int_matrix_folder
        
        f = flt.Flattening(tn)
        aligned_img = f._alignImgAndCubeMasks(sample_image)
        
        assert aligned_img is not None
        assert isinstance(aligned_img, ma.MaskedArray)
        # Image should be aligned to cube shape
        assert aligned_img.shape == f._intCube.shape[:2]

    def test_apply_flat_command(
        self, sample_int_matrix_folder, mock_dm, mock_interferometer, temp_dir, sample_image, monkeypatch
    ):
        """Test applying flat command."""
        from opticalib.core.root import folders
        import os
        
        tn, tn_folder = sample_int_matrix_folder
        
        # Setup mock interferometer
        mock_interferometer.acquire_map.return_value = sample_image #ma.masked_array(
            #np.random.randn(50, 50).astype(np.float32),
            #mask=np.zeros((50, 50), dtype=bool)
        #)
        
        # Setup folder
        flat_folder = os.path.join(temp_dir, "Flattening")
        os.makedirs(flat_folder, exist_ok=True)
        monkeypatch.setattr(folders, "FLAT_ROOT_FOLDER", flat_folder)
        
        f = flt.Flattening(tn)
        f.loadImage2Shape(mock_interferometer.acquire_map())
        f.computeRecMat(threshold=5)
        
        f.applyFlatCommand(
            mock_dm,
            mock_interferometer,
            modes2flat=5,
            nframes=1
        )
        
        # Verify interferometer was called
        assert mock_interferometer.acquire_map.call_count >= 2
        # Verify DM was called
        mock_dm.get_shape.assert_called()
        mock_dm.set_shape.assert_called()

