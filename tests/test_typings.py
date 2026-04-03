"""
Tests for opticalib.typings module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib import typings


class TestArrayStrFormatter:
    """Test array_str_formatter function."""

    def test_single_float_array_small_values(self):
        """Test formatting of a float array with small values."""
        arr = np.array([0.123, 0.456, 0.789])
        result = typings.array_str_formatter(arr)
        assert isinstance(result, str)
        assert "[" in result

    def test_single_int_array(self):
        """Test formatting of an integer array."""
        arr = np.array([1, 2, 3])
        result = typings.array_str_formatter(arr)
        assert isinstance(result, str)
        assert "," in result

    def test_single_float_array_large_values(self):
        """Test formatting of a float array with large values (scientific notation)."""
        arr = np.array([1e5, 2e5, 3e5])
        result = typings.array_str_formatter(arr)
        assert isinstance(result, str)
        assert "e" in result.lower()

    def test_single_float_array_small_floats(self):
        """Test formatting of a float array with very small values (scientific notation)."""
        arr = np.array([1e-5, 2e-5, 3e-5])
        result = typings.array_str_formatter(arr)
        assert isinstance(result, str)
        assert "e" in result.lower()

    def test_list_of_arrays_returns_list(self):
        """Test that a list of arrays returns a list of strings."""
        arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = typings.array_str_formatter(arrays)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_single_array_returns_string(self):
        """Test that a single array (not in a list) returns a string."""
        arr = np.array([1.5, 2.5])
        result = typings.array_str_formatter(arr)
        assert isinstance(result, str)

    def test_list_with_non_ndarray_elements(self):
        """Test that list elements are converted to ndarray."""
        arrays = [[1.0, 2.0], [3.0, 4.0]]
        result = typings.array_str_formatter(arrays)
        assert isinstance(result, list)
        assert len(result) == 2


class TestInstanceCheck:
    """Test InstanceCheck static methods."""

    def test_is_matrix_like_2d_ndarray(self):
        """Test is_matrix_like with a 2D numpy array."""
        arr = np.random.randn(5, 5)
        assert typings.InstanceCheck.is_matrix_like(arr) is True

    def test_is_matrix_like_1d_ndarray(self):
        """Test is_matrix_like with a 1D numpy array."""
        arr = np.random.randn(5)
        assert typings.InstanceCheck.is_matrix_like(arr) is False

    def test_is_matrix_like_masked_array(self):
        """Test is_matrix_like with a masked array (which is ImageData)."""
        data = np.random.randn(5, 5)
        mask = np.zeros((5, 5), dtype=bool)
        arr = ma.masked_array(data, mask=mask)
        # masked arrays are ImageData not MatrixLike
        assert typings.InstanceCheck.is_matrix_like(arr) is False

    def test_is_mask_like_bool_array(self):
        """Test is_mask_like with a boolean array."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        assert typings.InstanceCheck.is_mask_like(mask) is True

    def test_is_mask_like_uint8_array(self):
        """Test is_mask_like with a uint8 array."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        assert typings.InstanceCheck.is_mask_like(mask) is True

    def test_is_mask_like_1d_array(self):
        """Test is_mask_like with a 1D array returns False."""
        mask = np.zeros(10, dtype=bool)
        assert typings.InstanceCheck.is_mask_like(mask) is False

    def test_is_mask_like_non_array(self):
        """Test is_mask_like with a non-array returns False."""
        assert typings.InstanceCheck.is_mask_like(42) is False

    def test_is_image_like_masked_array(self):
        """Test is_image_like with a valid masked array."""
        data = np.random.randn(10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        img = ma.masked_array(data, mask=mask)
        assert typings.InstanceCheck.is_image_like(img) is True

    def test_is_image_like_plain_ndarray(self):
        """Test is_image_like with a plain ndarray returns False."""
        arr = np.random.randn(10, 10)
        assert typings.InstanceCheck.is_image_like(arr) is False

    def test_is_cube_like_3d_masked_array(self):
        """Test is_cube_like with a 3D masked array."""
        data = np.random.randn(10, 10, 5)
        mask = np.zeros((10, 10, 5), dtype=bool)
        cube = ma.masked_array(data, mask=mask)
        assert typings.InstanceCheck.is_cube_like(cube) is True

    def test_is_cube_like_2d_masked_array(self):
        """Test is_cube_like with a 2D masked array returns False."""
        data = np.random.randn(10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        img = ma.masked_array(data, mask=mask)
        assert typings.InstanceCheck.is_cube_like(img) is False

    def test_generic_check_dm_protocol(self):
        """Test generic_check with a DeformableMirrorDevice-like object."""

        class FakeDM:
            @property
            def nActs(self):
                return 100

            def set_shape(self, cmd, differential=False):
                pass

            def get_shape(self):
                return np.zeros(100)

            def uploadCmdHistory(self, cmdhist):
                pass

            def runCmdHistory(
                self, interf=None, delay=0, save=None, differential=False
            ):
                return "tn"

        dm = FakeDM()
        assert typings.InstanceCheck.generic_check(dm, "DeformableMirrorDevice") is True

    def test_generic_check_unknown_class(self):
        """Test generic_check with unknown class raises ValueError."""
        with pytest.raises(ValueError, match="not found in the current context"):
            typings.InstanceCheck.generic_check(object(), "UnknownClass")


class TestIsinstance:
    """Test the isinstance_ function."""

    def test_isinstance_image_data(self):
        """Test isinstance_ for ImageData."""
        data = np.random.randn(10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        img = ma.masked_array(data, mask=mask)
        assert typings.isinstance_(img, "ImageData") is True

    def test_isinstance_cube_data(self):
        """Test isinstance_ for CubeData."""
        data = np.random.randn(10, 10, 5)
        mask = np.zeros((10, 10, 5), dtype=bool)
        cube = ma.masked_array(data, mask=mask)
        assert typings.isinstance_(cube, "CubeData") is True

    def test_isinstance_matrix_like(self):
        """Test isinstance_ for MatrixLike."""
        arr = np.random.randn(5, 5)
        assert typings.isinstance_(arr, "MatrixLike") is True

    def test_isinstance_mask_data(self):
        """Test isinstance_ for MaskData."""
        mask = np.zeros((10, 10), dtype=bool)
        assert typings.isinstance_(mask, "MaskData") is True

    def test_isinstance_unknown_class(self):
        """Test isinstance_ raises ValueError for unknown class."""
        with pytest.raises(ValueError, match="Unknown class name"):
            typings.isinstance_(object(), "UnknownClass")

    def test_isinstance_interferometer_device(self):
        """Test isinstance_ for InterferometerDevice protocol."""

        class FakeInterf:
            def acquire_map(self, nframes=1, delay=0, rebin=1):
                return None

            def acquireFullFrame(self, **kwargs):
                return None

            def capture(self, numberOfFrames=1, folder_name=None):
                return "tn"

            def produce(self, tn):
                pass

        interf = FakeInterf()
        assert typings.isinstance_(interf, "InterferometerDevice") is True

    def test_isinstance_deformable_mirror_device(self):
        """Test isinstance_ for DeformableMirrorDevice protocol."""

        class FakeDM:
            @property
            def nActs(self):
                return 100

            def set_shape(self, cmd, differential=False):
                pass

            def get_shape(self):
                return np.zeros(100)

            def uploadCmdHistory(self, cmdhist):
                pass

            def runCmdHistory(
                self, interf=None, delay=0, save=None, differential=False
            ):
                return "tn"

        dm = FakeDM()
        assert typings.isinstance_(dm, "DeformableMirrorDevice") is True
