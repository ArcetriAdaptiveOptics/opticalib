"""
Tests for opticalib.analyzer.images_processing module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib.core.fitsarray import fits_array
from opticalib.analyzer import images_processing as ip


def _make_image(shape=(10, 10), circular=False):
    """Create a FitsMaskedArray with an optional circular mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions.
    circular : bool
        If True, mask everything outside the inscribed circle.

    Returns
    -------
    FitsMaskedArray
    """
    data = np.random.randn(*shape)
    mask = np.zeros(shape, dtype=bool)
    if circular:
        r = min(shape) // 2 - 2
        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        mask[(x - cx) ** 2 + (y - cy) ** 2 >= r**2] = True
    return fits_array(ma.masked_array(data, mask=mask))


def _make_cube(shape=(10, 10, 5), circular=False):
    """Create a FitsMaskedArray cube."""
    data = np.random.randn(*shape)
    mask = np.zeros(shape, dtype=bool)
    if circular:
        r = min(shape[:2]) // 2 - 2
        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        circle_mask = (x - cx) ** 2 + (y - cy) ** 2 >= r**2
        mask[circle_mask, :] = True
    return fits_array(ma.masked_array(data, mask=mask))


class TestFrame:
    """Tests for the frame function."""

    def test_frame_from_image_list(self):
        """Test retrieving a frame from a list of ImageData objects."""
        imgs = [_make_image() for _ in range(5)]
        result = ip.frame(2, imgs)

        assert result.shape == imgs[0].shape
        np.testing.assert_array_equal(result.data, imgs[2].data)

    def test_frame_from_cube(self):
        """Test retrieving a frame from a 3D cube."""
        cube = _make_cube((10, 10, 5))
        result = ip.frame(3, cube)

        assert result.shape == (10, 10)
        np.testing.assert_array_equal(result.data, cube[:, :, 3].data)

    def test_frame_index_zero(self):
        """Test retrieving the first frame."""
        imgs = [_make_image() for _ in range(3)]
        result = ip.frame(0, imgs)

        np.testing.assert_array_equal(result.data, imgs[0].data)

    def test_frame_out_of_range_raises(self):
        """Test that an out-of-range index raises IndexError."""
        imgs = [_make_image() for _ in range(3)]
        with pytest.raises(IndexError, match="Index out of range"):
            ip.frame(10, imgs)


class TestPistonUnwrap:
    """Tests for the piston_unwrap function."""

    def test_no_wrapping_needed(self):
        """Test that a smooth piston vector is unchanged."""
        piston = np.array([100.0, 200.0, 300.0], dtype=float)
        result = ip.piston_unwrap(piston, wavelength=632.8)

        assert result.shape == piston.shape

    def test_default_wavelength_message(self, capsys):
        """Test that missing wavelength prints a warning."""
        piston = np.array([100.0, 200.0, 300.0], dtype=float)
        ip.piston_unwrap(piston)
        captured = capsys.readouterr()
        assert "Wavelength not specified" in captured.out

    def test_wavelength_in_meters_converted(self):
        """Test that wavelength in metres is converted to nm internally."""
        piston = np.array([100.0, 200.0, 300.0], dtype=float)
        result_nm = ip.piston_unwrap(piston.copy(), wavelength=632.8)
        result_m = ip.piston_unwrap(piston.copy(), wavelength=632.8e-9)

        np.testing.assert_allclose(result_nm, result_m, rtol=1e-6)

    def test_with_commanded_piston(self):
        """Test unwrapping with a commanded piston vector."""
        commanded = np.array([100.0, 200.0, 300.0])
        measured = np.array([100.0, 200.0, 300.0])  # no offset
        result = ip.piston_unwrap(measured, commanded_piston_vec=commanded, wavelength=632.8)

        assert result.shape == measured.shape

    def test_output_shape_preserved(self):
        """Test that the output shape matches the input."""
        piston = np.random.randn(50) * 100
        result = ip.piston_unwrap(piston, wavelength=632.8)

        assert result.shape == piston.shape


class TestCreateCube:
    """Tests for the createCube function."""

    def test_from_image_list(self):
        """Test creating a cube from a list of ImageData."""
        imgs = [_make_image((8, 8)) for _ in range(4)]
        cube = ip.createCube(imgs)

        assert cube.shape == (8, 8, 4)

    def test_cube_values_match_input(self):
        """Test that cube frames match the input images."""
        imgs = [_make_image((5, 5)) for _ in range(3)]
        cube = ip.createCube(imgs)

        for i, img in enumerate(imgs):
            np.testing.assert_array_equal(cube[:, :, i].data, img.data)

    def test_non_list_input_raises(self):
        """Test that a non-list input raises TypeError."""
        with pytest.raises(TypeError, match="filelist must be a list"):
            ip.createCube(np.random.randn(5, 5))

    def test_single_image_cube(self):
        """Test creating a cube from a single-element list."""
        img = _make_image((6, 6))
        cube = ip.createCube([img])

        assert cube.ndim == 3
        assert cube.shape[2] == 1


class TestPushPullReductionAlgorithm:
    """Tests for the pushPullReductionAlgorithm function."""

    def test_basic_two_image_template(self):
        """Test the algorithm with the simplest [1, -1] template."""
        img = _make_image((20, 20))
        template = np.array([1, -1])
        result = ip.pushPullReductionAlgorithm([img, -img], template)

        assert result.shape == img.shape

    def test_three_image_template(self):
        """Test with a [1, -1, 1] push-pull-push template."""
        img = _make_image((20, 20))
        template = np.array([1, -1, 1])
        result = ip.pushPullReductionAlgorithm([img, -img, img], template)

        assert result.shape == img.shape

    def test_output_is_masked_array(self):
        """Test that the output is a masked array."""
        img = _make_image((20, 20))
        template = np.array([1, -1])
        result = ip.pushPullReductionAlgorithm([img, -img], template)

        assert isinstance(result, np.ma.MaskedArray)

    def test_custom_normalization(self):
        """Test with a custom normalization factor."""
        img = _make_image((20, 20))
        template = np.array([1, -1])
        result = ip.pushPullReductionAlgorithm([img, -img], template, normalization=2.0)

        assert result.shape == img.shape

    def test_mask_is_or_of_inputs(self):
        """Test that the output mask is the OR of all input masks."""
        data = np.ones((10, 10))
        mask1 = np.zeros((10, 10), dtype=bool)
        mask1[0, :] = True
        mask2 = np.zeros((10, 10), dtype=bool)
        mask2[:, 0] = True

        img1 = fits_array(ma.masked_array(data, mask=mask1))
        img2 = fits_array(ma.masked_array(data, mask=mask2))
        template = np.array([1, -1])
        result = ip.pushPullReductionAlgorithm([img1, img2], template)

        # The combined mask should cover both masked rows/cols
        assert result.mask[0, 0]
        assert result.mask[0, 5]
        assert result.mask[5, 0]


class TestModeRebinner:
    """Tests for the modeRebinner function."""

    def test_downsampling_by_2(self):
        """Test downsampling an image by factor 2."""
        img = _make_image((100, 100))
        result = ip.modeRebinner(img, 2)

        assert result.shape == (50, 50)

    def test_downsampling_by_4(self):
        """Test downsampling an image by factor 4."""
        img = _make_image((80, 80))
        result = ip.modeRebinner(img, 4)

        assert result.shape == (20, 20)

    def test_rebin_1_is_identity(self):
        """Test that rebinning by 1 returns an identical image."""
        img = _make_image((20, 20))
        result = ip.modeRebinner(img, 1)

        assert result.shape == img.shape

    def test_output_is_masked_array(self):
        """Test that the output is a masked array."""
        img = _make_image((40, 40))
        result = ip.modeRebinner(img, 2)

        assert isinstance(result, np.ma.MaskedArray)

    @pytest.mark.parametrize("method", ["averaging", "sum", "median", "bilinear", "bicubic"])
    def test_various_methods(self, method):
        """Test different rebinning methods."""
        img = _make_image((40, 40))
        result = ip.modeRebinner(img, 2, method=method)

        assert result.shape == (20, 20)

    def test_invalid_method_raises(self):
        """Test that an unsupported method raises ValueError."""
        img = _make_image((40, 40))
        with pytest.raises(ValueError, match="Unsupported rebin method"):
            ip.modeRebinner(img, 2, method="invalid_method")


class TestCubeRebinner:
    """Tests for the cubeRebinner function."""

    def test_downsample_cube_by_2(self):
        """Test downsampling a cube by factor 2."""
        cube = _make_cube((100, 100, 4))
        result = ip.cubeRebinner(cube, 2)

        assert result.shape == (50, 50, 4)

    def test_cube_frame_count_preserved(self):
        """Test that the number of frames in the cube is preserved."""
        n_frames = 6
        cube = _make_cube((60, 60, n_frames))
        result = ip.cubeRebinner(cube, 2)

        assert result.shape[2] == n_frames


class TestRebin2DArray:
    """Tests for the rebin2DArray function."""

    def test_downsampling(self):
        """Test basic downsampling."""
        arr = ma.masked_array(np.ones((100, 100)), mask=np.zeros((100, 100), dtype=bool))
        result = ip.rebin2DArray(arr, (50, 50))

        assert result.shape == (50, 50)

    def test_values_preserved_for_constant_array(self):
        """Test that mean-rebinning a constant array preserves the value."""
        val = 3.14
        arr = ma.masked_array(
            np.full((40, 40), val), mask=np.zeros((40, 40), dtype=bool)
        )
        result = ip.rebin2DArray(arr, (20, 20), method="mean")

        np.testing.assert_allclose(result.data[~result.mask], val, rtol=1e-6)

    def test_non_2d_raises(self):
        """Test that a non-2D input raises ValueError."""
        arr = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="2-dimensional"):
            ip.rebin2DArray(arr, (2, 2))

    def test_invalid_new_shape_raises(self):
        """Test that non-integer new_shape raises ValueError."""
        arr = ma.masked_array(np.ones((10, 10)), mask=np.zeros((10, 10), dtype=bool))
        with pytest.raises(ValueError, match="new_shape must be"):
            ip.rebin2DArray(arr, "bad_shape")

    def test_zero_new_shape_raises(self):
        """Test that zero new_shape raises ValueError."""
        arr = ma.masked_array(np.ones((10, 10)), mask=np.zeros((10, 10), dtype=bool))
        with pytest.raises(ValueError, match="positive"):
            ip.rebin2DArray(arr, (0, 5))

    def test_same_shape_returns_copy(self):
        """Test that same shape returns a copy without modification."""
        arr = ma.masked_array(np.ones((10, 10)), mask=np.zeros((10, 10), dtype=bool))
        result = ip.rebin2DArray(arr, (10, 10))

        assert result.shape == (10, 10)

    def test_invalid_method_raises(self):
        """Test that an invalid method name raises ValueError."""
        arr = ma.masked_array(np.ones((10, 10)), mask=np.zeros((10, 10), dtype=bool))
        with pytest.raises(ValueError, match="Unsupported rebin method"):
            ip.rebin2DArray(arr, (5, 5), method="invalid")


class TestCompFilteredImage:
    """Tests for the comp_filtered_image function."""

    def test_output_shape(self):
        """Test that the output has the same shape as the input."""
        img = _make_image((50, 50), circular=True)
        result = ip.comp_filtered_image(img)

        assert result.shape == img.shape

    def test_output_is_masked_array(self):
        """Test that the output is a masked array."""
        img = _make_image((50, 50), circular=True)
        result = ip.comp_filtered_image(img)

        assert isinstance(result, np.ma.MaskedArray)

    def test_with_freq_filter(self):
        """Test with an explicit frequency filter range."""
        img = _make_image((50, 50), circular=True)
        result = ip.comp_filtered_image(img, freq2filter=(0.05, 0.4))

        assert result.shape == img.shape

    def test_mask_preserved(self):
        """Test that the output mask matches the input mask."""
        img = _make_image((50, 50), circular=True)
        result = ip.comp_filtered_image(img)

        np.testing.assert_array_equal(result.mask, img.mask)


class TestComputePsd:
    """Tests for the compute_psd function."""

    def test_basic_output_shapes(self):
        """Test that output frequency and amplitude arrays have consistent shapes."""
        img = _make_image((60, 60), circular=True)
        fout, Aout = ip.compute_psd(img)

        assert fout.shape == Aout.shape
        assert len(fout) > 0

    def test_output_arrays_are_1d(self):
        """Test that both outputs are 1D."""
        img = _make_image((60, 60), circular=True)
        fout, Aout = ip.compute_psd(img)

        assert fout.ndim == 1
        assert Aout.ndim == 1

    def test_custom_nbins(self):
        """Test with a custom number of bins."""
        img = _make_image((60, 60), circular=True)
        nbins = 10
        fout, Aout = ip.compute_psd(img, nbins=nbins)

        assert len(Aout) == nbins

    def test_amplitude_non_negative(self):
        """Test that amplitude values are non-negative."""
        img = _make_image((60, 60), circular=True)
        _, Aout = ip.compute_psd(img)

        assert np.all(Aout >= 0)


class TestIntegratePsd:
    """Tests for the integrate_psd function."""

    def test_basic_output_shape(self):
        """Test that integrate_psd returns an array of the same length as y."""
        img = _make_image((60, 60), circular=True)
        _, y = ip.compute_psd(img)
        result = ip.integrate_psd(y, img)

        assert result.shape == y.shape

    def test_output_is_non_decreasing(self):
        """Test that the cumulative integral is non-decreasing."""
        y = np.abs(np.random.randn(20))
        img = _make_image((60, 60), circular=True)
        result = ip.integrate_psd(y, img)

        diffs = np.diff(result)
        assert np.all(diffs >= -1e-12)  # non-decreasing within numerical tolerance
