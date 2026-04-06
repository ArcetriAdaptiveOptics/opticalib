"""
Tests for opticalib.analyzer.timeseries module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib.core.fitsarray import fits_array
from opticalib.analyzer import timeseries


def _make_image(shape=(10, 10)):
    """Create a simple FitsMaskedArray.

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape.

    Returns
    -------
    FitsMaskedArray
    """
    data = np.random.randn(*shape)
    mask = np.zeros(shape, dtype=bool)
    return fits_array(ma.masked_array(data, mask=mask))


def _make_cube(shape=(10, 10, 5)):
    """Create a masked array cube.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Cube dimensions.

    Returns
    -------
    np.ma.MaskedArray
    """
    data = np.random.randn(*shape)
    mask = np.zeros(shape, dtype=bool)
    return ma.masked_array(data, mask=mask)


class TestAverageFrames:
    """Tests for the averageFrames function."""

    def test_average_from_image_list(self):
        """Test averaging a list of ImageData objects."""
        imgs = [_make_image((10, 10)) for _ in range(5)]
        result = timeseries.averageFrames(imgs)

        assert result.shape == (10, 10)
        assert isinstance(result, np.ma.MaskedArray)

    def test_average_from_cube(self):
        """Test averaging from a 3D masked array cube."""
        cube = _make_cube((10, 10, 5))
        result = timeseries.averageFrames(cube)

        assert result.shape == (10, 10)

    def test_average_values_correct(self):
        """Test that the averaged values are correct."""
        data = np.ones((5, 5))
        mask = np.zeros((5, 5), dtype=bool)
        imgs = [
            fits_array(ma.masked_array(data * i, mask=mask))
            for i in [1.0, 2.0, 3.0]
        ]
        result = timeseries.averageFrames(imgs)

        expected = np.full((5, 5), 2.0)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_average_with_file_selector(self):
        """Test averaging with a file_selector list."""
        imgs = [_make_image((8, 8)) for _ in range(6)]
        result = timeseries.averageFrames(imgs, file_selector=[0, 2, 4])

        assert result.shape == (8, 8)

    def test_average_with_first_last(self):
        """Test averaging a sub-range of images using first and last."""
        imgs = [_make_image((8, 8)) for _ in range(10)]
        result = timeseries.averageFrames(imgs, first=2, last=5)

        assert result.shape == (8, 8)

    def test_average_from_cube_sliced(self):
        """Test averaging a slice of a cube using first and last."""
        cube = _make_cube((10, 10, 10))
        result = timeseries.averageFrames(cube, first=0, last=5)

        assert result.shape == (10, 10)

    def test_average_with_thresh_flag(self):
        """Test averaging with the threshold flag enabled."""
        imgs = [_make_image((8, 8)) for _ in range(4)]
        result = timeseries.averageFrames(imgs, thresh=True)

        assert result.shape == (8, 8)
        assert isinstance(result, np.ma.MaskedArray)


class TestRunningMean:
    """Tests for the runningMean function."""

    def test_basic_running_mean(self):
        """Test the running mean of a simple increasing sequence."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = timeseries.runningMean(vec, 3)

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_output_length(self):
        """Test that output length equals len(vec) - npoints + 1."""
        vec = np.random.randn(20)
        npoints = 5
        result = timeseries.runningMean(vec, npoints)

        assert len(result) == len(vec) - npoints + 1

    def test_window_of_one_returns_same(self):
        """Test that a window of 1 returns the input unchanged."""
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        result = timeseries.runningMean(vec, 1)

        np.testing.assert_allclose(result, vec)

    def test_constant_array(self):
        """Test that running mean of a constant array is the same constant."""
        val = 7.5
        vec = np.full(15, val)
        result = timeseries.runningMean(vec, 4)

        np.testing.assert_allclose(result, val)

    def test_output_dtype_is_float(self):
        """Test that output is a float array."""
        vec = np.array([1, 2, 3, 4, 5], dtype=int)
        result = timeseries.runningMean(vec, 2)

        assert np.issubdtype(result.dtype, np.floating)


class TestStructFunc:
    """Tests for the structfunc function."""

    def test_basic_output_shape(self):
        """Test that the output shape matches the number of gaps."""
        vect = np.arange(1.0, 21.0)
        gapvect = np.array([1, 2])
        result = timeseries.structfunc(vect, gapvect)

        assert result.shape == (len(gapvect),)

    def test_output_is_non_negative(self):
        """Test that all structure function values are non-negative."""
        vect = np.random.randn(50)
        gapvect = np.array([1, 2, 3])
        result = timeseries.structfunc(vect, gapvect)

        assert np.all(result >= 0)

    def test_constant_signal_gives_zero(self):
        """Test that a constant signal produces a zero structure function."""
        vect = np.full(20, 3.14)
        gapvect = np.array([1, 2])
        result = timeseries.structfunc(vect, gapvect)

        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_single_gap(self):
        """Test with a single gap value."""
        vect = np.arange(1.0, 21.0)
        gapvect = np.array([2])
        result = timeseries.structfunc(vect, gapvect)

        assert result.shape == (1,)
        assert result[0] >= 0
