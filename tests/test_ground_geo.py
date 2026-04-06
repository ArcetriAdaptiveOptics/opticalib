"""
Tests for opticalib.ground.geo module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib.ground import geo


class TestQpupil:
    """Test qpupil function."""

    def test_basic_mask(self):
        """Test qpupil with a basic binary mask."""
        mask = np.zeros((100, 100), dtype=int)
        mask[30:70, 30:70] = 1  # square active region

        x0, y0, r, xx, yy = geo.qpupil(mask)

        assert isinstance(x0, float)
        assert isinstance(y0, float)
        assert r > 0
        assert xx.shape == mask.shape
        assert yy.shape == mask.shape

    def test_normalized_coordinates(self):
        """Test that qpupil normalizes coordinates to [-1, 1]."""
        mask = np.zeros((100, 100), dtype=int)
        mask[20:80, 20:80] = 1

        x0, y0, r, xx, yy = geo.qpupil(mask)

        idx = np.where(mask == 1)
        # Normalized coordinates in the active region should be in [-1, 1]
        assert np.max(np.abs(xx[idx])) <= 1.0 + 1e-10
        assert np.max(np.abs(yy[idx])) <= 1.0 + 1e-10

    def test_nocircle_mode(self):
        """Test qpupil with nocircle=1 (no normalization)."""
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1

        x0, y0, r, xx, yy = geo.qpupil(mask, nocircle=1)

        # With nocircle=1, x0, y0, r should be zeros
        assert x0 == 0
        assert y0 == 0
        assert r == 0
        assert xx.shape == mask.shape
        assert yy.shape == mask.shape

    def test_output_grid_shape(self):
        """Test that output coordinate grids have same shape as input mask."""
        shape = (64, 64)
        mask = np.zeros(shape, dtype=int)
        mask[16:48, 16:48] = 1

        x0, y0, r, xx, yy = geo.qpupil(mask)

        assert xx.shape == shape
        assert yy.shape == shape

    def test_center_estimation(self):
        """Test that the center is estimated correctly for a centered region."""
        size = 100
        mask = np.zeros((size, size), dtype=int)
        # Create a centered square region
        margin = 20
        mask[margin : size - margin, margin : size - margin] = 1

        x0, y0, r, xx, yy = geo.qpupil(mask)

        # Center should be approximately at size/2
        assert abs(x0 - size / 2) < 2.0
        assert abs(y0 - size / 2) < 2.0


class TestDrawMask:
    """Test draw_mask function."""

    def test_basic_circular_mask(self):
        """Test draw_mask creates a circular mask."""
        img = np.zeros((100, 100))
        cx, cy, r = 50, 50, 30

        result = geo.draw_mask(img, cx, cy, r)

        assert result.shape == img.shape
        # Center pixel should be 1 (inside circle)
        assert result[cx, cy] == 1
        # Corner should be 0 (outside circle)
        assert result[0, 0] == 0

    def test_circular_mask_radius(self):
        """Test that mask area is approximately pi*r^2."""
        img = np.zeros((200, 200))
        cx, cy, r = 100, 100, 40

        result = geo.draw_mask(img, cx, cy, r)

        area = np.sum(result == 1)
        expected_area = np.pi * r**2
        # Allow 5% tolerance
        assert abs(area - expected_area) / expected_area < 0.05

    def test_out_mode_zero(self):
        """Test draw_mask with out=0 (fill inside with 1)."""
        img = np.zeros((100, 100))
        result = geo.draw_mask(img, 50, 50, 30, out=0)

        # Inside the circle should be 1
        assert result[50, 50] == 1

    def test_out_mode_one(self):
        """Test draw_mask with out=1 (fill inside with 0)."""
        img = np.ones((100, 100))
        result = geo.draw_mask(img, 50, 50, 30, out=1)

        # Inside the circle should be 0
        assert result[50, 50] == 0

    def test_elliptical_mask(self):
        """Test draw_mask with elliptical radius (r as list of 2)."""
        img = np.zeros((100, 100))
        cx, cy = 50, 50
        r = [30, 20]  # elliptical radii

        result = geo.draw_mask(img, cx, cy, r)

        assert result.shape == img.shape
        # Center pixel should be 1
        assert result[cx, cy] == 1

    def test_off_center_mask(self):
        """Test draw_mask with off-center circle."""
        img = np.zeros((100, 100))
        cx, cy, r = 20, 20, 10

        result = geo.draw_mask(img, cx, cy, r)

        # The center of the circle should be inside
        assert result[cx, cy] == 1
        # Far corner should be outside
        assert result[99, 99] == 0


class TestDrawCircularMask:
    """Test draw_circular_mask function."""

    def test_with_masked_array(self):
        """Test draw_circular_mask with a masked array."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        mask[20:80, 20:80] = False
        img = ma.masked_array(data, mask=mask)

        result = geo.draw_circular_mask(img, radius=20)

        assert result is not None
        assert result.shape == img.shape

    def test_with_plain_array(self):
        """Test draw_circular_mask with a plain numpy array used as mask."""
        mask = np.zeros((100, 100), dtype=int)
        mask[20:80, 20:80] = 1

        result = geo.draw_circular_mask(mask, radius=20)

        assert result is not None
        assert result.shape == mask.shape

    def test_result_is_array(self):
        """Test that the result is a numpy array."""
        data = np.random.randn(100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        img = ma.masked_array(data, mask=mask)

        result = geo.draw_circular_mask(img, radius=15)

        assert isinstance(result, np.ndarray)
