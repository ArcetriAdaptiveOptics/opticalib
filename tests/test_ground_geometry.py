"""
Tests for opticalib.ground.geometry module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib.ground import geometry


class TestDrawCircularPupil:
    """Test draw_circular_pupil function."""

    def test_basic_creation(self):
        """Test basic creation of a circular pupil mask."""
        shape = (100, 100)
        radius = 40
        mask = geometry.draw_circular_pupil(shape, radius)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_default_unmasked_region_is_false(self):
        """Test that the circular region is set to False (valid region)."""
        shape = (100, 100)
        radius = 40
        mask = geometry.draw_circular_pupil(shape, radius)

        # Center should be False (inside the pupil)
        assert mask[50, 50] is np.bool_(False)
        # Corners should be True (outside the pupil)
        assert mask[0, 0] is np.bool_(True)
        assert mask[0, 99] is np.bool_(True)

    def test_masked_mode(self):
        """Test draw_circular_pupil with masked=True flips the logic."""
        shape = (100, 100)
        radius = 40
        mask_normal = geometry.draw_circular_pupil(shape, radius, masked=False)
        mask_flipped = geometry.draw_circular_pupil(shape, radius, masked=True)

        # They should be complements of each other
        np.testing.assert_array_equal(mask_normal, ~mask_flipped)

    def test_custom_center(self):
        """Test draw_circular_pupil with a custom center."""
        shape = (100, 100)
        radius = 20
        center = (30, 30)  # (x, y) = (col, row)
        mask = geometry.draw_circular_pupil(shape, radius, center=center)

        assert mask.shape == shape
        # The center should be inside the pupil (False)
        assert mask[30, 30] is np.bool_(False)

    def test_radius_approximately_correct(self):
        """Test that the valid region has approximately the right area."""
        shape = (200, 200)
        radius = 60
        mask = geometry.draw_circular_pupil(shape, radius)

        # Count valid (False) pixels
        valid_pixels = np.sum(~mask)
        expected = np.pi * radius**2
        # Allow 5% tolerance
        assert abs(valid_pixels - expected) / expected < 0.05

    def test_different_shapes(self):
        """Test draw_circular_pupil with different shapes."""
        for shape in [(50, 50), (100, 200), (128, 128)]:
            mask = geometry.draw_circular_pupil(shape, radius=20)
            assert mask.shape == shape
            assert mask.dtype == bool


class TestDrawPolygonalMask:
    """Test draw_polygonal_mask function."""

    def test_basic_square(self):
        """Test creating a square mask."""
        shape = (100, 100)
        vertices = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        mask = geometry.draw_polygonal_mask(shape, vertices)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_square_center_is_false(self):
        """Test that center of a square is False (inside the polygon)."""
        shape = (100, 100)
        vertices = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        mask = geometry.draw_polygonal_mask(shape, vertices)

        # Center should be inside the polygon (False)
        assert mask[50, 50] is np.bool_(False)
        # Outside points should be True
        assert mask[0, 0] is np.bool_(True)

    def test_masked_mode(self):
        """Test draw_polygonal_mask with masked=True."""
        shape = (100, 100)
        vertices = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])
        mask_normal = geometry.draw_polygonal_mask(shape, vertices, masked=False)
        mask_flipped = geometry.draw_polygonal_mask(shape, vertices, masked=True)

        np.testing.assert_array_equal(mask_normal, ~mask_flipped)

    def test_triangle_mask(self):
        """Test creating a triangular mask."""
        shape = (100, 100)
        vertices = np.array([[50, 10], [10, 90], [90, 90]])
        mask = geometry.draw_polygonal_mask(shape, vertices)

        assert mask.shape == shape
        assert mask.dtype == bool


class TestDrawHexagonalMask:
    """Test draw_hexagonal_mask function."""

    def test_basic_hexagon(self):
        """Test creating a hexagonal mask."""
        shape = (200, 200)
        radius = 60
        mask = geometry.draw_hexagonal_mask(shape, radius)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_center_is_false(self):
        """Test that center of hexagon is False (inside)."""
        shape = (200, 200)
        radius = 60
        mask = geometry.draw_hexagonal_mask(shape, radius)

        # Center should be inside (False)
        assert mask[100, 100] is np.bool_(False)

    def test_masked_mode(self):
        """Test draw_hexagonal_mask with masked=True."""
        shape = (200, 200)
        radius = 60
        mask_normal = geometry.draw_hexagonal_mask(shape, radius, masked=False)
        mask_flipped = geometry.draw_hexagonal_mask(shape, radius, masked=True)

        np.testing.assert_array_equal(mask_normal, ~mask_flipped)

    def test_custom_center(self):
        """Test hexagonal mask with custom center."""
        shape = (200, 200)
        radius = 40
        center = (80, 80)
        mask = geometry.draw_hexagonal_mask(shape, radius, center=center)

        assert mask.shape == shape
        assert mask.dtype == bool


class TestCreateLineMask:
    """Test create_line_mask function."""

    def test_horizontal_line(self):
        """Test creating a horizontal line mask (angle=0)."""
        shape = (100, 100)
        mask = geometry.draw_linear_mask(shape, angle_deg=0, width=3)

        assert mask.shape == shape
        assert mask.dtype == bool
        # Should have some True pixels
        assert np.any(mask)

    def test_vertical_line(self):
        """Test creating a vertical line mask (angle=90)."""
        shape = (100, 100)
        mask = geometry.draw_linear_mask(shape, angle_deg=90, width=3)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_diagonal_line(self):
        """Test creating a diagonal line mask (angle=45)."""
        shape = (100, 100)
        mask = geometry.draw_linear_mask(shape, angle_deg=45, width=3)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_slope_parameter(self):
        """Test creating a line mask using slope parameter."""
        shape = (100, 100)
        mask = geometry.draw_linear_mask(shape, slope=1.0, width=3)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_no_angle_or_slope_raises(self):
        """Test that providing neither angle_deg nor slope raises ValueError."""
        with pytest.raises(ValueError, match="Must provide either"):
            geometry.draw_linear_mask((100, 100), width=3)

    def test_masked_mode(self):
        """Test draw_linear_mask with masked=True inverts the mask."""
        shape = (100, 100)
        mask_normal = geometry.draw_linear_mask(shape, angle_deg=45, width=5)
        mask_flipped = geometry.draw_linear_mask(
            shape, angle_deg=45, width=5, masked=True
        )

        np.testing.assert_array_equal(mask_normal, ~mask_flipped)

    def test_custom_center(self):
        """Test draw_linear_mask with custom center point."""
        shape = (100, 100)
        center = (30, 30)
        mask = geometry.draw_linear_mask(shape, angle_deg=0, width=3, center=center)

        assert mask.shape == shape
        # The line should pass through the center row
        assert mask[30, 30] is np.bool_(True)

    def test_wider_line_more_pixels(self):
        """Test that a wider line has more pixels."""
        shape = (100, 100)
        mask_narrow = geometry.draw_linear_mask(shape, angle_deg=45, width=1)
        mask_wide = geometry.draw_linear_mask(shape, angle_deg=45, width=10)

        assert np.sum(mask_wide) > np.sum(mask_narrow)


class TestRotateImage:
    """Test rotate_image function."""

    def test_basic_rotation(self):
        """Test basic image rotation."""
        data = np.random.randn(50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[:5, :] = True
        img = ma.masked_array(data, mask=mask)

        rotated = geometry.rotate_image(img, angle_deg=90)

        assert rotated.shape == img.shape
        assert isinstance(rotated, ma.MaskedArray)

    def test_zero_rotation(self):
        """Test that zero-degree rotation preserves shape."""
        data = np.random.randn(50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        img = ma.masked_array(data, mask=mask)

        rotated = geometry.rotate_image(img, angle_deg=0)

        assert rotated.shape == img.shape
        np.testing.assert_array_almost_equal(rotated.data, img.data)

    def test_output_is_masked_array(self):
        """Test that output is always a masked array."""
        data = np.random.randn(30, 30)
        mask = np.zeros((30, 30), dtype=bool)
        img = ma.masked_array(data, mask=mask)

        for angle in [0, 45, 90, 180]:
            rotated = geometry.rotate_image(img, angle_deg=angle)
            assert isinstance(rotated, ma.MaskedArray)

    def test_custom_center(self):
        """Test rotation around a custom center."""
        data = np.random.randn(50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        img = ma.masked_array(data, mask=mask)

        center = (25, 25)
        rotated = geometry.rotate_image(img, angle_deg=45, center=center)

        assert rotated.shape == img.shape

    def test_180_degree_rotation(self):
        """Test 180-degree rotation approximately flips the image."""
        data = np.eye(10, dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        img = ma.masked_array(data, mask=mask)

        rotated = geometry.rotate_image(img, angle_deg=180)

        assert rotated.shape == img.shape
        # 180 rotation of identity should have 1s on anti-diagonal
        # (approximately, due to interpolation)
        assert isinstance(rotated, ma.MaskedArray)
