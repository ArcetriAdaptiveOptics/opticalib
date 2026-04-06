"""
Tests for opticalib.simulator.factory_functions module.
"""

import pytest
import numpy as np
from opticalib.simulator import factory_functions as ff


class TestGetAlpaoCoordsMask:
    """Test getAlpaoCoordsMask function."""

    @pytest.mark.parametrize("nacts", [88, 97, 277])
    def test_returns_coords_and_mask(self, nacts):
        """Test that getAlpaoCoordsMask returns coordinates and mask."""
        coords, mask = ff.getAlpaoCoordsMask(nacts)

        assert isinstance(coords, np.ndarray)
        assert isinstance(mask, np.ndarray)

    @pytest.mark.parametrize("nacts", [88, 97, 277])
    def test_coords_shape(self, nacts):
        """Test that coordinates have shape (2, nacts)."""
        coords, mask = ff.getAlpaoCoordsMask(nacts)

        assert coords.shape == (2, nacts)

    @pytest.mark.parametrize("nacts", [88, 97, 277])
    def test_mask_shape_default(self, nacts):
        """Test that mask has the default shape (512, 512)."""
        coords, mask = ff.getAlpaoCoordsMask(nacts)

        assert mask.shape == (512, 512)

    def test_custom_shape(self):
        """Test that custom shape is respected."""
        nacts = 88
        shape = (256, 256)
        coords, mask = ff.getAlpaoCoordsMask(nacts, shape=shape)

        assert mask.shape == shape

    def test_mask_is_boolean(self):
        """Test that mask is boolean."""
        coords, mask = ff.getAlpaoCoordsMask(88)

        assert mask.dtype == bool

    def test_mask_has_valid_pupil(self):
        """Test that mask has a valid pupil region (not all True)."""
        coords, mask = ff.getAlpaoCoordsMask(88)

        # There should be some False (valid) pixels in the mask
        assert np.any(~mask)

    def test_coords_are_within_image(self):
        """Test that actuator coordinates are within the image bounds."""
        nacts = 88
        shape = (512, 512)
        coords, mask = ff.getAlpaoCoordsMask(nacts, shape=shape)

        # Coordinates should be integer-valued and within bounds
        assert np.all(coords[0] >= 0)
        assert np.all(coords[1] >= 0)
        assert np.all(coords[0] < shape[1])
        assert np.all(coords[1] < shape[0])


class TestPixelScale:
    """Test pixel_scale function."""

    @pytest.mark.parametrize("nacts,expected", [
        (88, 102.4),
        (97, 68.26667),
    ])
    def test_known_pixel_scales(self, nacts, expected):
        """Test pixel scale for known DM configurations."""
        ps = ff.pixel_scale(nacts)
        assert abs(ps - expected) < 0.01

    @pytest.mark.parametrize("nacts", [88, 97, 277])
    def test_returns_float(self, nacts):
        """Test that pixel_scale returns a float."""
        ps = ff.pixel_scale(nacts)
        assert isinstance(ps, float)

    @pytest.mark.parametrize("nacts", [88, 97, 277])
    def test_positive_pixel_scale(self, nacts):
        """Test that pixel scale is positive."""
        ps = ff.pixel_scale(nacts)
        assert ps > 0


class TestGetPetalmirrorMask:
    """Test getPetalmirrorMask function."""

    def test_basic_creation(self):
        """Test basic creation of petal mirror mask."""
        shape = (200, 200)
        pupil_radius = 80
        mask = ff.getPetalmirrorMask(shape, pupil_radius)

        assert mask.shape == shape
        assert mask.dtype == bool

    def test_with_custom_central_segment_radius(self):
        """Test petal mirror mask with custom central segment radius."""
        shape = (200, 200)
        pupil_radius = 80
        central_radius = 15
        mask = ff.getPetalmirrorMask(shape, pupil_radius, central_radius)

        assert mask.shape == shape

    def test_default_central_segment_radius(self):
        """Test that default central segment radius is used when not provided."""
        shape = (200, 200)
        pupil_radius = 80
        # Should not raise
        mask = ff.getPetalmirrorMask(shape, pupil_radius)
        assert mask is not None

    def test_mask_has_mixed_values(self):
        """Test that the petal mask has both True and False values."""
        shape = (200, 200)
        pupil_radius = 80
        mask = ff.getPetalmirrorMask(shape, pupil_radius)

        # Should have both True (masked) and False (valid) regions
        assert np.any(mask)
        assert np.any(~mask)

    def test_different_shapes(self):
        """Test petal mask with different image shapes."""
        for shape in [(150, 150), (200, 200), (300, 300)]:
            mask = ff.getPetalmirrorMask(shape, pupil_radius=60)
            assert mask.shape == shape
            assert mask.dtype == bool
