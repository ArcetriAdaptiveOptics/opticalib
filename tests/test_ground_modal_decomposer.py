"""
Tests for opticalib.ground.modal_decomposer module.
"""
import pytest
import numpy as np
import numpy.ma as ma
from opticalib.ground import modal_decomposer as md
from arte.types.mask import CircularMask


class TestZernikeFitter:
    """Test ZernikeFitter class."""

    def test_init_no_mask(self):
        """Test ZernikeFitter initialization without mask."""
        fitter = md.ZernikeFitter()
        
        assert fitter._fit_mask is None
        assert fitter.auxmask is None
        assert fitter._mgen is None

    def test_init_with_mask(self, circular_mask):
        """Test ZernikeFitter initialization with mask."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None
        assert fitter._mgen is not None

    def test_init_with_masked_array(self, sample_image):
        """Test ZernikeFitter initialization with masked array."""
        fitter = md.ZernikeFitter(fit_mask=sample_image)
        
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None
        assert fitter._mgen is not None

    def test_set_fit_mask(self):
        """Test setting fit mask."""
        fitter = md.ZernikeFitter()
        
        # Create a circular mask
        size = 100
        mask = np.zeros((size, size), dtype=bool)
        y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
        radius = size / 2 - 5
        mask = (x**2 + y**2) > radius**2
        
        fitter.setFitMask(mask)
        
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None
        assert fitter._mgen is not None

    def test_set_fit_mask_with_circular_mask(self):
        """Test setting fit mask with CircularMask object."""
        fitter = md.ZernikeFitter()
        
        cmask = CircularMask((100, 100))
        fitter.setFitMask(cmask)
        
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None
        assert fitter._mgen is not None

    def test_fit_mask_property(self, circular_mask):
        """Test fitMask property."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        mask = fitter.fitMask
        assert mask is not None
        assert isinstance(mask, np.ndarray)

    # FIXME
    def test_fit_basic(self, sample_image):
        """Test basic fitting of Zernike modes."""
        fitter = md.ZernikeFitter(fit_mask=sample_image.mask)

        modes = [1, 2, 3, 4]
        coeffs, mat = fitter.fit(sample_image, modes)

        assert coeffs is not None
        assert mat is not None
        assert len(coeffs) == len(modes)
        assert mat.shape[0] == np.sum(~sample_image.mask)
        assert mat.shape[1] == len(modes)

    def test_fit_without_mask(self, sample_image):
        """Test fitting without pre-set mask."""
        fitter = md.ZernikeFitter()
        
        modes = [1, 2, 3]
        coeffs, mat = fitter.fit(sample_image, modes)
        
        assert coeffs is not None
        assert mat is not None
        assert len(coeffs) == len(modes)

    # FIXME
    def test_remove_zernike_default(self, sample_image):
        """Test removing default Zernike modes (piston, tip, tilt)."""
        fitter = md.ZernikeFitter(fit_mask=sample_image.mask)
        
        filtered = fitter.removeZernike(sample_image)
        
        assert filtered is not None
        assert isinstance(filtered, ma.MaskedArray)
        assert filtered.shape == sample_image.shape

    # FIXME
    def test_remove_zernike_custom_modes(self, sample_image):
        """Test removing custom Zernike modes."""
        fitter = md.ZernikeFitter(fit_mask=sample_image.mask)
        
        filtered = fitter.removeZernike(sample_image, zernike_index_vector=[1, 2, 3, 4])
        
        assert filtered is not None
        assert isinstance(filtered, ma.MaskedArray)
        assert filtered.shape == sample_image.shape

    # FIXME
    def test_make_surface_with_image(self, sample_image):
        """Test making surface from image."""
        fitter = md.ZernikeFitter(fit_mask=sample_image.mask)
        
        modes = [1, 2, 3]
        surface = fitter.makeSurface(modes, image=sample_image)
        
        assert surface is not None
        assert isinstance(surface, ma.MaskedArray)
        assert surface.shape == sample_image.shape

    def test_make_surface_without_image(self, circular_mask):
        """Test making surface without image (using fit mask)."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        modes = [1]
        surface = fitter.makeSurface(modes)
        
        assert surface is not None
        assert isinstance(surface, np.ndarray) or isinstance(surface, ma.MaskedArray)

    def test_make_surface_multiple_modes(self, circular_mask):
        """Test making surface with multiple modes."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        modes = [1, 2, 3]
        surface = fitter.makeSurface(modes)
        
        assert surface is not None

    def test_make_surface_no_image_no_mask(self):
        """Test making surface without image or mask raises error."""
        fitter = md.ZernikeFitter()
        
        with pytest.raises(ValueError):
            fitter.makeSurface([1])

    # FIXME
    def test_filter_modes(self, sample_image):
        """Test filtering modes from image."""
        fitter = md.ZernikeFitter(fit_mask=sample_image.mask)
        
        modes = [1, 2, 3]
        filtered = fitter.filterModes(sample_image, modes)
        
        assert filtered is not None
        assert isinstance(filtered, ma.MaskedArray)
        assert filtered.shape == sample_image.shape

    def test_fit_on_roi_global(self, sample_image):
        """Test fitting on ROIs with global mode."""
        fitter = md.ZernikeFitter(fit_mask=sample_image)
        
        modes = [1, 2, 3]
        coeffs = fitter.fitOnRoi(sample_image, modes2fit=modes, mode='global')
        
        assert coeffs is not None
        assert len(coeffs) == len(modes)

    def test_fit_on_roi_local(self, sample_image):
        """Test fitting on ROIs with local mode."""
        fitter = md.ZernikeFitter(fit_mask=sample_image)
        
        modes = [1, 2, 3]
        coeffs = fitter.fitOnRoi(sample_image, modes2fit=modes, mode='local')
        
        assert coeffs is not None
        assert len(coeffs.shape) == 2  # Should be 2D array (n_rois, n_modes)

    def test_fit_on_roi_invalid_mode(self, sample_image):
        """Test fitting on ROIs with invalid mode raises error."""
        fitter = md.ZernikeFitter(fit_mask=sample_image)
        
        with pytest.raises(ValueError, match="mode must be 'global' or 'local'"):
            fitter.fitOnRoi(sample_image, modes2fit=[1, 2, 3], mode='invalid')

    def test_no_mask_context_manager(self, sample_image):
        """Test no_mask context manager."""
        fitter = md.ZernikeFitter(fit_mask=sample_image)
        
        # Store original values
        original_mask = fitter._fit_mask
        original_mgen = fitter._mgen
        
        with fitter.no_mask():
            assert fitter._fit_mask is None
            assert fitter._mgen is None
            assert fitter.auxmask is None
            
            # Should still be able to fit
            coeffs, mat = fitter.fit(sample_image, [1, 2, 3])
            assert coeffs is not None
        
        # Values should be restored
        assert fitter._fit_mask is original_mask
        assert fitter._mgen is original_mgen

    def test_get_mode_from_generator(self, circular_mask):
        """Test getting mode from generator."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        mode = fitter._get_mode_from_generator(1)
        
        assert mode is not None
        assert isinstance(mode, np.ndarray) or isinstance(mode, ma.MaskedArray)

    def test_make_sure_on_cpu(self, sample_image):
        """Test _make_sure_on_cpu method."""
        fitter = md.ZernikeFitter()
        
        result = fitter._make_sure_on_cpu(sample_image)
        
        assert result is not None
        assert isinstance(result, ma.MaskedArray)


class TestKLFitter:
    """Test KLFitter class."""

    def test_init_no_mask(self):
        """Test KLFitter initialization without mask."""
        fitter = md.KLFitter(nKLModes=10)
        
        assert fitter.nModes == 10
        assert fitter._fit_mask is None
        assert fitter.auxmask is None
        assert fitter._mgen is None

    def test_init_with_mask(self, circular_mask):
        """Test KLFitter initialization with mask."""
        fitter = md.KLFitter(nKLModes=10, fit_mask=circular_mask)
        
        assert fitter.nModes == 10
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None
        assert fitter._mgen is not None

    def test_fit_basic(self, sample_image):
        """Test basic fitting of KL modes."""
        fitter = md.KLFitter(nKLModes=10, fit_mask=sample_image)
        
        modes = [0, 1, 2]
        coeffs, mat = fitter.fit(sample_image, modes)
        
        assert coeffs is not None
        assert mat is not None
        assert len(coeffs) == len(modes)

    def test_get_mode_from_generator(self, circular_mask):
        """Test getting KL mode from generator."""
        fitter = md.KLFitter(nKLModes=10, fit_mask=circular_mask)
        
        mode = fitter._get_mode_from_generator(0)
        
        assert mode is not None
        assert isinstance(mode, np.ndarray) or isinstance(mode, ma.MaskedArray)

    # FIXME
    def test_make_surface(self, sample_image):
        """Test making KL surface."""
        fitter = md.KLFitter(nKLModes=10, fit_mask=sample_image.mask)
        
        modes = [0, 1]
        surface = fitter.makeSurface(modes, image=sample_image)
        
        assert surface is not None
        assert isinstance(surface, ma.MaskedArray)
        assert surface.shape == sample_image.shape

    # FIXME
    def test_filter_modes(self, sample_image):
        """Test filtering KL modes."""
        fitter = md.KLFitter(nKLModes=10, fit_mask=sample_image.mask)
        
        modes = [0, 1, 2]
        filtered = fitter.filterModes(sample_image, modes)
        
        assert filtered is not None
        assert isinstance(filtered, ma.MaskedArray)
        assert filtered.shape == sample_image.shape


class TestRBFitter:
    """Test RBFitter class."""

    def test_init_no_mask(self):
        """Test RBFitter initialization without mask."""
        fitter = md.RBFitter()
        
        assert fitter.rbfFunction == "TPS_RBF"
        assert fitter._coordinates is None
        assert fitter._eps == 1.0
        assert fitter._fit_mask is None

    def test_init_with_mask(self, circular_mask):
        """Test RBFitter initialization with mask."""
        fitter = md.RBFitter(fit_mask=circular_mask)
        
        assert fitter._fit_mask is not None
        assert fitter.auxmask is not None

    def test_init_with_custom_params(self):
        """Test RBFitter initialization with custom parameters."""
        fitter = md.RBFitter(
            rbfFunction="gaussian",
            eps=2.0
        )
        
        assert fitter.rbfFunction == "gaussian"
        assert fitter._eps == 2.0

    def test_init_with_coordinates(self, circular_mask):
        """Test RBFitter initialization with coordinates."""
        coords = np.array([[10, 20], [30, 40], [50, 60]])
        fitter = md.RBFitter(coords=coords, fit_mask=circular_mask)
        
        assert fitter._coordinates is not None
        assert np.array_equal(fitter._coordinates, coords)

    def test_fit_basic(self, sample_image):
        """Test basic fitting of RBF modes."""
        fitter = md.RBFitter(fit_mask=sample_image)
        
        # RBF modes are typically indexed from 0
        modes = [0, 1, 2]
        coeffs, mat = fitter.fit(sample_image, modes)
        
        assert coeffs is not None
        assert mat is not None
        assert len(coeffs) == len(modes)

    def test_get_mode_from_generator(self, circular_mask):
        """Test getting RBF mode from generator."""
        fitter = md.RBFitter(fit_mask=circular_mask)
        
        mode = fitter._get_mode_from_generator(0)
        
        assert mode is not None
        assert isinstance(mode, np.ndarray) or isinstance(mode, ma.MaskedArray)

    # FIXME
    def test_make_surface(self, sample_image):
        """Test making RBF surface."""
        fitter = md.RBFitter(fit_mask=sample_image.mask)
        
        modes = [0, 1]
        surface = fitter.makeSurface(modes, image=sample_image)
        
        assert surface is not None
        assert isinstance(surface, ma.MaskedArray)
        assert surface.shape == sample_image.shape

    # FIXME
    def test_filter_modes(self, sample_image):
        """Test filtering RBF modes."""
        fitter = md.RBFitter(fit_mask=sample_image.mask)
        
        modes = [0, 1, 2]
        filtered = fitter.filterModes(sample_image, modes)
        
        assert filtered is not None
        assert isinstance(filtered, ma.MaskedArray)
        assert filtered.shape == sample_image.shape


class TestModeFitterShared:
    """Test shared functionality from _ModeFitter base class."""

    def test_create_fit_mask_from_img(self, sample_image):
        """Test creating fit mask from image."""
        fitter = md.ZernikeFitter()
        
        mgen = fitter._create_fit_mask_from_img(sample_image)
        
        assert mgen is not None

    def test_create_fit_mask_from_img_invalid(self):
        """Test creating fit mask from invalid image raises error."""
        fitter = md.ZernikeFitter()
        
        # Create invalid input (not a masked array and not convertible)
        invalid_img = "not an image"
        
        with pytest.raises(ValueError):
            fitter._create_fit_mask_from_img(invalid_img)

    def test_temporary_mgen_from_image(self, sample_image):
        """Test temporary mgen context manager."""
        fitter = md.ZernikeFitter()
        
        # Initially no mgen
        assert fitter._mgen is None
        
        # Use context manager
        with fitter._temporary_mgen_from_image(sample_image) as (img, was_temp):
            assert was_temp is True
            assert fitter._mgen is not None
            assert img is not None
        
        # Should be restored to None
        assert fitter._mgen is None

    def test_temporary_mgen_with_existing(self, circular_mask, sample_image):
        """Test temporary mgen context manager with existing mgen."""
        fitter = md.ZernikeFitter(fit_mask=circular_mask)
        
        original_mgen = fitter._mgen
        assert original_mgen is not None
        
        # Use context manager
        with fitter._temporary_mgen_from_image(sample_image) as (img, was_temp):
            assert was_temp is False  # Should not create temporary
            assert fitter._mgen is original_mgen
        
        # Should still be original
        assert fitter._mgen is original_mgen

