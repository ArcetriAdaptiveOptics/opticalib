"""
Tests for opticalib.ground.roi module.
"""
import pytest
import numpy as np
import numpy.ma as ma
from opticalib.ground import roi


class TestRoiGenerator:
    """Test roiGenerator function."""

    def test_roi_generator_basic(self, sample_image):
        """Test roiGenerator with basic image."""
        roi_list = roi.roiGenerator(sample_image)
        
        assert isinstance(roi_list, list)
        # Should return at least one ROI if image has valid pixels
        if np.sum(~sample_image.mask) > 0:
            assert len(roi_list) > 0
            for r in roi_list:
                assert r.shape == sample_image.shape
                assert isinstance(r, np.ndarray)

    def test_roi_generator_fully_masked(self):
        """Test roiGenerator with fully masked image."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)
        
        roi_list = roi.roiGenerator(masked_img)
        assert len(roi_list) == 0

    def test_roi_generator_multiple_rois(self):
        """Test roiGenerator with multiple disconnected regions."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create two disconnected regions
        mask[20:30, 20:30] = False
        mask[70:80, 70:80] = False
        masked_img = ma.masked_array(data, mask=mask)
        
        roi_list = roi.roiGenerator(masked_img)
        assert len(roi_list) >= 2

    def test_roi_generator_small_rois_filtered(self):
        """Test that small ROIs (< 100 pixels) are filtered out."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create a small region (< 100 pixels)
        mask[45:50, 45:50] = False  # 5x5 = 25 pixels
        masked_img = ma.masked_array(data, mask=mask)
        
        roi_list = roi.roiGenerator(masked_img)
        # Small ROI should be filtered out
        assert len(roi_list) == 0


class TestImgCut:
    """Test imgCut function."""

    def test_img_cut_basic(self, sample_image):
        """Test imgCut with basic image."""
        cut_img = roi.imgCut(sample_image)
        
        assert isinstance(cut_img, ma.MaskedArray)
        # Cut image should be smaller or equal to original
        assert cut_img.shape[0] <= sample_image.shape[0]
        assert cut_img.shape[1] <= sample_image.shape[1]

    def test_img_cut_fully_masked(self):
        """Test imgCut with fully masked image."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)
        
        cut_img = roi.imgCut(masked_img)
        # Should return original image if no finite pixels
        assert cut_img.shape == masked_img.shape

    def test_img_cut_centered_region(self):
        """Test imgCut with centered valid region."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create a centered valid region
        mask[40:60, 40:60] = False
        masked_img = ma.masked_array(data, mask=mask)
        
        cut_img = roi.imgCut(masked_img)
        # Should cut to approximately 20x20 region
        assert cut_img.shape[0] <= 25  # Allow some margin
        assert cut_img.shape[1] <= 25

    def test_img_cut_no_nan(self):
        """Test imgCut with image containing no NaN values."""
        data = np.random.randn(100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)
        
        cut_img = roi.imgCut(masked_img)
        # Should return full image or slightly trimmed
        assert cut_img.shape[0] >= 90
        assert cut_img.shape[1] >= 90


class TestCubeMasterMask:
    """Test cubeMasterMask function."""

    def test_cube_master_mask_basic(self, sample_cube):
        """Test cubeMasterMask with basic cube."""
        master_mask = roi.cubeMasterMask(sample_cube)
        
        assert master_mask.shape == sample_cube.shape[:2]
        assert isinstance(master_mask, np.ndarray)
        assert master_mask.dtype == bool

    def test_cube_master_mask_combines_masks(self):
        """Test that master mask combines all frame masks."""
        # Create cube with different masks per frame
        data = np.random.randn(50, 50, 3).astype(np.float32)
        masks = [
            np.zeros((50, 50), dtype=bool),
            np.zeros((50, 50), dtype=bool),
            np.zeros((50, 50), dtype=bool)
        ]
        masks[0][:10, :10] = True
        masks[1][:15, :15] = True
        masks[2][:20, :20] = True
        
        cube = ma.masked_array(
            data,
            mask=np.stack(masks, axis=2)
        )
        
        master_mask = roi.cubeMasterMask(cube)
        # Master mask should include all masked regions
        assert np.all(master_mask[:10, :10])  # All frames mask this
        assert np.all(master_mask[:15, :15])  # At least one frame masks this

    def test_cube_master_mask_single_frame(self):
        """Test cubeMasterMask with single frame cube."""
        data = np.random.randn(50, 50, 1).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        mask[:10, :10] = True
        cube = ma.masked_array(
            data,
            mask=np.broadcast_to(mask[..., np.newaxis], data.shape)
        )
        
        master_mask = roi.cubeMasterMask(cube)
        np.testing.assert_array_equal(master_mask, mask)


# class TestRemapOnNewMask:
#     """Test remap_on_new_mask function."""

#     def test_remap_on_new_mask_basic(self):
#         """Test remap_on_new_mask with basic masks."""
#         # Create old and new masks
#         old_mask = np.zeros((10, 10), dtype=bool)
#         old_mask[:5, :] = True  # Top half masked
#         new_mask = np.zeros((10, 10), dtype=bool)
#         new_mask[:, :5] = True  # Left half masked
        
#         # Create data on old mask
#         old_valid = np.sum(~old_mask)  # 50 pixels
#         data = np.random.randn(old_valid, 5)
        
#         remapped = roi.remap_on_new_mask(data, old_mask, new_mask)
        
#         new_valid = np.sum(~new_mask)  # 50 pixels
#         assert remapped.shape == (new_valid, 5)

#     def test_remap_on_new_mask_same_mask(self):
#         """Test remap_on_new_mask with same mask."""
#         mask = np.zeros((10, 10), dtype=bool)
#         mask[:5, :] = True
#         valid = np.sum(~mask)
        
#         data = np.random.randn(valid, 3)
#         remapped = roi.remap_on_new_mask(data, mask, mask)
        
#         np.testing.assert_array_almost_equal(remapped, data)

#     def test_remap_on_new_mask_transpose(self):
#         """Test remap_on_new_mask with transposed data."""
#         old_mask = np.zeros((10, 10), dtype=bool)
#         old_mask[:5, :] = True
#         new_mask = np.zeros((10, 10), dtype=bool)
#         new_mask[:, :5] = True
        
#         old_valid = np.sum(~old_mask)
#         # Data is transposed (N, valid_pixels)
#         data = np.random.randn(5, old_valid)
        
#         remapped = roi.remap_on_new_mask(data, old_mask, new_mask)
#         new_valid = np.sum(~new_mask)
#         assert remapped.shape == (5, new_valid)

#     def test_remap_on_new_mask_error_new_larger(self):
#         """Test remap_on_new_mask raises error when new mask has more valid pixels."""
#         old_mask = np.zeros((10, 10), dtype=bool)
#         old_mask[:5, :] = True  # 50 valid pixels
#         new_mask = np.zeros((10, 10), dtype=bool)
#         new_mask[:3, :] = True  # 70 valid pixels
        
#         old_valid = np.sum(~old_mask)
#         data = np.random.randn(old_valid, 3)
        
#         with pytest.raises(ValueError, match="Cannot reshape"):
#             roi.remap_on_new_mask(data, old_mask, new_mask)

#     def test_remap_on_new_mask_error_wrong_dimensions(self):
#         """Test remap_on_new_mask raises error with wrong dimensions."""
#         old_mask = np.zeros((10, 10), dtype=bool)
#         old_mask[:5, :] = True
#         new_mask = np.zeros((10, 10), dtype=bool)
#         new_mask[:5, :] = True
        
#         old_valid = np.sum(~old_mask)
#         # Wrong first dimension
#         data = np.random.randn(old_valid + 10, 3)
        
#         with pytest.raises(ValueError, match="Mask length"):
#             roi.remap_on_new_mask(data, old_mask, new_mask)

#     def test_remap_on_new_mask_error_3d(self):
#         """Test remap_on_new_mask raises error with 3D array."""
#         old_mask = np.zeros((10, 10), dtype=bool)
#         new_mask = np.zeros((10, 10), dtype=bool)
        
#         # 3D data
#         data = np.random.randn(50, 3, 2)
        
#         with pytest.raises(ValueError, match="Can only operate on 2D arrays"):
#             roi.remap_on_new_mask(data, old_mask, new_mask)

