"""
Tests for opticalib.ground.roi module.
"""

import pytest
import numpy as np
import numpy.ma as ma
from opticalib.ground import roi


class TestRoiGenerator:
    """Test roi_generator function."""

    def test_roi_generator_basic(self, sample_image):
        """Test roi_generator with basic image."""
        roi_list = roi.roi_generator(sample_image)

        assert isinstance(roi_list, list)
        # Should return at least one ROI if image has valid pixels
        if np.sum(~sample_image.mask) > 0:
            assert len(roi_list) > 0
            for r in roi_list:
                assert r.shape == sample_image.shape
                assert isinstance(r, np.ndarray)

    def test_roi_generator_fully_masked(self):
        """Test roi_generator with fully masked image."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)

        roi_list = roi.roi_generator(masked_img)
        assert len(roi_list) == 0

    def test_roi_generator_multiple_rois(self):
        """Test roi_generator with multiple disconnected regions."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create two disconnected regions
        mask[20:30, 20:30] = False
        mask[70:80, 70:80] = False
        masked_img = ma.masked_array(data, mask=mask)

        roi_list = roi.roi_generator(masked_img)
        assert len(roi_list) >= 2

    def test_roi_generator_small_rois_filtered(self):
        """Test that small ROIs (< 100 pixels) are filtered out."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create a small region (< 100 pixels)
        mask[45:50, 45:50] = False  # 5x5 = 25 pixels
        masked_img = ma.masked_array(data, mask=mask)

        roi_list = roi.roi_generator(masked_img)
        # Small ROI should be filtered out
        assert len(roi_list) == 0


class TestImgCut:
    """Test img_cut function."""

    def test_img_cut_basic(self, sample_image):
        """Test img_cut with basic image."""
        cut_img = roi.img_cut(sample_image)

        assert isinstance(cut_img, ma.MaskedArray)
        # Cut image should be smaller or equal to original
        assert cut_img.shape[0] <= sample_image.shape[0]
        assert cut_img.shape[1] <= sample_image.shape[1]

    def test_img_cut_fully_masked(self):
        """Test img_cut with fully masked image."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)

        cut_img = roi.img_cut(masked_img)
        # Should return original image if no finite pixels
        assert cut_img.shape == masked_img.shape

    def test_img_cut_centered_region(self):
        """Test img_cut with centered valid region."""
        data = np.random.randn(100, 100)
        mask = np.ones((100, 100), dtype=bool)
        # Create a centered valid region
        mask[40:60, 40:60] = False
        masked_img = ma.masked_array(data, mask=mask)

        cut_img = roi.img_cut(masked_img)
        # Should cut to approximately 20x20 region
        assert cut_img.shape[0] <= 25  # Allow some margin
        assert cut_img.shape[1] <= 25

    def test_img_cut_no_nan(self):
        """Test img_cut with image containing no NaN values."""
        data = np.random.randn(100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        masked_img = ma.masked_array(data, mask=mask)

        cut_img = roi.img_cut(masked_img)
        # Should return full image or slightly trimmed
        assert cut_img.shape[0] >= 90
        assert cut_img.shape[1] >= 90


class TestCubeMasterMask:
    """Test cube_master_mask function."""

    def test_cube_master_mask_basic(self, sample_cube):
        """Test cube_master_mask with basic cube."""
        master_mask = roi.cube_master_mask(sample_cube)

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
            np.zeros((50, 50), dtype=bool),
        ]
        masks[0][:10, :10] = True
        masks[1][:15, :15] = True
        masks[2][:20, :20] = True

        cube = ma.masked_array(data, mask=np.stack(masks, axis=2))

        master_mask = roi.cube_master_mask(cube)
        # Master mask should include all masked regions
        assert np.all(master_mask[:10, :10])  # All frames mask this
        assert np.all(master_mask[:15, :15])  # At least one frame masks this

    def test_cube_master_mask_single_frame(self):
        """Test cube_master_mask with single frame cube."""
        data = np.random.randn(50, 50, 1).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        mask[:10, :10] = True
        cube = ma.masked_array(
            data, mask=np.broadcast_to(mask[..., np.newaxis], data.shape)
        )

        master_mask = roi.cube_master_mask(cube)
        np.testing.assert_array_equal(master_mask, mask)


@pytest.fixture
def overlapping_cube():
    """
    Non-square cube with 3 frames whose masks overlap differently, so each
    pixel region is masked in a known number of frames.

    - ``[:10, :10]``    masked in 3/3 frames
    - ``[10:20, :10]``  masked in 2/3 frames
    - ``[20:30, :10]``  masked in 1/3 frames
    - elsewhere         masked in 0/3 frames
    """
    ny, nx, nf = 40, 60, 3
    data = np.random.randn(ny, nx, nf).astype(np.float32)
    masks = np.zeros((ny, nx, nf), dtype=bool)
    masks[:10, :10, :] = True
    masks[10:20, :10, :2] = True
    masks[20:30, :10, 0] = True
    return ma.masked_array(data, mask=masks)


class TestCubeMasterMaskMethods:
    """Test the ``method`` and ``mean_threshold`` options of cube_master_mask."""

    @staticmethod
    def _count(cube):
        return np.sum(cube.mask, axis=2)

    @pytest.mark.parametrize("method", ["logor", "logand", "mean"])
    def test_output_shape_and_dtype(self, overlapping_cube, method):
        """Every method returns a 2D boolean mask matching the frame shape."""
        master_mask = roi.cube_master_mask(overlapping_cube, method=method)

        assert master_mask.shape == overlapping_cube.shape[:2]
        assert master_mask.dtype == bool

    def test_default_method_is_logor(self, overlapping_cube):
        """Omitting ``method`` is equivalent to ``method='logor'``."""
        np.testing.assert_array_equal(
            roi.cube_master_mask(overlapping_cube),
            roi.cube_master_mask(overlapping_cube, method="logor"),
        )

    def test_logor_is_union(self, overlapping_cube):
        """``logor`` masks pixels masked in at least one frame."""
        master_mask = roi.cube_master_mask(overlapping_cube, method="logor")

        np.testing.assert_array_equal(
            master_mask, self._count(overlapping_cube) >= 1
        )
        assert np.all(master_mask[:30, :10])
        assert not np.any(master_mask[30:, :])
        assert not np.any(master_mask[:, 10:])

    def test_logand_is_intersection(self, overlapping_cube):
        """``logand`` masks only pixels masked in every frame."""
        master_mask = roi.cube_master_mask(overlapping_cube, method="logand")

        np.testing.assert_array_equal(
            master_mask, self._count(overlapping_cube) == overlapping_cube.shape[2]
        )
        assert np.all(master_mask[:10, :10])
        assert not np.any(master_mask[10:, :])

    def test_logand_subset_of_logor(self, overlapping_cube):
        """The intersection mask is always contained in the union mask."""
        and_mask = roi.cube_master_mask(overlapping_cube, method="logand")
        or_mask = roi.cube_master_mask(overlapping_cube, method="logor")

        assert not np.any(and_mask & ~or_mask)

    def test_mean_default_threshold_is_majority(self, overlapping_cube):
        """With the default threshold (0.5), ``mean`` is a majority vote."""
        master_mask = roi.cube_master_mask(overlapping_cube, method="mean")

        assert np.all(master_mask[:10, :10])  # 3/3
        assert np.all(master_mask[10:20, :10])  # 2/3
        assert not np.any(master_mask[20:30, :10])  # 1/3
        assert not np.any(master_mask[30:, :])  # 0/3
        assert not np.any(master_mask[:, 10:])  # 0/3

    @pytest.mark.parametrize(
        "threshold, min_count",
        [
            (0.2, 1),  # 1/3 > 0.2
            (0.5, 2),  # 2/3 > 0.5
            (0.7, 3),  # 3/3 > 0.7
        ],
    )
    def test_mean_custom_threshold(self, overlapping_cube, threshold, min_count):
        """``mean_threshold`` controls how many frames must mask a pixel."""
        master_mask = roi.cube_master_mask(
            overlapping_cube, method="mean", mean_threshold=threshold
        )

        np.testing.assert_array_equal(
            master_mask, self._count(overlapping_cube) >= min_count
        )

    def test_mean_threshold_is_strict(self):
        """A pixel whose mean equals the threshold is not masked."""
        data = np.random.randn(20, 30, 2).astype(np.float32)
        masks = np.zeros((20, 30, 2), dtype=bool)
        masks[:5, :5, 0] = True  # mean = 0.5 on this region
        cube = ma.masked_array(data, mask=masks)

        master_mask = roi.cube_master_mask(cube, method="mean", mean_threshold=0.5)
        assert not np.any(master_mask)

    def test_mean_threshold_zero_matches_logor(self, overlapping_cube):
        """``mean`` with threshold 0 reproduces ``logor``."""
        np.testing.assert_array_equal(
            roi.cube_master_mask(overlapping_cube, method="mean", mean_threshold=0.0),
            roi.cube_master_mask(overlapping_cube, method="logor"),
        )

    def test_mean_high_threshold_matches_logand(self, overlapping_cube):
        """``mean`` with a threshold just below 1 reproduces ``logand``."""
        np.testing.assert_array_equal(
            roi.cube_master_mask(
                overlapping_cube, method="mean", mean_threshold=1 - 1e-6
            ),
            roi.cube_master_mask(overlapping_cube, method="logand"),
        )

    def test_mean_threshold_one_masks_nothing(self, overlapping_cube):
        """No mean can exceed 1, so threshold 1 yields an empty mask."""
        master_mask = roi.cube_master_mask(
            overlapping_cube, method="mean", mean_threshold=1.0
        )
        assert not np.any(master_mask)

    @pytest.mark.parametrize("method", ["logor", "logand"])
    def test_mean_threshold_ignored_by_logical_methods(
        self, overlapping_cube, method
    ):
        """``mean_threshold`` has no effect on ``logor``/``logand``."""
        np.testing.assert_array_equal(
            roi.cube_master_mask(overlapping_cube, method=method, mean_threshold=0.0),
            roi.cube_master_mask(overlapping_cube, method=method, mean_threshold=0.99),
        )

    @pytest.mark.parametrize("method", ["logor", "logand", "mean"])
    def test_identical_frames_all_methods_agree(self, sample_cube, method):
        """When all frames share the same mask, every method returns it."""
        master_mask = roi.cube_master_mask(sample_cube, method=method)
        np.testing.assert_array_equal(master_mask, sample_cube.mask[:, :, 0])

    @pytest.mark.parametrize("method", ["or", "LOGOR", "median", ""])
    def test_unknown_method_raises(self, overlapping_cube, method):
        """An unsupported method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            roi.cube_master_mask(overlapping_cube, method=method)

    @pytest.mark.parametrize("method", ["logor", "logand", "mean"])
    def test_apply_sets_master_mask_on_every_frame(self, overlapping_cube, method):
        """With ``apply=True`` the cube is returned with the master mask on all frames."""
        expected = roi.cube_master_mask(overlapping_cube.copy(), method=method)

        result = roi.cube_master_mask(overlapping_cube, method=method, apply=True)

        assert isinstance(result, ma.MaskedArray)
        assert result.shape == overlapping_cube.shape
        for i in range(result.shape[2]):
            np.testing.assert_array_equal(result.mask[:, :, i], expected)


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
