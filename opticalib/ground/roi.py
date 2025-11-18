"""
Module containing functions for region of interest (ROI) generation and other image utilities
within the Opticalib framework.

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
"""

import numpy as _np
from skimage import measure as _meas
from opticalib import typings as _ot


def roiGenerator(img: _ot.ImageData) -> list[_ot.ImageData]:
    """
    This function generates a list of `n_masks` roi from the input image.

    Parameters
    ----------
    img: ImageData | np.ma.maskedArray
        input image from which the roi are generated.

    Returns
    -------
    roiList: list
        List of the first `n_masks` roi found in the image.
    """
    # Labelled pixel islands in image mask
    labels = _meas.label(_np.invert(img.mask))
    roiList = []
    for i in range(1, labels.max() + 1):
        maski = _np.zeros(labels.shape, dtype=bool)
        maski[_np.where(labels == i)] = 1
        final_roi = _np.ma.mask_or(_np.invert(maski), img.mask)
        # Eliminating possible islands with less than 100 pixels
        if _np.invert(final_roi).sum() < 100:
            continue
        roiList.append(final_roi)
    return roiList


def imgCut(img: _ot.ImageData):
    """
    Cuts the image to the bounding box of the finite (non-NaN) pixels in the masked image.

    Parameters
    ----------
    image : np.ma.maskedArray
        The original masked image array.

    Returns
    -------
    cutImg = np.ma.maskedArray
        The cut image within the bounding box of finite pixels.
    """
    # Find indices of finite (non-NaN) pixels
    finite_coords = _np.argwhere(_np.isfinite(img))
    # If there are no finite pixels, return the original image
    if finite_coords.size == 0:
        return img
    top_left = finite_coords.min(axis=0)
    bottom_right = finite_coords.max(axis=0)
    cutImg = img[top_left[0] : bottom_right[0] + 1, top_left[1] : bottom_right[1] + 1]
    return cutImg


def cubeMasterMask(cube: _ot.CubeData) -> _ot.ImageData:
    """
    Generates a master mask for a cube by combining the masks of all individual frames.

    Parameters
    ----------
    cube : np.ma.maskedArray
        The input cube where each slice along the last axis is a masked image.

    Returns
    -------
    master_mask : np.ma.maskedArray
        The master mask that combines all individual masks in the cube.
    """
    master_mask = _np.ma.logical_or.reduce(
        [cube[:, :, i].mask for i in range(cube.shape[-1])]
    )
    return master_mask


def remap_on_new_mask(
    data: _ot.ArrayLike, old_mask: _ot.MaskData, new_mask: _ot.MaskData
) -> _ot.ArrayLike:
    """
    Remaps the matrix data defined on valid values
    of old_mask to valid values on new_mask.

    Parameters
    ----------
    data : xp.ndarray
        2D array of shape (sum(1-old_mask), N)
    old_mask : xp.ndarray
        2D boolean array defining the old mask
    new_mask : xp.ndarray
        2D boolean array defining the new mask

    Returns
    -------
    remapped_data : xp.ndarray
        2D array of shape (sum(1-new_mask), N)
    """
    old_len = _np.sum(1 - old_mask)
    new_len = _np.sum(1 - new_mask)

    if old_len < new_len:
        raise ValueError(f"Cannot reshape from {old_len} to {new_len}")

    # Handle dimention mismatch
    transpose = False
    if _np.shape(data)[0] != old_len:
        data = data.T
        transpose = True

    if _np.shape(data)[0] != old_len:
        raise ValueError(
            f"Mask length {old_len} is incompatible with dimensions {data.shape}"
        )
    elif len(_np.shape(data)) > 2:
        raise ValueError("Can only operate on 2D arrays")

    N = data.shape[1]
    remasked_data = _np.zeros([int(new_len), N])

    # Define helper Function
    def reshape_on_mask(vec, mask):
        """
        Reshape a given array on a 2D mask.

        Parameters
        ----------
        vec : array-like
            1D array of shape sum(1-mask)
        mask : 2D boolean array
            Mask to reshape on

        Returns
        -------
        image : 2D array
            2D array with shape of mask, with vec values in ~mask
        """
        image = _np.zeros(mask.shape, dtype=_np.float)
        image[~mask] = vec
        image = _np.reshape(image, mask.shape)
        return _np.array(image)

    for j in range(N):
        old_data_2D = reshape_on_mask(data[:, j], old_mask)
        remasked_data[:, j] = old_data_2D[~new_mask]

    if transpose:
        remasked_data = remasked_data.T

    return remasked_data
