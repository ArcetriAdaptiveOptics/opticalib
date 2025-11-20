import numpy as np
from skimage import draw
from arte.types.mask import CircularMask
from opticalib import typings as _ot


def draw_circular_mask(
    shape: tuple[int, int] | list[int], center: int, radius: int
) -> _ot.MaskData:
    """
    Draws a circular boolean mask.

    Parameters
    ----------
    image_shape: tuple of ints
        The shape of the image (height, width).
    center: tuple of floats
        The (x, y) coordinates of the circle's center.
    radius: float
        The radius of the circle.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the circular area set to False.
    """
    mask = np.ones(shape, dtype=bool)
    rr, cc = draw.disk((center[1], center[0]), radius, shape=shape)
    mask[rr, cc] = False
    return mask


def draw_polygonal_mask(
    shape: tuple[int, int] | list[int], vertices: _ot.ArrayLike
) -> _ot.MaskData:
    """
    Draws a polygonal boolean mask.

    Parameters
    ----------
    image_shape: tuple of ints
        The shape of the image (height, width).
    vertices: np.ndarray
        An array of shape (N, 2) containing the (x, y) coordinates of the polygon's vertices.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the polygonal area set to False.
    """
    mask = np.ones(shape, dtype=bool)
    rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], shape=shape)
    mask[rr, cc] = False
    return mask


def find_circular_pupil(image: _ot.ImageData, method: str = "COG") -> _ot.MaskData:
    """
    Finds the circular pupil in the given image.

    Wrapper to the `arte.types.mask.CircularMask.fromMaskedArray` method.

    Parameters
    ----------
    image: np.ndarray or np.ma.maskedArray
        The input image in which to find the circular pupil.
    method: str
        The method used to find the circular pupil. Options are:
        - "COG" (Default);
        - "ImageMoments";
        - "RANSAC";
        - "correlation".

    Refer to arte.types.mask.CircularMask for more details on each method.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the circular pupil area set to False.
    """
    mask = CircularMask.fromMaskedArray(image, method=method).mask()
    return mask
