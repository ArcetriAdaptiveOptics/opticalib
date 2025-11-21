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

def rotate_image(
    masked_img: _ot.ImageData,
    angle_deg: float,
    center: tuple[int, int] | None = None,
    order: int = 1):
    """
    Rotate masked image and point coordinates about a center.

    Parameters
    ----------
    masked_img : np.ma.MaskedArray
        2D masked array.
    points : (N,2) ndarray
        Pixel coordinates (row, col) or (y, x). Must match image indexing.
    angle_deg : float
        Counter-clockwise rotation angle in degrees.
    center : (cy, cx) or None
        Rotation center. If None uses image geometric center.
    order : int
        Interpolation order (0=nearest,1=linear). Higher -> slower.

    Returns
    -------
    rotated_img : np.ma.MaskedArray
    rotated_points : (N,2) ndarray
    """
    from scipy.ndimage import affine_transform
    img = masked_img.data
    msk = masked_img.mask
    h, w = img.shape
    if center is None:
        center = ((h - 1) / 2.0, (w - 1) / 2.0)

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix in (row, col) space
    R = np.array([[c, -s],
                  [s,  c]])

    # Affine transform expects matrix mapping output -> input.
    # For pure rotation about center: R^{-1} = R.T
    Rin = R.T

    # Offset term for affine_transform: offset = center - Rin @ center
    offset = np.array(center) - Rin @ np.array(center)

    rotated_data = affine_transform(
        img,
        Rin,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0,
        prefilter=(order > 1)
    )

    rotated_mask = affine_transform(
        msk.astype(float),
        Rin,
        offset=offset,
        order=0,
        mode='constant',
        cval=1.0  # outside becomes masked
    ) > 0.5
    
    rotated_img = np.ma.masked_array(rotated_data, mask=rotated_mask)
    return rotated_img
