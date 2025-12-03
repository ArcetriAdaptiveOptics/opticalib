import numpy as np
from skimage import draw
from arte.types.mask import CircularMask
from opticalib import typings as _ot


def draw_circular_pupil(
    shape: tuple[int, int] | list[int],
    radius: int,
    center: int = None,
    masked: bool = False,
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
    masked: bool
        If True, flips the logic, and sets the circular area to True.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the circular area set to False.
    """
    mask = np.ones(shape, dtype=bool)
    if not center:
        center = (shape[1] // 2, shape[0] // 2)
    rr, cc = draw.disk((center[1], center[0]), radius, shape=shape)
    mask[rr, cc] = False
    if masked:
        mask = ~mask
    return mask


def draw_polygonal_mask(
    shape: tuple[int, int] | list[int], vertices: _ot.ArrayLike, masked: bool = False
) -> _ot.MaskData:
    """
    Draws a polygonal boolean mask.

    Parameters
    ----------
    image_shape: tuple of ints
        The shape of the image (height, width).
    vertices: np.ndarray
        An array of shape (N, 2) containing the (x, y) coordinates of the polygon's vertices.
    masked: bool
        If True, flips the logic, and sets the polygonal area to True.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the polygonal area set to False.
    """
    mask = np.ones(shape, dtype=bool)
    rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], shape=shape)
    mask[rr, cc] = False
    if masked:
        mask = ~mask
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
    order: int = 1,
):
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
    R = np.array([[c, -s], [s, c]])

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
        mode="constant",
        cval=0.0,
        prefilter=(order > 1),
    )

    rotated_mask = (
        affine_transform(
            msk.astype(float),
            Rin,
            offset=offset,
            order=0,
            mode="constant",
            cval=1.0,  # outside becomes masked
        )
        > 0.5
    )

    rotated_img = np.ma.masked_array(rotated_data, mask=rotated_mask)
    return rotated_img


def draw_hexagonal_mask(
    shape: tuple[int, int] | list[int],
    radius: int,
    center: int = None,
    masked: bool = False,
) -> _ot.MaskData:
    """
    Draws a hexagonal boolean mask.

    Parameters
    ----------
    image_shape: tuple of ints
        The shape of the image (height, width).
    radius: float
        The radius of the hexagon.
    center: tuple of floats
        The (x, y) coordinates of the hexagon's center.

    Returns
    -------
    mask: np.ndarray
        A binary mask with the hexagonal area set to False.
    """
    if not center:
        center = (shape[1] // 2, shape[0] // 2)
    vertexes = np.array(
        [
            [center[0] + radius * np.cos(np.pi / 3 * i) for i in range(6)],
            [np.ceil(center[1]) + radius * np.sin(np.pi / 3 * i) for i in range(6)],
        ],
        dtype=int,
    ).T
    return draw_polygonal_mask(shape, vertexes, masked=masked)


def create_line_mask(
    shape: tuple[int, int],
    angle_deg: float | None = None,
    width: int = 1,
    center: tuple[int, int] | None = None,
    slope: float | None = None,
    masked: bool = False,
) -> _ot.MaskData:
    """
    Create a boolean mask with a line of given width.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the mask (height, width), e.g., (2000, 2000)
    angle_deg : float, optional
        Angle in degrees with respect to horizontal (counterclockwise positive)
    width : float
        Width of the line in pixels
    center : tuple[int, int] | None, optional
        Point (x, y) through which the line passes
    slope : float, optional
        Angular coefficient m (alternative to angle_deg)
    masked : bool, optional
        If True, inverts the mask logic (line area set to True)

    Returns
    -------
    mask : ndarray
        Boolean mask with True where the line is
    """
    if angle_deg is None and slope is None:
        raise ValueError("Must provide either angle_deg or slope")

    # Convert slope to angle if needed
    if angle_deg is not None:
        theta = np.deg2rad(angle_deg)
    else:
        theta = np.arctan(slope)

    if center is None:
        center = (shape[1] // 2, shape[0] // 2)  # (x, y)

    # Line coefficients: a*x + b*y + c = 0
    a = np.sin(theta)
    b = -np.cos(theta)
    c = -a * center[0] - b * center[1]

    # Create coordinate grids
    # Note: y is row index (0 at top), x is column index
    height, width_px = shape
    y, x = np.ogrid[0:height, 0:width_px]

    # Compute signed distance to line
    # Since a² + b² = sin²θ + cos²θ = 1, denominator is 1
    distance = np.abs(a * x + b * y + c)

    # Create mask: pixels within half-width of the line
    mask = distance <= (width / 2.0)
    if masked:
        mask = ~mask
    return mask
