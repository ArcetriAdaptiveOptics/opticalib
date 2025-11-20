import os
import numpy as np
import xupy as xp
from opticalib import folders as fp, typings as _t
from opticalib.core.read_config import load_yaml_config as cl
from opticalib.ground.modal_decomposer import ZernikeFitter

_alpao_list = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "alpao_conf.yaml"
)


def getAlpaoCoordsMask(nacts: int, shape: tuple[int] = (512, 512)):
    """
    Generates the coordinates of the DM actuators for a given DM size and actuator sequence.

    Parameters
    ----------
    Nacts : int
        Total number of actuators in the DM.

    Returns
    -------
    np.array
        Array of coordinates of the actuators.
    """
    dms = cl(_alpao_list)[f"DM{nacts}"]
    nacts_row_sequence = dms["coords"]
    opt_diameter = float(dms["opt_diameter"])
    pixel_scale = float(dms["pixel_scale"])
    # Coordinates creation
    n_dim = nacts_row_sequence[-1]
    upper_rows = nacts_row_sequence[:-1]
    lower_rows = [l for l in reversed(upper_rows)]
    center_rows = [n_dim] * upper_rows[0]
    rows_number_of_acts = upper_rows + center_rows + lower_rows
    n_rows = len(rows_number_of_acts)
    cx = np.array([], dtype=int)
    cy = np.array([], dtype=int)
    for i in range(n_rows):
        cx = np.concatenate(
            (
                cx,
                np.arange(rows_number_of_acts[i])
                + (n_dim - rows_number_of_acts[i]) // 2,
            )
        )
        cy = np.concatenate((cy, np.full(rows_number_of_acts[i], i)))
    coords = np.array([cx, cy])
    # Mask creation
    height, width = shape
    cx, cy = width / 2, height / 2
    radius = (opt_diameter * pixel_scale) / 2  # radius in pixels
    y, x = np.ogrid[:height, :width]
    mask = (x - cx) ** 2 + (y - cy) ** 2 >= radius**2
    return coords, mask


def getActuatorGeometry(
    n_act: int, dimension: int, geom: str = "default", angle_offset: float = 0.0
):
    """
    Generates the coordinates of the DM actuators based on the specified geometry.

    Parameters
    ----------
    n_act : int
        Number of actuators along one dimension.
    dimension : int
        Size of the DM in pixels.
    geom : str, optional
        Geometry type:
        - 'circular'
        - 'alpao'
        - 'default' (squared grid)
    angle_offset : float, optional
        Angle offset in degrees for circular geometry, by default 0.0.

    Returns
    -------
    x : np.ndarray
        X coordinates of the actuators.
    y : np.ndarray
        Y coordinates of the actuators.
    n_act_tot : int
        Total number of actuators.
    """
    step = float(dimension) / float(n_act)
    match geom:
        case "circular":
            if n_act % 2 == 0:
                na = xp.arange(xp.ceil((n_act + 1) / 2)) * 6
            else:
                step *= float(n_act) / float(n_act - 1)
                na = xp.arange(xp.ceil(n_act / 2.0)) * 6
            na[0] = 1  # The first value is always 1
            n_act_tot = int(xp.sum(na))
            pol_coords = xp.zeros((2, n_act_tot))
            ka = 0
            for ia in range(len(na)):
                n_angles = int(na[ia])
                for ja in range(n_angles):
                    pol_coords[0, ka] = (
                        360.0 / na[ia] * ja + angle_offset
                    )  # Angle in degrees
                    pol_coords[1, ka] = ia * step  # Radial distance
                    ka += 1
            x_c, y_c = dimension / 2, dimension / 2  # center
            x = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])) + x_c
            y = pol_coords[1] * xp.sin(xp.radians(pol_coords[0])) + y_c
        case "alpao":
            x, y = xp.meshgrid(
                xp.linspace(0, dimension, n_act), xp.linspace(0, dimension, n_act)
            )
            x, y = x.ravel(), y.ravel()
            x_c, y_c = dimension / 2, dimension / 2  # center
            rho = xp.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            rho_max = (
                dimension * (9 / 8 - n_act / (24 * 16))
            ) / 2  # slightly larger than dimension, depends on n_act
            n_act_tot = len(rho[rho <= rho_max])
            x = x[rho <= rho_max]
            y = y[rho <= rho_max]
        case _:
            x, y = xp.meshgrid(
                xp.linspace(0, dimension, n_act), xp.linspace(0, dimension, n_act)
            )
            x, y = x.ravel(), y.ravel()
            n_act_tot = n_act**2
    x = xp.asnumpy(x)
    y = xp.asnumpy(y)
    return x, y, n_act_tot


def pixel_scale(nacts: int):
    """
    Returns the pixel scale of the DM.

    Parameters
    ----------
    nacts : int
        Number of actuators in the DM.

    Returns
    -------
    float
        Pixel scale of the DM.
    """
    dm = cl(_alpao_list)[f"DM{nacts}"]
    return float(dm["pixel_scale"])


def generateZernikeMatrix(modes: int | list[int], mask: _t.MaskData):
    """
    Generates a matrix of Zernike polynomials projected on a given mask.

    Parameters
    ----------
    nacts : int
        Number of actuators in the DM.
    n_modes : int
        Number of Zernike modes to generate.
    mask : _t.MaskData
        Mask to project the Zernike polynomials on.

    Returns
    -------
    np.ndarray
        Matrix of Zernike polynomials projected on the mask.
    """
    valixpx = np.sum(mask == 0)
    if isinstance(modes, int):
        zerns = list(range(1, modes + 1))
    else:
        if not all([i != 0 for i in modes]):
            raise ValueError("Index 0 not permitted.")
        zerns = modes
    nzerns = len(zerns)
    zfit = ZernikeFitter(mask)
    ZM = np.zeros((valixpx, nzerns))
    for i in range(nzerns):
        surf = zfit.makeSurface([zerns[i]])
        masked_data = surf[~mask]
        ZM[:, i] = masked_data
    return ZM


def createMaskAndCoords(
    coords: str | np.ndarray,
    geometry: str = "circle",
    radius: float = None,
    vertices: np.ndarray = None,
    mask_size: int = 1000,
    padding: int = 10,
    mirror_axis: str = None,
):
    """
    General function to create a mask and scaled actuator coordinates for any deformable mirror.

    Parameters
    ----------
    coords_file : str
        Path to the FITS file containing actuator coordinates in meters.
        Expected format: Nx2 array with [x, y] coordinates.

    geometry : str, optional
        Type of mirror geometry. Options are 'circle' or 'polygon'. Default is 'circle'.

    radius : float, optional
        Radius of the circular mirror aperture in meters. Required if geometry='circle'.

    vertices : np.ndarray, optional
        Array of polygon vertices in meters with shape (N, 2) for [x, y] coordinates.
        Required if geometry='polygon'. Vertices should be ordered sequentially.

    mask_size : int, optional
        Size of the mask in pixels (for the largest dimension). Default is 1000.

    padding : int, optional
        Number of pixels to pad around the mask to avoid edge effects. Default is 10.

    mirror_axis : str, optional
        If provided, creates a symmetric mask by mirroring along the specified axis.
        Options: 'horizontal', 'vertical', None. Default is None (no mirroring).

    Returns
    -------
    final_mask : np.ndarray
        Boolean mask where True indicates invalid/masked pixels.

    act_px_coords : np.ndarray
        Actuator coordinates in pixel space (Nx2 array).

    """
    from opticalib.ground.osutils import load_fits as lf
    from skimage.draw import polygon2mask

    # Load actuator coordinates in meters
    if isinstance(coords, str):
        act_coords = lf(coords)
    else:
        act_coords = coords
    if act_coords.shape[1] != 2:
        raise ValueError(
            f"Expected actuator coordinates with shape (N, 2), got {act_coords.shape}"
        )

    # Determine mirror physical dimensions
    x_min, y_min = act_coords.min(axis=0)
    x_max, y_max = act_coords.max(axis=0)
    x_extent = x_max - x_min  # meters
    y_extent = y_max - y_min  # meters

    # Create mask based on geometry
    if geometry == "circle":
        if radius is None:
            raise ValueError("Radius must be provided for circular geometry")

        # Create circular mask
        mask_diameter = mask_size
        center = mask_diameter / 2
        y_grid, x_grid = np.ogrid[:mask_diameter, :mask_diameter]
        distance_from_center = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
        mask = distance_from_center > (mask_diameter / 2 - 1)

        # Physical scale: map 2*radius (diameter) to mask_diameter pixels
        pixel_scale = (2 * radius) / mask_diameter

    elif geometry == "polygon":
        if vertices is None:
            raise ValueError("Vertices must be provided for polygon geometry")

        vertices = np.asarray(vertices)
        if vertices.shape[1] != 2:
            raise ValueError(
                f"Expected vertices with shape (N, 2), got {vertices.shape}"
            )

        # Normalize vertices to [0, 1] range
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_extent = v_max - v_min

        # Scale vertices to pixel coordinates
        scale_factor = mask_size / v_extent.max()
        vertices_px = (vertices - v_min) * scale_factor

        # Create polygon mask
        mask_shape = (
            int(np.ceil(vertices_px[:, 1].max())) + 1,
            int(np.ceil(vertices_px[:, 0].max())) + 1,
        )

        # Convert vertices to row, col format for polygon2mask
        poly_coords = np.column_stack([vertices_px[:, 1], vertices_px[:, 0]])
        mask = ~polygon2mask(mask_shape, poly_coords)

        # Physical scale
        pixel_scale = v_extent.max() / mask_size

    else:
        raise ValueError(
            f"Unknown geometry type: {geometry}. Use 'circle' or 'polygon'."
        )

    # Apply mirroring if requested
    if mirror_axis == "horizontal":
        # Mirror horizontally (left-right)
        half_mask = mask[:, : mask.shape[1] // 2]
        mirrored = np.fliplr(half_mask)
        mask = np.hstack([half_mask, mirrored])
    elif mirror_axis == "vertical":
        # Mirror vertically (up-down)
        half_mask = mask[: mask.shape[0] // 2, :]
        mirrored = np.flipud(half_mask)
        mask = np.vstack([half_mask, mirrored])

    # Convert actuator coordinates to pixel coordinates
    # Center the coordinates
    act_coords_centered = act_coords - act_coords.min(axis=0)

    # Scale to pixels
    act_px_coords = act_coords_centered / pixel_scale

    # Center actuators in the mask
    mask_center = np.array([mask.shape[1] / 2, mask.shape[0] / 2])
    act_center = np.array([act_px_coords[:, 0].mean(), act_px_coords[:, 1].mean()])
    act_px_coords += mask_center - act_center

    # Apply padding to mask
    final_mask = np.pad(mask, padding, mode="constant", constant_values=True)

    # Adjust actuator coordinates for padding
    act_px_coords += padding

    return final_mask, act_px_coords


__all__ = [
    "getAlpaoCoordsMask",
    "getActuatorGeometry",
    "pixel_scale",
    "generateZernikeMatrix",
]
