import os
import numpy as np
import xupy as xp
from opticalib import typings as _t
from opticalib.core.read_config import load_yaml_config as cl
from opticalib.ground.modal_decomposer import ZernikeFitter
from opticalib.ground import geometry as geo

_alpao_list = os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_API"), "alpao_conf.yaml"
)


def getAlpaoCoordsMask(
    nacts: int, shape: tuple[int] = (512, 512)
) -> tuple[_t.ArrayLike, _t.MaskData]:
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


def getPetalmirrorMask(
    shape: tuple[int, int], pupil_radius: int, central_segment_radius: int | None = None
) -> _t.MaskData:
    """
    Generates a petal-shaped mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the mask (height, width).
    pupil_radius : int
        Radius of the pupil.
    central_segment_radius : int, optional
        Radius of the central segment. If not provided, defaults to 26.6% of the
        pupil_radius.

    Returns
    -------
    mask : _t.MaskData
        Petal-shaped boolean mask.
    """
    if central_segment_radius is None:
        central_segment_radius = np.ceil(0.266666666666 * pupil_radius)

    hexagon_outer = geo.draw_hexagonal_mask(
        shape, radius=central_segment_radius + 10, masked=True
    )
    hexagon_inner = geo.draw_hexagonal_mask(
        shape, radius=central_segment_radius, masked=False
    )

    hexagon_ring = hexagon_inner ^ hexagon_outer

    line1 = geo.create_line_mask(shape, angle_deg=60, width=10)
    line2 = geo.create_line_mask(shape, angle_deg=120, width=10)
    line3 = geo.create_line_mask(shape, angle_deg=180, width=10)

    cross = ~(line1 ^ line2 ^ line3)
    cross[hexagon_inner == False] = True

    segmask = hexagon_ring ^ cross

    pupil = geo.draw_circular_pupil(shape, radius=pupil_radius, masked=False)

    segmask[pupil == 1] = 1
    segmask[hexagon_ring == 0] = 1

    return segmask


__all__ = [
    "getAlpaoCoordsMask",
    "getActuatorGeometry",
    "pixel_scale",
    "generateZernikeMatrix",
]
