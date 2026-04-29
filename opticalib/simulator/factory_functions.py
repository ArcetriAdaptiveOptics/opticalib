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


def getPetalmirrorMaskAndCoords(
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
    coords : np.ndarray
        Coordinates of the centers of the segments in the petal-shaped mask.
    """
    if central_segment_radius is None:
        central_segment_radius = np.ceil(0.26666667* pupil_radius)

    hexagon_outer = geo.draw_hexagonal_mask(
        shape, radius=central_segment_radius + 10, masked=True
    )
    hexagon_inner = geo.draw_hexagonal_mask(
        shape, radius=central_segment_radius, masked=False
    )

    hexagon_ring = hexagon_inner ^ hexagon_outer

    line1 = geo.draw_linear_mask(shape, angle_deg=60, width=10)
    line2 = geo.draw_linear_mask(shape, angle_deg=120, width=10)
    line3 = geo.draw_linear_mask(shape, angle_deg=180, width=10)

    cross = ~(line1 ^ line2 ^ line3)
    cross[hexagon_inner == False] = True

    segmask = hexagon_ring ^ cross

    pupil = geo.draw_circular_pupil(shape, radius=pupil_radius, masked=False)

    segmask[pupil == 1] = 1
    segmask[hexagon_ring == 0] = 1
    
    offset = (shape[0] // 2, shape[1] // 2)
    dr = pupil_radius - central_segment_radius
    
    y_act1 = np.ceil(dr * 0.25 + offset[0] + central_segment_radius)
    y_act2 = np.ceil(dr * 0.75 + offset[0] + central_segment_radius)
    
    x_left = np.ceil((y_act2-y_act1) * np.tan(np.deg2rad(-30)) + offset[1])
    x_right = np.ceil((y_act2-y_act1) * np.tan(np.deg2rad(30)) + offset[1])

    s0 = np.array(
        [
            [y_act1, offset[0]],
            [y_act2, x_left],
            [y_act2, x_right],
        ]
    )
    
    coords = np.zeros((18, 2))
    
    for jj, a in enumerate([0, 60, -60, 120, -120, 180]):
        rotmat = np.array(
            [
                [np.cos(np.deg2rad(a)), -np.sin(np.deg2rad(a))],
                [np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))],
            ]
        )
        coords[jj*3:jj*3+3, :] = ((s0-offset) @ rotmat) + offset

    return segmask, coords


__all__ = [
    "getAlpaoCoordsMask",
    "getPetalmirrorMaskAndCoords",
    "pixel_scale",
    "generateZernikeMatrix",
]
