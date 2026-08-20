"""
Parallel-grid packing and demultiplexing for influence-function acquisition.

Pack actuators that are at least ``min_spacing`` apart in ``act_coord`` space
into simultaneous poke groups, then demux the multi-peak push–pull images
back into per-actuator IFs.
"""

from __future__ import annotations

import numpy as _np
import numpy.ma as _ma
from opticalib.core import _types as _ot


def pack_actuators(
    act_coord: _ot.ArrayLike,
    modes_list: _ot.ArrayLike,
    min_spacing: float,
) -> list[list[int]]:
    """
    Greedy packing: assign each actuator to the first group where distance to
    all members is >= ``min_spacing``, else start a new group.

    Parameters
    ----------
    act_coord : array-like
        Shape ``(2, n_acts)`` actuator grid coordinates (e.g. ``dm.act_coord``).
    modes_list : array-like
        Actuator indices to pack.
    min_spacing : float
        Minimum Euclidean distance in ``act_coord`` units between members of
        the same group.

    Returns
    -------
    groups : list[list[int]]
        Packed actuator groups (order follows first appearance in ``modes_list``).
    """
    if min_spacing <= 0:
        raise ValueError(f"min_spacing must be > 0, got {min_spacing}")

    coords = _np.asarray(act_coord, dtype=float)
    if coords.ndim != 2 or coords.shape[0] != 2:
        raise ValueError(
            f"act_coord must have shape (2, n_acts), got {coords.shape}"
        )

    modes = _np.asarray(modes_list, dtype=int).ravel()
    if modes.size == 0:
        return []

    groups: list[list[int]] = []
    group_coords: list[_np.ndarray] = []

    for act in modes:
        if act < 0 or act >= coords.shape[1]:
            raise ValueError(
                f"actuator index {act} out of range for act_coord "
                f"with {coords.shape[1]} actuators"
            )
        xy = coords[:, act]
        placed = False
        for g_idx, g_xy in enumerate(group_coords):
            dists = _np.linalg.norm(g_xy - xy[None, :], axis=1)
            if _np.all(dists >= min_spacing):
                groups[g_idx].append(int(act))
                group_coords[g_idx] = _np.vstack([g_xy, xy])
                placed = True
                break
        if not placed:
            groups.append([int(act)])
            group_coords.append(xy[None, :].copy())

    return groups


def groups_to_padded_array(groups: list[list[int]], fill: int = -1) -> _np.ndarray:
    """Pad groups to a rectangular ``(n_groups, max_size)`` int array."""
    if not groups:
        return _np.zeros((0, 0), dtype=int)
    max_size = max(len(g) for g in groups)
    out = _np.full((len(groups), max_size), fill, dtype=int)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
    return out


def padded_array_to_groups(arr: _ot.ArrayLike, fill: int = -1) -> list[list[int]]:
    """Inverse of :func:`groups_to_padded_array`."""
    a = _np.asarray(arr, dtype=int)
    if a.size == 0:
        return []
    if a.ndim == 1:
        a = a[None, :]
    return [[int(v) for v in row if int(v) != fill] for row in a]


def build_parallel_cmd_matrix(
    n_acts: int,
    groups: list[list[int]],
    modal_base: _ot.Optional[_ot.MatrixLike] = None,
) -> _np.ndarray:
    """
    Build a command matrix with one column per group.

    Each column is the sum of the selected modal-base vectors (default: zonal
    unit vectors) for actuators in that group.
    """
    n_groups = len(groups)
    if modal_base is None:
        cmd = _np.zeros((n_acts, n_groups), dtype=float)
        for j, g in enumerate(groups):
            for a in g:
                cmd[a, j] = 1.0
        return cmd

    mb = _np.asarray(modal_base, dtype=float)
    cmd = _np.zeros((n_acts, n_groups), dtype=float)
    for j, g in enumerate(groups):
        for a in g:
            cmd[:, j] += mb[:, a]
    return cmd


def _valid_bbox(mask: _np.ndarray) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) bounding box of valid (False) mask pixels."""
    valid = ~_np.asarray(mask, dtype=bool)
    if not _np.any(valid):
        raise ValueError("Image has no valid (unmasked) pixels")
    rows = _np.any(valid, axis=1)
    cols = _np.any(valid, axis=0)
    r_idx = _np.where(rows)[0]
    c_idx = _np.where(cols)[0]
    return int(r_idx[0]), int(r_idx[-1]) + 1, int(c_idx[0]), int(c_idx[-1]) + 1


def _seed_pixel_coords(
    act_xy: _np.ndarray,
    bbox: tuple[int, int, int, int],
) -> _np.ndarray:
    """
    Scale actuator-grid coords into the image bbox.

    ``act_xy`` shape ``(n, 2)`` with columns ``(x, y)`` in act units.
    Returns pixel coords ``(n, 2)`` as ``(col, row)``.
    """
    r0, r1, c0, c1 = bbox
    ax = act_xy[:, 0]
    ay = act_xy[:, 1]
    ax_min, ax_max = float(ax.min()), float(ax.max())
    ay_min, ay_max = float(ay.min()), float(ay.max())
    # Prefer spanning both axes; fall back to mid-box if degenerate.
    if ax_max - ax_min < 1e-12:
        cols = _np.full(len(ax), 0.5 * (c0 + c1 - 1))
    else:
        cols = c0 + (ax - ax_min) / (ax_max - ax_min) * max(c1 - c0 - 1, 1)
    if ay_max - ay_min < 1e-12:
        rows = _np.full(len(ay), 0.5 * (r0 + r1 - 1))
    else:
        rows = r0 + (ay - ay_min) / (ay_max - ay_min) * max(r1 - r0 - 1, 1)
    return _np.column_stack([cols, rows])


def _local_peak(
    abs_img: _np.ndarray,
    mask: _np.ndarray,
    seed_col: float,
    seed_row: float,
    search_radius: float,
) -> tuple[float, float]:
    """Return (col, row) of max |IF| near the seed within ``search_radius``."""
    h, w = abs_img.shape
    r0 = max(0, int(_np.floor(seed_row - search_radius)))
    r1 = min(h, int(_np.ceil(seed_row + search_radius)) + 1)
    c0 = max(0, int(_np.floor(seed_col - search_radius)))
    c1 = min(w, int(_np.ceil(seed_col + search_radius)) + 1)
    patch = abs_img[r0:r1, c0:c1].copy()
    pmask = _np.asarray(mask[r0:r1, c0:c1], dtype=bool)
    if patch.size == 0 or _np.all(pmask):
        return float(seed_col), float(seed_row)
    # Distance gate inside the patch
    yy, xx = _np.mgrid[r0:r1, c0:c1]
    dist2 = (xx - seed_col) ** 2 + (yy - seed_row) ** 2
    patch[pmask | (dist2 > search_radius**2)] = -_np.inf
    if not _np.isfinite(patch).any():
        return float(seed_col), float(seed_row)
    flat = int(_np.argmax(patch))
    pr, pc = _np.unravel_index(flat, patch.shape)
    return float(c0 + pc), float(r0 + pr)


def _fit_affine(src: _np.ndarray, dst: _np.ndarray) -> _np.ndarray:
    """
    Fit ``dst ~= A @ [x, y, 1]`` with ``A`` shape ``(2, 3)``.

    ``src`` / ``dst`` are ``(n, 2)``. Falls back to translation-only or
    seeded identity when under-determined.
    """
    n = src.shape[0]
    if n == 0:
        return _np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if n == 1:
        # Pure translation from seed→peak
        dx = dst[0, 0] - src[0, 0]
        dy = dst[0, 1] - src[0, 1]
        return _np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])

    ones = _np.ones((n, 1))
    X = _np.hstack([src, ones])  # (n, 3)
    # Solve for each destination coordinate
    try:
        coef_x, _, _, _ = _np.linalg.lstsq(X, dst[:, 0], rcond=None)
        coef_y, _, _, _ = _np.linalg.lstsq(X, dst[:, 1], rcond=None)
        return _np.vstack([coef_x, coef_y])
    except _np.linalg.LinAlgError:
        dx = float(_np.mean(dst[:, 0] - src[:, 0]))
        dy = float(_np.mean(dst[:, 1] - src[:, 1]))
        return _np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])


def _apply_affine(A: _np.ndarray, xy: _np.ndarray) -> _np.ndarray:
    """Apply ``(2, 3)`` affine to ``(n, 2)`` points → ``(n, 2)``."""
    ones = _np.ones((xy.shape[0], 1))
    return (A @ _np.hstack([xy, ones]).T).T


def _affine_mean_scale(A: _np.ndarray) -> float:
    """Mean isotropic scale (pixels per act-coord unit) from affine linear part."""
    lin = A[:, :2]
    s = 0.5 * (_np.linalg.norm(lin[0]) + _np.linalg.norm(lin[1]))
    return float(max(s, 1e-6))


def demux_group_image(
    group_img: _ot.ImageData,
    act_indices: _ot.ArrayLike,
    act_coord: _ot.ArrayLike,
    parallel_spacing: float,
    window_frac: float = 0.45,
) -> dict[int, _ma.MaskedArray]:
    """
    Split a multi-peak group IF into per-actuator windowed IFs.

    Parameters
    ----------
    group_img : masked array
        Reduced push–pull image for one parallel group.
    act_indices : array-like
        Actuator indices present in this group.
    act_coord : array-like
        Shape ``(2, n_acts)``.
    parallel_spacing : float
        Packing spacing (act-coord units); sets window radius.
    window_frac : float
        Window radius = ``window_frac * parallel_spacing`` in act units,
        converted to pixels via the fitted affine scale.

    Returns
    -------
    dict
        Mapping ``actuator_index -> MaskedArray`` IF (same shape as input).
    """
    acts = _np.asarray(act_indices, dtype=int).ravel()
    coords = _np.asarray(act_coord, dtype=float)
    if acts.size == 0:
        return {}

    data = _np.ma.getdata(group_img).astype(float, copy=True)
    mask = _np.ma.getmaskarray(group_img)
    abs_img = _np.abs(data)
    abs_img = _np.where(mask, 0.0, abs_img)

    bbox = _valid_bbox(mask)
    act_xy = coords[:, acts].T  # (n, 2) as (x, y)
    seeds = _seed_pixel_coords(act_xy, bbox)  # (col, row)

    # Search radius for peak: half spacing in pixel units estimated from bbox span
    r0, r1, c0, c1 = bbox
    ax_span = max(float(_np.ptp(act_xy[:, 0])), 1.0)
    ay_span = max(float(_np.ptp(act_xy[:, 1])), 1.0)
    px_per_act = 0.5 * (
        max(c1 - c0 - 1, 1) / ax_span + max(r1 - r0 - 1, 1) / ay_span
    )
    search_r = max(0.5 * parallel_spacing * px_per_act, 3.0)

    peaks = _np.zeros_like(seeds)
    for i, (sc, sr) in enumerate(seeds):
        peaks[i] = _local_peak(abs_img, mask, sc, sr, search_r)

    A = _fit_affine(act_xy, peaks)
    centers = _apply_affine(A, act_xy)  # (col, row)
    scale = _affine_mean_scale(A)
    radius_px = max(window_frac * parallel_spacing * scale, 1.5)

    h, w = data.shape
    yy, xx = _np.mgrid[0:h, 0:w]
    out: dict[int, _ma.MaskedArray] = {}
    for i, act in enumerate(acts):
        col_c, row_c = centers[i]
        circle = (xx - col_c) ** 2 + (yy - row_c) ** 2 > radius_px**2
        win_data = data.copy()
        # Zero outside the isolation window but keep the original pupil mask.
        # Masking the window exterior makes Flattening's cube master mask
        # (union of per-slice masks) empty when many groups are combined.
        win_data[circle] = 0.0
        out[int(act)] = _ma.masked_array(win_data, mask=mask.copy())
    return out


def demux_parallel_cube(
    group_images: list[_ot.ImageData],
    groups: list[list[int]],
    act_coord: _ot.ArrayLike,
    parallel_spacing: float,
    n_acts: int,
    window_frac: float = 0.45,
) -> list[_ma.MaskedArray]:
    """
    Demux all group images into a list of length ``n_acts`` (zeros for
    unmeasured actuators).
    """
    if not group_images:
        raise ValueError("No group images to demux")
    ref = group_images[0]
    shape = _np.ma.getdata(ref).shape
    ref_mask = _np.ma.getmaskarray(ref)
    empty = _ma.masked_array(_np.zeros(shape, dtype=float), mask=_np.ones(shape, dtype=bool))
    cube: list[_ma.MaskedArray] = [empty.copy() for _ in range(n_acts)]

    for img, group in zip(group_images, groups):
        parts = demux_group_image(
            img, group, act_coord, parallel_spacing, window_frac=window_frac
        )
        for act, if_img in parts.items():
            if 0 <= act < n_acts:
                cube[act] = if_img

    # Unmeasured acts: keep fully masked zeros (already set)
    # Measured acts that somehow missing: leave empty
    _ = ref_mask  # reserved for future master-mask union
    return cube
