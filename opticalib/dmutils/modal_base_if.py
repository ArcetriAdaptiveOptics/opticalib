"""
Build DM command modal bases (Zernike / KL) from a measured IF cube.

Optical surfaces come from ARTE via ``ZernikeFitter`` / ``KLFitter``.
They are projected onto the zonal IF map of an ``INTMatrices`` tracking
number; the result is an actuator command matrix ``(n_acts, n_modes)``
saved under ``ModalBases/`` for ``IFFUNC.modal_base`` in the YAML.

This is for OPD cubes (PhaseCam-style ``IMCube.fits``), not PWFS slope maps.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.ma as ma

from opticalib.core.root import folders
from opticalib.ground import modal_decomposer as _md
from opticalib.ground import osutils as _osu

_CUBE_FILE = "IMCube.fits"
_MATRIX_FILE = "cmd_matrix.fits"


def _as_numpy(arr):
    if arr is None:
        return None
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr)


def _load_intmat_files(tn: str):
    fold = os.path.join(folders.INTMAT_ROOT_FOLDER, tn)
    cube_path = os.path.join(fold, _CUBE_FILE)
    if not os.path.isfile(cube_path):
        raise FileNotFoundError(f"Missing interaction cube: {cube_path}")

    cube = _osu.load_fits(cube_path)
    cmd = None
    for name in (_MATRIX_FILE, "cmdMatrix.fits"):
        path = os.path.join(fold, name)
        if os.path.isfile(path):
            cmd = _osu.load_fits(path)
            break
    if cmd is None:
        raise FileNotFoundError(
            f"Missing command matrix ({_MATRIX_FILE} or cmdMatrix.fits) in {fold}"
        )
    return cube, np.asarray(cmd, dtype=np.float64)


def _master_mask(cube) -> np.ndarray:
    """True = invalid (numpy.ma convention), intersection of all cube planes."""
    if not ma.isMaskedArray(cube):
        data = _as_numpy(cube)
        return ~np.isfinite(data).all(axis=2)
    mask = np.ma.getmaskarray(cube)
    return np.any(mask.astype(bool), axis=2)


def _interaction_matrix(cube, master_mask: np.ndarray) -> np.ndarray:
    """Flatten valid pixels: G shape (npix, ncmd). Cube is (ny, nx, ncmd)."""
    data = np.ma.filled(cube, np.nan) if ma.isMaskedArray(cube) else _as_numpy(cube)
    if data.ndim != 3:
        raise ValueError(f"IMCube must be 3-D (ny, nx, ncmd), got shape {data.shape}")
    valid = ~master_mask
    ncmd = data.shape[2]
    npix = int(valid.sum())
    if npix == 0:
        raise ValueError("Interaction cube master mask has no valid pixels")
    g = np.zeros((npix, ncmd), dtype=np.float64)
    for i in range(ncmd):
        g[:, i] = data[:, :, i][valid]
    if not np.isfinite(g).all():
        raise ValueError("Interaction cube has non-finite values on valid pixels")
    return g


def _zonal_g(g: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    """Map actuator commands to OPD: G_zonal = G @ pinv(cmdMatrix)."""
    cmd = np.asarray(cmd, dtype=np.float64)
    if cmd.ndim != 2:
        raise ValueError(f"cmdMatrix must be 2-D (nacts, ncmd), got {cmd.shape}")
    nacts, ncmd = cmd.shape
    if g.shape[1] != ncmd:
        raise ValueError(
            f"IMCube ncmd={g.shape[1]} does not match cmdMatrix columns={ncmd}"
        )
    if nacts == ncmd and np.allclose(cmd, np.eye(nacts)):
        return g
    return g @ np.linalg.pinv(cmd)


def _make_fitter(kind: str, master_mask: np.ndarray, n_kl: int):
    dummy = ma.masked_array(
        np.zeros(master_mask.shape, dtype=np.float64), mask=master_mask
    )
    if kind == "zernike":
        return _md.ZernikeFitter(fit_mask=dummy)
    if kind == "kl":
        return _md.KLFitter(nKLModes=n_kl, fit_mask=dummy)
    raise ValueError("kind must be 'zernike' or 'kl'")


def _surface_on_mask(fitter, kind: str, index: int, master_mask: np.ndarray) -> np.ndarray:
    if kind == "zernike":
        surf = fitter.make_surface([index])
    else:
        surf = fitter._get_mode_from_generator(index)
    surf = _as_numpy(np.ma.filled(surf, 0.0) if ma.isMaskedArray(surf) else surf)
    if surf.shape != master_mask.shape:
        raise ValueError(
            f"Generated mode shape {surf.shape} != cube mask {master_mask.shape}"
        )
    return surf[~master_mask].astype(np.float64)


def _default_indices(kind: str, n_modes: int) -> list[int]:
    if kind == "zernike":
        # Skip Noll piston (index 1)
        return list(range(2, n_modes + 2))
    # ARTE KarhunenLoeveGenerator caches modes as 0 .. n-1
    return list(range(0, n_modes))


def make_modal_base_if(
    tn: str,
    kind: str = "zernike",
    n_modes: int = 50,
    mode_indices: list[int] | None = None,
    save_name: str | None = None,
    save: bool = True,
    rcond: float = 1e-3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Project ARTE Zernike or KL surfaces onto a measured IF cube.

    Parameters
    ----------
    tn : str
        Tracking number under ``INTMatrices/``.
    kind : {'zernike', 'kl'}
        Optical basis to generate.
    n_modes : int
        Number of modes if ``mode_indices`` is not given.
    mode_indices : list of int, optional
        Explicit generator indices. Zernike uses Noll (1=piston);
        default skips piston. KL uses ARTE indices ``0 .. n_modes-1``.
    save_name : str, optional
        FITS name in ``ModalBases/`` (``.fits`` appended if missing).
        Default ``zernike_<N>.fits`` / ``kl_<N>.fits``.
    save : bool
        Write the FITS file.
    rcond : float
        Cutoff for ``lstsq`` (relative to largest singular value).
    normalize : bool
        Scale each command column to RMS 1.

    Returns
    -------
    cmd_base : ndarray
        Shape ``(n_acts, n_modes)``. YAML: ``modal_base: zernike_50`` (no suffix).
    """
    kind = str(kind).lower()
    if kind not in ("zernike", "kl"):
        raise ValueError("kind must be 'zernike' or 'kl'")

    cube, cmd = _load_intmat_files(tn)
    master = _master_mask(cube)
    g = _zonal_g(_interaction_matrix(cube, master), cmd)
    nacts = g.shape[1]

    indices = (
        list(mode_indices)
        if mode_indices is not None
        else _default_indices(kind, int(n_modes))
    )
    n_out = len(indices)
    n_kl = (max(indices) + 1) if kind == "kl" else n_out
    fitter = _make_fitter(kind, master, n_kl=max(n_kl, n_out))

    cmd_base = np.zeros((nacts, n_out), dtype=np.float64)
    for j, idx in enumerate(indices):
        s = _surface_on_mask(fitter, kind, int(idx), master)
        c, *_ = np.linalg.lstsq(g, s, rcond=rcond)
        if normalize:
            rms = float(np.sqrt(np.mean(c * c)))
            if rms > 0:
                c = c / rms
        cmd_base[:, j] = c

    if save:
        if save_name is None:
            save_name = f"{kind}_{n_out}.fits"
        if not save_name.endswith(".fits"):
            save_name = save_name + ".fits"
        os.makedirs(folders.MODALBASE_ROOT_FOLDER, exist_ok=True)
        out = os.path.join(folders.MODALBASE_ROOT_FOLDER, save_name)
        _osu.save_fits(out, cmd_base, overwrite=True)
        yaml_key = save_name[:-5]
        print(f"Saved modal base {cmd_base.shape} to {out}")
        print(f"Set IFFUNC.modal_base: {yaml_key}")
    return cmd_base
