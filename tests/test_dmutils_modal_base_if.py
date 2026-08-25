"""Tests for opticalib.dmutils.modal_base_if.make_modal_base_if."""

import os

import numpy as np
import numpy.ma as ma
import pytest

from opticalib.core.root import folders
from opticalib.dmutils.modal_base_if import make_modal_base_if
from opticalib.ground import osutils
from opticalib.ground.geometry import draw_circular_pupil


def _gaussian(ny, nx, cy, cx, sigma=3.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


def _synthetic_intmat(folder, tn, n_side=3, size=32):
    nacts = n_side * n_side
    pupil = draw_circular_pupil((size, size), radius=size / 2 - 2)
    cube_data = np.zeros((size, size, nacts), dtype=np.float64)
    spacing = size / (n_side + 1)
    k = 0
    for iy in range(n_side):
        for ix in range(n_side):
            cy = spacing * (iy + 1)
            cx = spacing * (ix + 1)
            cube_data[:, :, k] = _gaussian(size, size, cy, cx)
            k += 1
    mask3 = np.broadcast_to(pupil[..., np.newaxis], cube_data.shape)
    cube = ma.masked_array(cube_data, mask=mask3)
    tn_dir = os.path.join(folder, tn)
    os.makedirs(tn_dir, exist_ok=True)
    osutils.save_fits(os.path.join(tn_dir, "IMCube.fits"), cube, overwrite=True)
    osutils.save_fits(
        os.path.join(tn_dir, "cmd_matrix.fits"), np.eye(nacts), overwrite=True
    )
    return nacts


class TestMakeModalBaseIf:
    def test_zernike_shape_and_save(self, temp_dir, monkeypatch):
        intmat = os.path.join(temp_dir, "INTMatrices")
        modal = os.path.join(temp_dir, "ModalBases")
        os.makedirs(intmat, exist_ok=True)
        os.makedirs(modal, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", intmat)
        monkeypatch.setattr(folders, "MODALBASE_ROOT_FOLDER", modal)

        tn = "20200101_000000"
        nacts = _synthetic_intmat(intmat, tn)
        n_modes = 3
        base = make_modal_base_if(tn, kind="zernike", n_modes=n_modes, save=True)

        assert base.shape == (nacts, n_modes)
        assert np.isfinite(base).all()
        out = os.path.join(modal, f"zernike_{n_modes}.fits")
        assert os.path.isfile(out)
        loaded = np.asarray(osutils.load_fits(out))
        np.testing.assert_allclose(loaded, base, rtol=1e-6)

    def test_kl_shape(self, temp_dir, monkeypatch):
        intmat = os.path.join(temp_dir, "INTMatrices")
        modal = os.path.join(temp_dir, "ModalBases")
        os.makedirs(intmat, exist_ok=True)
        os.makedirs(modal, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", intmat)
        monkeypatch.setattr(folders, "MODALBASE_ROOT_FOLDER", modal)

        tn = "20200101_000001"
        nacts = _synthetic_intmat(intmat, tn)
        n_modes = 3
        base = make_modal_base_if(tn, kind="kl", n_modes=n_modes, save=False)

        assert base.shape == (nacts, n_modes)
        assert np.isfinite(base).all()
        assert not os.path.isfile(os.path.join(modal, f"kl_{n_modes}.fits"))

    def test_missing_cube_raises(self, temp_dir, monkeypatch):
        intmat = os.path.join(temp_dir, "INTMatrices")
        os.makedirs(intmat, exist_ok=True)
        monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", intmat)
        with pytest.raises(FileNotFoundError):
            make_modal_base_if("missing_tn", kind="zernike", n_modes=2, save=False)
