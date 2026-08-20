"""
Tests for parallel-grid IF packing and demultiplexing.
"""

import os
from unittest.mock import patch

import numpy as np
import numpy.ma as ma
import pytest

from opticalib.dmutils import iff_parallel as ipar
from opticalib.dmutils import iff_preparation as ifa
from opticalib.dmutils import iff_processing as ifp
from opticalib.ground import osutils


def _regular_grid_coords(nx=5, ny=5):
    """Build (2, nx*ny) act_coord on an integer grid."""
    xs, ys = [], []
    for y in range(ny):
        for x in range(nx):
            xs.append(x)
            ys.append(y)
    return np.array([xs, ys], dtype=float)


class TestPackActuators:
    def test_spacing_invariants(self):
        coords = _regular_grid_coords(4, 4)
        modes = np.arange(16)
        for spacing in (2.0, 6.0):
            groups = ipar.pack_actuators(coords, modes, spacing)
            flat = [a for g in groups for a in g]
            assert sorted(flat) == list(modes)
            assert len(flat) == len(set(flat))
            for g in groups:
                for i, a in enumerate(g):
                    for b in g[i + 1 :]:
                        d = np.linalg.norm(coords[:, a] - coords[:, b])
                        assert d >= spacing - 1e-9

    def test_larger_spacing_more_groups(self):
        coords = _regular_grid_coords(4, 4)
        modes = np.arange(16)
        g2 = ipar.pack_actuators(coords, modes, 2.0)
        g6 = ipar.pack_actuators(coords, modes, 6.0)
        assert len(g6) >= len(g2)

    def test_pad_roundtrip(self):
        groups = [[0, 5, 10], [1], [2, 7]]
        arr = ipar.groups_to_padded_array(groups)
        assert arr.shape == (3, 3)
        assert arr[1, 1] == -1
        assert ipar.padded_array_to_groups(arr) == groups

    def test_invalid_spacing(self):
        coords = _regular_grid_coords(2, 2)
        with pytest.raises(ValueError):
            ipar.pack_actuators(coords, [0, 1], 0)


class TestParallelCmdHistory:
    def _iff_config(self, parallel_spacing=0):
        return {
            "timing": 1,
            "TRIGGER": {
                "trailing_zeros": 0,
                "modes_list": [],
                "amplitude": 0.1,
                "template": [],
                "modal_base": "zonal",
            },
            "REGISTRATION": {
                "trailing_zeros": 0,
                "modes_list": [],
                "amplitude": 0.1,
                "template": [],
                "modal_base": "zonal",
            },
            "IFFUNC": {
                "trailing_zeros": 0,
                "padding_zeros": 0,
                "modes_list": np.arange(16),
                "amplitude": 0.1,
                "template": [1, -1, 1],
                "shuffle": False,
                "n_repetitions": 1,
                "modal_base": "zonal",
                "parallel_spacing": parallel_spacing,
            },
        }

    def _dm_with_coords(self, mock_dm):
        mock_dm.n_acts = 16
        mock_dm.mirrorModes = np.eye(16, dtype=np.float32)
        mock_dm.act_coord = _regular_grid_coords(4, 4)
        return mock_dm

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_parallel_history_shorter_than_sequential(
        self, mock_get_iff_config, mock_dm
    ):
        mock_get_iff_config.return_value = self._iff_config(0)
        dm = self._dm_with_coords(mock_dm)
        modes = np.arange(16)
        prep = ifa.IFFCapturePreparation(dm)
        seq = prep.create_timed_cmd_history(
            modesList=modes, modesAmp=0.1, template=[1, -1, 1], modalBase="zonal"
        )
        prep2 = ifa.IFFCapturePreparation(dm)
        par = prep2.create_timed_cmd_history(
            modesList=modes,
            modesAmp=0.1,
            template=[1, -1, 1],
            modalBase="zonal",
            parallel_spacing=2.0,
        )
        assert par.shape[1] < seq.shape[1]
        assert prep2._parallel_groups is not None
        assert len(prep2._parallel_groups) == prep2._cmdMatrix.shape[1]
        # Each group column has multiple non-zeros matching group size
        for j, g in enumerate(prep2._parallel_groups):
            nz = np.flatnonzero(np.abs(prep2._cmdMatrix[:, j]) > 0)
            assert set(nz.tolist()) == set(g)

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_spacing_zero_matches_sequential(self, mock_get_iff_config, mock_dm):
        mock_get_iff_config.return_value = self._iff_config(0)
        dm = self._dm_with_coords(mock_dm)
        modes = np.arange(8)
        prep_a = ifa.IFFCapturePreparation(dm)
        a = prep_a.create_timed_cmd_history(
            modesList=modes, modesAmp=0.2, template=[1, -1], modalBase="zonal"
        )
        prep_b = ifa.IFFCapturePreparation(dm)
        b = prep_b.create_timed_cmd_history(
            modesList=modes,
            modesAmp=0.2,
            template=[1, -1],
            modalBase="zonal",
            parallel_spacing=0,
        )
        np.testing.assert_allclose(a, b)
        assert prep_b._parallel_groups is None

    @patch("opticalib.dmutils.iff_preparation._rif.get_iff_config")
    def test_info_to_save_includes_parallel_artifacts(
        self, mock_get_iff_config, mock_dm
    ):
        mock_get_iff_config.return_value = self._iff_config(0)
        dm = self._dm_with_coords(mock_dm)
        prep = ifa.IFFCapturePreparation(dm)
        prep.create_timed_cmd_history(
            modesList=np.arange(16),
            modesAmp=0.1,
            template=[1, -1, 1],
            modalBase="zonal",
            parallel_spacing=2.0,
        )
        info = prep.get_info_to_save()
        assert "parallel_groups" in info
        assert "act_coord" in info
        assert info["parallel_groups"].ndim == 2
        assert info["act_coord"].shape == (2, 16)
        assert float(np.asarray(info["parallel_spacing"]).ravel()[0]) == 2.0


class TestDemux:
    def test_demux_recovers_peaks(self):
        h, w = 64, 64
        yy, xx = np.mgrid[0:h, 0:w]
        centers = [(16.0, 16.0), (48.0, 16.0), (16.0, 48.0)]  # (col, row)
        acts = [0, 1, 2]
        # act_coord spaced by 6 units
        act_coord = np.zeros((2, 3), dtype=float)
        act_coord[:, 0] = [0, 0]
        act_coord[:, 1] = [6, 0]
        act_coord[:, 2] = [0, 6]

        data = np.zeros((h, w), dtype=float)
        for (c, r), amp in zip(centers, (1.0, 0.8, 1.2)):
            data += amp * np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * 2.5**2))
        mask = np.zeros((h, w), dtype=bool)
        img = ma.masked_array(data, mask=mask)

        parts = ipar.demux_group_image(img, acts, act_coord, parallel_spacing=6.0)
        assert set(parts) == set(acts)
        for act, (c, r) in zip(acts, centers):
            peak = parts[act]
            peak_data = np.where(peak.mask, 0.0, peak.data)
            pr, pc = np.unravel_index(np.argmax(np.abs(peak_data)), peak_data.shape)
            assert abs(pc - c) <= 2
            assert abs(pr - r) <= 2
            # Neighbor peaks suppressed in data (zeroed), pupil mask retained
            for oc, or_ in centers:
                if (oc, or_) == (c, r):
                    continue
                assert abs(peak_data[int(or_), int(oc)]) < 0.15 * np.max(
                    np.abs(peak_data)
                )
            # Mask matches input pupil (not the isolation circle)
            np.testing.assert_array_equal(peak.mask, img.mask)

    def test_partial_modes_leave_zeros(self):
        h, w = 32, 32
        yy, xx = np.mgrid[0:h, 0:w]
        act_coord = np.array([[0.0, 4.0], [0.0, 0.0]])
        data = np.exp(-((xx - 8) ** 2 + (yy - 16) ** 2) / 8.0) + np.exp(
            -((xx - 24) ** 2 + (yy - 16) ** 2) / 8.0
        )
        img = ma.masked_array(data, mask=False)
        cube = ipar.demux_parallel_cube(
            [img],
            groups=[[0, 1]],
            act_coord=act_coord,
            parallel_spacing=4.0,
            n_acts=5,
        )
        assert len(cube) == 5
        assert np.any(np.abs(np.ma.filled(cube[0], 0)) > 0)
        assert np.any(np.abs(np.ma.filled(cube[1], 0)) > 0)
        assert np.allclose(np.ma.filled(cube[4], 0), 0)

    def test_demux_parallel_modes_rewrites_files(self, temp_dir, monkeypatch):
        from opticalib.core.root import folders

        iff_root = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_root, exist_ok=True)
        monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_root)
        monkeypatch.setattr(ifp, "_ifFold", iff_root)

        tn = "20260101_000000"
        fold = os.path.join(iff_root, tn)
        os.makedirs(fold)

        groups = [[0, 2], [1]]
        act_coord = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        spacing = 3.0
        osutils.save_fits(
            os.path.join(fold, "parallel_groups.fits"),
            ipar.groups_to_padded_array(groups),
            overwrite=True,
        )
        osutils.save_fits(os.path.join(fold, "act_coord.fits"), act_coord, overwrite=True)
        osutils.save_fits(
            os.path.join(fold, "parallel_spacing.fits"),
            np.array([spacing]),
            overwrite=True,
        )

        h, w = 40, 40
        yy, xx = np.mgrid[0:h, 0:w]
        # Group 0: peaks at acts 0 and 2
        g0 = np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / 8) + np.exp(
            -((xx - 10) ** 2 + (yy - 30) ** 2) / 8
        )
        g1 = np.exp(-((xx - 30) ** 2 + (yy - 10) ** 2) / 8)
        osutils.save_fits(
            os.path.join(fold, "mode_00000.fits"),
            ma.masked_array(g0, mask=False),
            overwrite=True,
        )
        osutils.save_fits(
            os.path.join(fold, "mode_00001.fits"),
            ma.masked_array(g1, mask=False),
            overwrite=True,
        )
        osutils.save_fits(
            os.path.join(fold, "modes_list.fits"),
            np.array([0, 1]),
            overwrite=True,
            header={"N_REP": 1, "SHUFFLE": False, "PAR_SPC": spacing},
        )
        osutils.save_fits(
            os.path.join(fold, "amplitude.fits"),
            np.array([0.1, 0.1]),
            overwrite=True,
        )
        osutils.save_fits(
            os.path.join(fold, "cmd_matrix.fits"),
            np.eye(3)[:, [0, 1]],
            overwrite=True,
        )
        osutils.save_fits(
            os.path.join(fold, "index_list.fits"), np.array([0, 1]), overwrite=True
        )
        osutils.save_fits(
            os.path.join(fold, "template.fits"), np.array([1, -1, 1]), overwrite=True
        )
        osutils.save_fits(
            os.path.join(fold, "registration_modes.fits"),
            np.array([]),
            overwrite=True,
        )

        info = {
            "FILES": {
                "amplitude": np.array([0.1, 0.1]),
                "modes_list": np.array([0, 1]),
                "template": np.array([1, -1, 1]),
                "index_list": np.array([0, 1]),
                "registration_modes": np.array([]),
                "shuffle": False,
                "n_repetitions": 1,
            },
            "IFFUNC": {
                "modes_list": [0, 1],
                "amplitude": 0.1,
                "template": [1, -1, 1],
                "parallel_spacing": spacing,
            },
        }

        with patch.object(ifp._rif, "update_iff_config"):
            measured = ifp.demux_parallel_modes(tn, info)

        assert sorted(measured) == [0, 1, 2]
        for a in measured:
            assert os.path.isfile(os.path.join(fold, f"mode_{a:05d}.fits"))
        modes = osutils.load_fits(os.path.join(fold, "modes_list.fits"))
        np.testing.assert_array_equal(np.asarray(modes), [0, 1, 2])

    def test_missing_parallel_metadata(self, temp_dir, monkeypatch):
        from opticalib.core.root import folders

        iff_root = os.path.join(temp_dir, "IFFunctions")
        os.makedirs(iff_root, exist_ok=True)
        monkeypatch.setattr(ifp, "_ifFold", iff_root)
        tn = "20260101_000001"
        os.makedirs(os.path.join(iff_root, tn))
        assert ifp._has_parallel_metadata(tn) is False
