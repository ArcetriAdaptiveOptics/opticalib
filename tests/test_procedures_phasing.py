"""
Characterization tests for :class:`opticalib.procedures.phasing.SPL`.

These tests pin the CURRENT observable behaviour (files written, FITS headers,
device call order, returned pistons) and act as a regression baseline for a
refactor. Known bugs are pinned with strict xfails asserting the CORRECT
behaviour.
"""

import os

import numpy as np
import pytest

from opticalib.procedures import phasing
from opticalib.ground import osutils as osu

FRAME = 300
BKG = 10.0
LAMBDAS = np.array([450, 500, 550, 600, 650])


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeFilter:
    """Tunable filter fake recording into a shared ordered list."""

    def __init__(self, calls):
        self.calls = calls
        self.wl = None
        self.mode = None

    def move_to(self, wl):
        self.calls.append(("move_to", wl))
        self.wl = wl

    def set_bandwidth_mode(self, n):
        self.calls.append(("set_bandwidth_mode", n))
        self.mode = n


class FakeCamera:
    """Camera fake producing a synthetic fringed PSF depending on wavelength."""

    def __init__(self, calls, flt, blobs=((150, 150),)):
        self.calls = calls
        self.filter = flt
        self.blobs = blobs
        self.rng = np.random.default_rng(1234)
        self.exptimes_us = []
        self.acq_modes = []  # filter mode active at each acquisition
        self.acq_wls = []

    def set_exptime(self, exptime_us):
        self.calls.append(("set_exptime", exptime_us))
        self.exptimes_us.append(exptime_us)

    def acquire_frames(self, nframes):
        self.calls.append(("acquire_frames", nframes))
        self.acq_modes.append(self.filter.mode)
        self.acq_wls.append(self.filter.wl)
        img = BKG + self.rng.normal(0, 1.0, (FRAME, FRAME))
        if self.filter.mode == 1:  # black: background only
            return img
        yy, xx = np.mgrid[0:FRAME, 0:FRAME]
        period = (self.filter.wl or 500) / 10.0
        for (cx, cy) in self.blobs:
            env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0**2))
            img = img + 1000.0 * env * (1 + 0.8 * np.cos(2 * np.pi * yy / period))
        return img


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "SPL"
    fringes = tmp_path / "SPLFringes"
    data.mkdir()
    fringes.mkdir()
    monkeypatch.setattr(phasing._fn, "SPL_DATA_ROOT_FOLDER", str(data), raising=False)
    monkeypatch.setattr(
        phasing._fn, "SPL_FRINGES_ROOT_FOLDER", str(fringes), raising=False
    )
    monkeypatch.setattr(phasing, "_splconf", {})
    # tracking-number lookups (get_file_list) walk osutils._OPTDATA
    monkeypatch.setattr(osu, "_OPTDATA", str(tmp_path))
    calls = []
    flt = FakeFilter(calls)
    cam = FakeCamera(calls, flt)
    return dict(
        data=str(data), fringes=str(fringes), calls=calls, filter=flt, camera=cam
    )


@pytest.fixture
def spl(env):
    s = phasing.SPL(env["camera"], env["filter"], "tnfr")
    # acquire() needs a known filter mode; do not pollute the call recorder
    s.set_filter_mode("narrow")
    env["calls"].clear()
    return s


@pytest.fixture
def spl_nomode(env):
    """SPL instance for which no filter mode was ever set."""
    return phasing.SPL(env["camera"], env["filter"], "tnfr")


class _ReadableFilter(FakeFilter):
    """FakeFilter exposing the plico `get_bandwidth_mode` getter."""

    def __init__(self, calls, code):
        super().__init__(calls)
        self.mode = code

    def get_bandwidth_mode(self):
        return self.mode


def _moves(calls):
    return [c[1] for c in calls if c[0] == "move_to"]


def _hdr(path):
    return osu.load_fits(path).header


# --------------------------------------------------------------------------
# acquire
# --------------------------------------------------------------------------


class TestAcquire:
    def test_creates_files(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=LAMBDAS)
        folder = os.path.join(env["data"], tn)
        assert os.path.isdir(folder)
        files = set(os.listdir(folder))
        assert "lambda_vector.fits" in files
        assert "darkFrame.fits" in files
        raws = sorted(f for f in files if f.startswith("rawframe"))
        assert len(raws) == len(LAMBDAS)
        assert files == {"lambda_vector.fits", "darkFrame.fits"} | set(raws)
        assert spl._last_measure_tn == tn
        np.testing.assert_array_equal(
            np.asarray(osu.load_fits(os.path.join(folder, "lambda_vector.fits"))),
            LAMBDAS,
        )

    def test_rawframe_filename_format_numpy_ints(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=LAMBDAS)
        files = set(os.listdir(os.path.join(env["data"], tn)))
        for wl in LAMBDAS:
            assert f"rawframe_{int(wl)}nm.fits" in files

    def test_rawframe_filename_format_floats(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=np.array([450.0, 500.5]))
        files = set(os.listdir(os.path.join(env["data"], tn)))
        assert "rawframe_450.0nm.fits" in files
        assert "rawframe_500.5nm.fits" in files

    def test_headers_and_exptimes(self, env, spl):
        tn = spl.acquire(2.0, lambda_vector=LAMBDAS)
        folder = os.path.join(env["data"], tn)
        # gain vector: <530 -> 2 ; <550 -> 1 ; >650 -> 1 ; else 0.5
        expected_gain = {450: 2.0, 500: 2.0, 550: 0.5, 600: 0.5, 650: 0.5}
        for wl in LAMBDAS:
            h = _hdr(os.path.join(folder, f"rawframe_{wl}nm.fits"))
            assert h["WAVELEN"] == wl
            assert h["EXPTIME"] == pytest.approx(expected_gain[int(wl)] * 2.0)
            assert h["EXPGAIN"] == pytest.approx(expected_gain[int(wl)])

    def test_camera_exptimes_in_microseconds(self, env, spl):
        env["calls"].clear()
        spl.acquire(2.0, lambda_vector=LAMBDAS)
        # dark at 1 s first, then per-wavelength 4,4,1,1,1 s; repeated values
        # are skipped by set_exptime (current != requested only)
        assert env["camera"].exptimes_us == [1e6, 4e6, 1e6]

    def test_exptime_skip_when_unchanged(self, env, spl):
        spl.set_exptime(1.5)
        spl.set_exptime(1.5)
        assert env["camera"].exptimes_us == [1.5e6]

    def test_move_order_and_final_600(self, env, spl):
        spl.acquire(1.0, lambda_vector=LAMBDAS[::-1])
        moves = _moves(env["calls"])
        assert moves == list(LAMBDAS[::-1]) + [600]

    def test_acquire_frames_each_wavelength_after_move(self, env, spl):
        spl.acquire(1.0, lambda_vector=LAMBDAS, nframes=3)
        sweep = [c for c in env["calls"] if c[0] in ("move_to", "acquire_frames")]
        # first acquire is the dark (before any move); then move/acquire pairs
        assert sweep[0] == ("acquire_frames", 1)
        body = sweep[1:-1]
        for i, wl in enumerate(LAMBDAS):
            assert body[2 * i] == ("move_to", wl)
            assert body[2 * i + 1] == ("acquire_frames", 3)
        assert sweep[-1] == ("move_to", 600)

    def test_default_lambda_vector(self, env, spl):
        tn = spl.acquire(1.0)
        got = np.asarray(osu.load_fits(os.path.join(env["data"], tn, "lambda_vector.fits")))
        np.testing.assert_array_equal(got, np.arange(440, 721, 20))
        assert got.max() == 720
        assert _moves(env["calls"]) == list(np.arange(440, 721, 20)) + [600]

    @pytest.mark.parametrize("bad", [[399, 500], [500, 721], [300], [800, 450]])
    def test_out_of_range_raises(self, env, spl, bad):
        with pytest.raises(ValueError):
            spl.acquire(1.0, lambda_vector=bad)
        assert env["calls"] == []
        assert os.listdir(env["data"]) == []

    def test_range_boundaries_accepted(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=[400, 720])
        folder = os.path.join(env["data"], tn)
        assert os.path.exists(os.path.join(folder, "rawframe_400nm.fits"))
        assert os.path.exists(os.path.join(folder, "rawframe_720nm.fits"))

    def test_single_wavelength(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=[500])
        assert _moves(env["calls"]) == [500, 600]
        assert os.path.exists(
            os.path.join(env["data"], tn, "rawframe_500nm.fits")
        )

    def test_invalid_filter_mode_raises(self, env, spl):
        with pytest.raises(ValueError):
            spl.acquire(1.0, filter_mode="bogus", lambda_vector=LAMBDAS)
        with pytest.raises(ValueError):
            spl.set_filter_mode("bogus")
        assert env["calls"] == []

    @pytest.mark.parametrize(
        "mode,n", [("narrow", 8), ("medium", 4), ("wide", 2), ("black", 1)]
    )
    def test_set_filter_mode_values(self, env, spl, mode, n):
        spl.set_filter_mode(mode)
        assert env["calls"] == [("set_bandwidth_mode", n)]

    def test_dark_preacquired_is_saved_not_reacquired(self, env, spl):
        dark = spl.acquire_dark_frame(2.0, nframes=1)
        assert dark.header["EXPTIME"] == 2.0
        assert dark.header["ISDARK"] is True
        env["calls"].clear()
        tn = spl.acquire(1.0, lambda_vector=LAMBDAS)
        # exactly nwl frames acquired: no dark inside acquire
        # without `filter_mode` the sweep runs in the remembered mode (narrow, 8)
        assert env["camera"].acq_modes[1:] == [8] * len(LAMBDAS)
        assert [c for c in env["calls"] if c[0] == "acquire_frames"] == [
            ("acquire_frames", 1)
        ] * len(LAMBDAS)
        saved = osu.load_fits(os.path.join(env["data"], tn, "darkFrame.fits"))
        assert saved.header["EXPTIME"] == 2.0
        assert saved.header["ISDARK"]
        assert saved.shape == (FRAME, FRAME)

    def test_dark_acquired_inside_acquire(self, env, spl):
        assert spl._darkFrame1sec is None
        tn = spl.acquire(3.0, lambda_vector=LAMBDAS)
        assert spl._darkFrame1sec is not None
        # the dark is acquired in black mode (1) before any sweep move
        assert env["calls"][0] == ("set_exptime", 1e6)
        assert env["calls"][1] == ("set_bandwidth_mode", 1)
        assert env["camera"].acq_modes[0] == 1
        saved = osu.load_fits(os.path.join(env["data"], tn, "darkFrame.fits"))
        assert saved.header["EXPTIME"] == 1.0
        assert saved.header["NFRAMES"] == 1
        assert saved.header["ISRAW"]

    def test_dark_frame_scaled_by_exptime(self, env, spl):
        dark = spl.acquire_dark_frame(4.0, nframes=1)
        # background ~10 counts / 4 s
        assert np.median(np.asarray(dark)) == pytest.approx(BKG / 4.0, rel=0.05)
        assert env["calls"][:3] == [
            ("set_exptime", 4e6),
            ("set_bandwidth_mode", 1),
            ("acquire_frames", 1),
        ]

    def test_filter_mode_with_preacquired_dark(self, env, spl):
        spl.acquire_dark_frame(1.0)
        env["calls"].clear()
        spl.acquire(1.0, filter_mode="narrow", lambda_vector=LAMBDAS)
        assert env["calls"][0] == ("set_bandwidth_mode", 8)
        assert env["camera"].acq_modes[1:] == [8] * len(LAMBDAS)

    def test_filter_mode_without_dark_sweeps_in_requested_mode(self, env, spl):
        spl.acquire(1.0, filter_mode="narrow", lambda_vector=LAMBDAS)
        # first acquisition is the dark; the sweep frames follow
        assert env["camera"].acq_modes[1:] == [8] * len(LAMBDAS)

    def test_filter_mode_without_dark_call_order(self, env, spl):
        """Dark (black) first, then the requested mode, then the sweep."""
        spl.acquire(1.0, filter_mode="medium", lambda_vector=LAMBDAS)
        modes = [c[1] for c in env["calls"] if c[0] == "set_bandwidth_mode"]
        # fixture mode (narrow, 8) is restored after the dark, then "medium"
        assert modes == [1, 8, 4]
        assert env["camera"].acq_modes[0] == 1
        assert env["camera"].acq_modes[1:] == [4] * len(LAMBDAS)

    def test_black_sweep_mode_rejected_before_side_effects(self, env, spl):
        with pytest.raises(ValueError):
            spl.acquire(1.0, filter_mode="black", lambda_vector=LAMBDAS)
        # no filter call, no dark acquisition, no data folder
        assert env["calls"] == []
        assert env["camera"].acq_modes == []
        assert spl._darkFrame1sec is None
        assert os.listdir(env["data"]) == []

    def test_preacquired_dark_without_any_mode_raises(self, env, spl_nomode):
        spl_nomode.acquire_dark_frame(1.0)
        env["calls"].clear()
        with pytest.raises(ValueError):
            spl_nomode.acquire(1.0, lambda_vector=LAMBDAS)
        assert env["calls"] == []
        assert os.listdir(env["data"]) == []

    def test_preacquired_dark_sweeps_in_remembered_mode(self, env, spl_nomode):
        spl_nomode.set_filter_mode("wide")
        spl_nomode.acquire_dark_frame(1.0)
        env["calls"].clear()
        spl_nomode.acquire(1.0, lambda_vector=LAMBDAS)
        assert env["camera"].acq_modes[1:] == [2] * len(LAMBDAS)

    def test_no_mode_known_raises_before_side_effects(self, env, spl_nomode):
        with pytest.raises(ValueError):
            spl_nomode.acquire(1.0, filter_mode=None, lambda_vector=LAMBDAS)
        assert env["calls"] == []
        assert os.listdir(env["data"]) == []

    def test_explicit_vector_720_accepted_721_rejected(self, env, spl):
        tn = spl.acquire(1.0, lambda_vector=[720])
        assert os.path.exists(os.path.join(env["data"], tn, "rawframe_720nm.fits"))
        env["calls"].clear()
        with pytest.raises(ValueError):
            spl.acquire(1.0, lambda_vector=[721])
        assert env["calls"] == []

    def test_none_uses_filters_current_mode(self, env, spl_nomode):
        flt = _ReadableFilter(env["calls"], 2)
        env["camera"].filter = flt
        spl = phasing.SPL(env["camera"], flt, "tnfr")
        spl._filter_mode = "narrow"  # remembered differs from the current one
        spl.acquire(1.0, lambda_vector=LAMBDAS)
        assert env["camera"].acq_modes[1:] == [2] * len(LAMBDAS)

    def test_none_black_filter_falls_back_to_remembered(self, env):
        flt = _ReadableFilter(env["calls"], 1)
        env["camera"].filter = flt
        spl = phasing.SPL(env["camera"], flt, "tnfr")
        spl._filter_mode = "medium"
        spl.acquire(1.0, lambda_vector=LAMBDAS)
        assert env["camera"].acq_modes[1:] == [4] * len(LAMBDAS)

    def test_none_black_filter_nothing_remembered_raises(self, env):
        flt = _ReadableFilter(env["calls"], 1)
        spl = phasing.SPL(env["camera"], flt, "tnfr")
        with pytest.raises(ValueError):
            spl.acquire(1.0, lambda_vector=LAMBDAS)
        assert env["calls"] == []
        assert os.listdir(env["data"]) == []

    def test_dark_restores_mode_filter_had_before(self, env):
        flt = _ReadableFilter(env["calls"], 4)
        env["camera"].filter = flt
        spl = phasing.SPL(env["camera"], flt, "tnfr")
        spl._filter_mode = "narrow"
        spl.acquire_dark_frame(1.0)
        assert spl._filter_mode == "medium"
        assert env["camera"].acq_modes == [1]
        assert flt.mode == 4

    def test_dark_restores_mode_when_camera_raises(self, env):
        class _BrokenCamera(FakeCamera):
            def acquire_frames(self, nframes):
                self.calls.append(("acquire_frames", nframes))
                raise RuntimeError("camera down")

        flt = _ReadableFilter(env["calls"], 8)  # narrow
        cam = _BrokenCamera(env["calls"], flt)
        spl = phasing.SPL(cam, flt, "tnfr")
        assert spl._filter_mode == "narrow"
        env["calls"].clear()
        with pytest.raises(RuntimeError, match="camera down"):
            spl.acquire_dark_frame(1.0)
        # black was set for the dark, then the remembered mode was restored
        assert env["calls"] == [
            ("set_exptime", 1e6),
            ("set_bandwidth_mode", 1),
            ("acquire_frames", 1),
            ("set_bandwidth_mode", 8),
        ]
        assert flt.mode == 8
        assert spl._filter_mode == "narrow"

    def test_masked_input_mask_used(self, env, spl):
        mask = np.zeros((FRAME, FRAME), dtype=bool)
        mask[0, 0] = True
        tn = spl.acquire(1.0, lambda_vector=[500], mask=mask)
        img = osu.load_fits(os.path.join(env["data"], tn, "rawframe_500nm.fits"))
        assert np.ma.getmaskarray(img)[0, 0]


# --------------------------------------------------------------------------
# constructor / small helpers
# --------------------------------------------------------------------------


class TestConstructor:
    def test_camera_none_string(self, env):
        s = phasing.SPL("none", env["filter"], None)
        assert s._camera is None
        assert s._fringes_fold is None

    def test_init_seeds_filter_mode_from_filter(self, env):
        flt = _ReadableFilter(env["calls"], 8)
        assert phasing.SPL(env["camera"], flt, None)._filter_mode == "narrow"
        flt = _ReadableFilter(env["calls"], 1)
        assert phasing.SPL(env["camera"], flt, None)._filter_mode is None

    def test_set_tn_fringes(self, env, spl):
        assert spl._fringes_fold == os.path.join(env["fringes"], "tnfr")
        spl.set_tn_fringes(None)
        assert spl._fringes_fold is None and spl._tnfringes is None
        spl.set_tn_fringes("abc")
        assert spl._fringes_fold == os.path.join(env["fringes"], "abc")


# --------------------------------------------------------------------------
# process_psfs / analysis
# --------------------------------------------------------------------------


def _make_templates(env, matrix, lambda_vector, M=5, k=3, seed=0):
    """Fringes folder where Fringe_{k:05d} equals `matrix` in the selected columns."""
    fold = os.path.join(env["fringes"], "tnfr")
    os.makedirs(fold, exist_ok=True)
    delta = np.array([0.0, 10.0, 20.0, 30.0, 40.0])[:M] * 1.0
    lam = np.arange(400, 705, 5, dtype=float)
    osu.save_fits(os.path.join(fold, "Differential_piston.fits"), delta)
    osu.save_fits(os.path.join(fold, "Lambda.fits"), lam)
    idx = np.isin(np.round(lam / 5).astype(int) * 5, lambda_vector)
    rng = np.random.default_rng(seed)
    for i in range(1, M):
        if i == k:
            f = rng.normal(0, 1e-3, (matrix.shape[0], lam.size))
            f[:, idx] = np.asarray(matrix)
        else:
            f = rng.normal(0, 1, (matrix.shape[0], lam.size))
        osu.save_fits(os.path.join(fold, "Fringe_%05d.fits" % i), f)
    return delta


class TestProcessAndAnalysis:
    def _acquire(self, spl):
        # Pre-acquire the dark and request a non-black mode: otherwise the
        # current code leaves the filter in black mode and frames are empty.
        spl.acquire_dark_frame(1.0)
        return spl.acquire(1.0, filter_mode="narrow", lambda_vector=LAMBDAS)

    def test_process_without_tn_raises(self, env, spl):
        with pytest.raises(ValueError):
            spl.process_psfs(n_psfs=1)

    def test_process_outputs(self, env, spl):
        tn = self._acquire(spl)
        spl.process_psfs(tn=tn, n_psfs=1, initial_half_size=100)
        folder = os.path.join(env["data"], tn)
        files = set(os.listdir(folder))
        for wl in LAMBDAS:
            assert f"rawpsf0_{wl}nm.fits" in files
            assert f"rot_psf0_{wl}nm.fits" in files
        assert "psf0_cube.fits" in files
        cube = osu.load_fits(os.path.join(folder, "psf0_cube.fits"), memmap=False)
        assert cube.shape == (80, 80, len(LAMBDAS))
        h = _hdr(os.path.join(folder, f"rawpsf0_{LAMBDAS[0]}nm.fits"))
        assert h["PSFNUM"] == 0
        assert h["CENTX"] == pytest.approx(150, abs=2)
        assert h["CENTY"] == pytest.approx(150, abs=2)
        h = _hdr(os.path.join(folder, f"rot_psf0_{LAMBDAS[0]}nm.fits"))
        assert h["ROTATED"] is False
        assert h["ROTANG"] == 0.0

    def test_process_uses_last_measure_tn(self, env, spl):
        tn = self._acquire(spl)
        spl.process_psfs(n_psfs=1, initial_half_size=100)
        assert os.path.exists(os.path.join(env["data"], tn, "psf0_cube.fits"))

    def test_process_final_half_size(self, env, spl):
        tn = self._acquire(spl)
        spl.process_psfs(
            tn=tn, n_psfs=1, initial_half_size=100, final_half_size=30
        )
        cube = osu.load_fits(os.path.join(env["data"], tn, "psf0_cube.fits"), memmap=False)
        assert cube.shape == (60, 60, len(LAMBDAS))

    def test_end_to_end_recovers_piston(self, env, spl):
        tn = self._acquire(spl)
        spl.process_psfs(tn=tn, n_psfs=1, initial_half_size=100)
        folder = os.path.join(env["data"], tn)
        cube = osu.load_fits(os.path.join(folder, "psf0_cube.fits"), memmap=False)
        matrix, _ = spl._compute_matrix(LAMBDAS, cube)
        assert matrix.shape == (80, len(LAMBDAS))

        k = 3  # file Fringe_00003 is the matching template
        delta = _make_templates(env, matrix, LAMBDAS, M=5, k=k)

        pists = spl.analysis(tn=tn, n_psfs=1, initial_half_size=100)

        # Templates are read for i = 1..M-1 and the result is delta[idp] with
        # idp the 0-based index in that list: Fringe_{k} -> delta[k-1].
        # (Not delta[k]: current mapping is shifted by one w.r.t. file number.)
        assert len(pists) == 1
        assert pists[0] == delta[k - 1]
        assert pists[0] != delta[k]

        res = os.path.join(folder, "psf0_fringes_result.fits")
        assert os.path.exists(res)
        h = _hdr(res)
        assert h["DPIST"] == delta[k - 1]
        assert "DPISTSM" in h
        assert h["TEMPIDX"] == k - 1
        np.testing.assert_array_equal(spl._last_lambdavec, LAMBDAS)

    def test_analysis_without_tn_uses_last_measurement(self, env, spl):
        tn = self._acquire(spl)
        spl.process_psfs(tn=tn, n_psfs=1, initial_half_size=100)
        cube = osu.load_fits(os.path.join(env["data"], tn, "psf0_cube.fits"), memmap=False)
        matrix, _ = spl._compute_matrix(LAMBDAS, cube)
        delta = _make_templates(env, matrix, LAMBDAS, M=5, k=2)
        pists = spl.analysis(n_psfs=1, initial_half_size=100)
        assert pists == [delta[1]]

    def test_analysis_without_fringes_folder_raises(self, env):
        s = phasing.SPL(env["camera"], env["filter"], None)
        s.acquire_dark_frame(1.0)
        tn = s.acquire(1.0, filter_mode="narrow", lambda_vector=LAMBDAS)
        with pytest.raises(
            ValueError, match="No fringe templates tracking number set"
        ):
            s.analysis(tn=tn, n_psfs=1, initial_half_size=100)


# --------------------------------------------------------------------------
# detection and cropping
# --------------------------------------------------------------------------


def _blob_frame(centres, size=300, sigma=8.0, amp=500.0, seed=0):
    rng = np.random.default_rng(seed)
    img = 10 + rng.normal(0, 1.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for (cx, cy) in centres:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return img


class TestDetectAndCrop:
    def test_detect_single_psf(self, spl):
        c = spl.detect_psf_centroids(_blob_frame([(120, 90)]), n_psfs=1)
        assert len(c) == 1
        assert all(isinstance(v, int) for v in c[0])
        assert abs(c[0][0] - 120) <= 1 and abs(c[0][1] - 90) <= 1

    def test_detect_two_psfs_sorted_by_y_then_x(self, spl):
        # (x, y): the PSF with smaller y comes first, regardless of x
        img = _blob_frame([(60, 220), (230, 60)])
        c = spl.detect_psf_centroids(img, n_psfs=2)
        assert len(c) == 2
        assert abs(c[0][0] - 230) <= 1 and abs(c[0][1] - 60) <= 1
        assert abs(c[1][0] - 60) <= 1 and abs(c[1][1] - 220) <= 1

    @pytest.mark.parametrize("pos", [None, [], [[], []], [[1, 2], []]])
    def test_unfilled_expected_psf_pos_falls_back_to_y_x(self, spl, monkeypatch, pos):
        # template-style unfilled entries must not crash nor change the order
        monkeypatch.setattr(phasing, "_splconf", {"expected_psf_pos": pos})
        c = spl.detect_psf_centroids(_blob_frame([(60, 220), (230, 60)]), n_psfs=2)
        assert abs(c[0][0] - 230) <= 1 and abs(c[1][0] - 60) <= 1

    def test_expected_psf_pos_is_x_y_and_sets_order(self, spl, monkeypatch):
        # [x, y] pairs: detections are ordered as the expected positions
        monkeypatch.setattr(
            phasing, "_splconf", {"expected_psf_pos": [[62, 218], [228, 62]]}
        )
        c = spl.detect_psf_centroids(_blob_frame([(60, 220), (230, 60)]), n_psfs=2)
        assert abs(c[0][0] - 60) <= 1 and abs(c[0][1] - 220) <= 1
        assert abs(c[1][0] - 230) <= 1 and abs(c[1][1] - 60) <= 1

    def test_detect_n_psfs_one_keeps_largest(self, spl):
        img = _blob_frame([(60, 60)], sigma=5.0) + _blob_frame(
            [(220, 220)], sigma=12.0, seed=1
        ) - 10
        c = spl.detect_psf_centroids(img, n_psfs=1)
        assert len(c) == 1
        assert abs(c[0][0] - 220) <= 1 and abs(c[0][1] - 220) <= 1

    def test_detect_min_pixels_filters_everything(self, spl):
        c = spl.detect_psf_centroids(
            _blob_frame([(100, 100)], sigma=3.0), n_psfs=1, min_pixels=10**6
        )
        assert c == []

    def test_detect_flat_frame_empty(self, spl):
        assert spl.detect_psf_centroids(np.full((100, 100), 5.0), n_psfs=1) == []

    def test_detect_two_psfs_requested_but_one_present(self, spl):
        c = spl.detect_psf_centroids(_blob_frame([(100, 100)]), n_psfs=2)
        assert len(c) == 1

    def test_crop_interior(self, spl):
        img = np.arange(100 * 100, dtype=float).reshape(100, 100)
        crops, boxes = spl.crop_around_centroids(img, [(50, 40)], half_size=10)
        assert len(crops) == 1
        assert crops[0].shape == (20, 20)
        assert boxes == [(40, 30, 20, 20)]
        np.testing.assert_array_equal(crops[0], img[30:50, 40:60])

    def test_crop_two_centroids_preserve_order(self, spl):
        img = np.zeros((100, 100))
        _, boxes = spl.crop_around_centroids(img, [(20, 20), (70, 60)], half_size=5)
        assert boxes == [(15, 15, 10, 10), (65, 55, 10, 10)]

    def test_crop_clipped_at_edges(self, spl):
        img = np.zeros((100, 120))
        crops, boxes = spl.crop_around_centroids(
            img, [(3, 4), (118, 98)], half_size=10
        )
        assert boxes[0] == (0, 0, 13, 14)
        assert crops[0].shape == (14, 13)
        assert boxes[1] == (108, 88, 12, 12)
        assert crops[1].shape == (12, 12)

    def test_crop_half_size_list_x_then_y(self, spl):
        img = np.zeros((100, 100))
        crops, boxes = spl.crop_around_centroids(img, [(50, 50)], half_size=[5, 10])
        assert boxes == [(45, 40, 10, 20)]
        assert crops[0].shape == (20, 10)

    def test_crop_default_half_size_is_75(self, spl):
        img = np.zeros((300, 300))
        _, boxes = spl.crop_around_centroids(img, [(150, 150)])
        assert boxes == [(75, 75, 150, 150)]

    def test_crop_is_copy_and_empty_centroids(self, spl):
        img = np.zeros((50, 50))
        crops, _ = spl.crop_around_centroids(img, [(25, 25)], half_size=5)
        crops[0][:] = 9
        assert img.max() == 0
        assert spl.crop_around_centroids(img, [], half_size=5) == ([], [])


# --------------------------------------------------------------------------
# bad-pixel healing
# --------------------------------------------------------------------------


class TestHealBadPixels:
    def _img(self):
        yy, xx = np.mgrid[0:40, 0:40]
        return 100.0 + 0.1 * xx + 0.1 * yy

    def test_hot_pixel_healed(self, spl):
        img = self._img()
        img[20, 20] += 5000.0
        orig = img.copy()
        out = spl._heal_bad_pixels(img)
        assert out.shape == img.shape
        assert out[20, 20] < 200
        np.testing.assert_array_equal(img, orig)  # input not mutated
        # everything else is untouched
        mask = np.ones(img.shape, bool)
        mask[20, 20] = False
        np.testing.assert_array_equal(out[mask], orig[mask])

    def test_clean_image_returned_unchanged(self, spl):
        # with zero hot pixels nothing is healed
        img = self._img()
        out = spl._heal_bad_pixels(img)
        assert out.shape == img.shape
        np.testing.assert_array_equal(out, img)
        assert out is not img

    def test_constant_image_unchanged(self, spl):
        img = np.full((20, 20), 7.0)
        np.testing.assert_array_equal(spl._heal_bad_pixels(img), img)

    @pytest.mark.parametrize("method", ["median", "mean", "gaussian", np.min])
    def test_methods_preserve_shape(self, spl, method):
        img = self._img()
        img[10, 10] += 5000.0
        out = spl._heal_bad_pixels(img, method=method)
        assert out.shape == img.shape
        assert out[10, 10] < img[10, 10]

    def test_unknown_method_raises(self, spl):
        img = self._img()
        img[10, 10] += 5000.0
        with pytest.raises(ValueError):
            spl._heal_bad_pixels(img, method="nope")


# --------------------------------------------------------------------------
# generalization: "none" devices, _save_frame, datapath, module-level analysis
# --------------------------------------------------------------------------


class TestReadFilterMode:
    @pytest.mark.parametrize("code,name", [(8, "narrow"), (4, "medium"), (2, "wide")])
    def test_codes(self, code, name):
        assert phasing._read_filter_mode(_ReadableFilter([], code)) == name

    def test_black_is_none(self):
        assert phasing._read_filter_mode(_ReadableFilter([], 1)) is None

    def test_none_and_missing_getter(self):
        assert phasing._read_filter_mode(None) is None
        assert phasing._read_filter_mode(FakeFilter([])) is None

    def test_getter_raising(self):
        class Bad(FakeFilter):
            def get_bandwidth_mode(self):
                raise RuntimeError("boom")

        assert phasing._read_filter_mode(Bad([])) is None

    def test_unknown_code(self):
        assert phasing._read_filter_mode(_ReadableFilter([], 99)) is None


class TestComputeMatrixNaN:
    def test_nan_column_zeroed(self, spl):
        cube = np.ones((6, 6, 3))
        cube[:, :, 1] = 0  # zero area -> 0/0 -> NaN
        cube = np.ma.masked_array(cube, mask=np.zeros(cube.shape, bool))
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix, _ = spl._compute_matrix(np.array([450, 500, 550]), cube)
        assert not np.isnan(np.asarray(matrix)).any()
        assert np.all(np.asarray(matrix)[:, 1] == 0)


class TestNoneDevices:
    def test_tunable_filter_none_string_no_connection(self, env, monkeypatch):
        def boom():
            raise AssertionError("tunable filter connection attempted")

        monkeypatch.setattr(phasing, "_get_tunable_filter", boom)
        for value in ("none", "None"):
            s = phasing.SPL("none", value, None)
            assert s._filter is None
            assert s._camera is None


class TestSaveFrame:
    def test_header_merge_and_mask_default(self, env, spl, tmp_path):
        img = np.full((8, 8), 3.0)
        spl._save_frame(
            str(tmp_path), img, 500, exptime=2.0, expgain=0.5,
            header={"CAMNAME": "x", "EXPGAIN": 9.0},
        )
        out = osu.load_fits(str(tmp_path / "rawframe_500nm.fits"))
        h = out.header
        assert h["EXPTIME"] == 2.0 and h["WAVELEN"] == 500
        assert h["CAMNAME"] == "x"
        assert h["EXPGAIN"] == 9.0  # extra header wins
        assert not np.ma.getmaskarray(out).any()

    def test_explicit_mask(self, env, spl, tmp_path):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 2] = True
        spl._save_frame(str(tmp_path), np.ones((8, 8)), 450.5, 1.0, 1.0, mask=mask)
        out = osu.load_fits(str(tmp_path / "rawframe_450.5nm.fits"))
        assert np.ma.getmaskarray(out)[1, 2]

    def test_does_not_use_instance_state(self, env, spl, tmp_path):
        spl._curr_exptime = 123.0
        before = dict(vars(spl))
        spl._save_frame(str(tmp_path), np.ones((8, 8)), 600, exptime=2.0, expgain=0.5)
        assert vars(spl) == before
        h = _hdr(str(tmp_path / "rawframe_600nm.fits"))
        assert h["EXPTIME"] == 2.0
        assert h["EXPGAIN"] == 0.5
        assert h["WAVELEN"] == 600


class TestResolveDatapath:
    def test_datapath_wins(self, env, spl):
        spl._last_measure_tn = "last"
        assert spl._resolve_datapath("tn1", "/some/where") == "/some/where"

    def test_tn(self, env, spl):
        spl._last_measure_tn = "last"
        assert spl._resolve_datapath("tn1") == os.path.join(env["data"], "tn1")

    def test_last_tn(self, env, spl):
        spl._last_measure_tn = "last"
        assert spl._resolve_datapath() == os.path.join(env["data"], "last")

    def test_neither_raises(self, env, spl):
        with pytest.raises(ValueError):
            spl._resolve_datapath()


def _copy_measurement(src, dst):
    os.makedirs(dst)
    import shutil

    for f in os.listdir(src):
        if f.startswith("rawframe") or f in ("darkFrame.fits", "lambda_vector.fits"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))


@pytest.fixture
def measured(env, spl):
    """Flat measurement, processed, with matching templates; returns details."""
    spl.acquire_dark_frame(1.0)
    tn = spl.acquire(1.0, filter_mode="narrow", lambda_vector=LAMBDAS)
    folder = os.path.join(env["data"], tn)
    spl.process_psfs(tn=tn, n_psfs=1, initial_half_size=100)
    cube = osu.load_fits(os.path.join(folder, "psf0_cube.fits"), memmap=False)
    matrix, _ = spl._compute_matrix(LAMBDAS, cube)
    delta = _make_templates(env, matrix, LAMBDAS, M=5, k=3)
    pists = spl.analysis(tn=tn, n_psfs=1, initial_half_size=100)
    return dict(tn=tn, folder=folder, cube=cube, pists=pists, delta=delta)


class TestNestedDatapath:
    @pytest.mark.parametrize("parent", ["nested", "frame_run"])
    def test_process_and_analysis_nested(self, env, spl, measured, parent):
        nested = os.path.join(env["data"], parent, "cam0")
        _copy_measurement(measured["folder"], nested)

        spl.process_psfs(n_psfs=1, initial_half_size=100, datapath=nested)
        files = set(os.listdir(nested))
        for wl in LAMBDAS:
            assert f"rawpsf0_{wl}nm.fits" in files
            assert f"rot_psf0_{wl}nm.fits" in files
        assert "psf0_cube.fits" in files
        # nothing leaked outside of the nested folder
        assert os.listdir(os.path.join(env["data"], parent)) == ["cam0"]
        cube = osu.load_fits(os.path.join(nested, "psf0_cube.fits"), memmap=False)
        np.testing.assert_array_equal(
            np.asarray(cube), np.asarray(measured["cube"])
        )

        pists = spl.analysis(n_psfs=1, initial_half_size=100, datapath=nested)
        assert pists == measured["pists"]
        assert os.path.exists(os.path.join(nested, "psf0_fringes_result.fits"))

    def test_datapath_ignores_tn(self, env, spl, measured):
        nested = os.path.join(env["data"], "nested", "cam0")
        _copy_measurement(measured["folder"], nested)
        pists = spl.analysis(
            tn="does_not_exist", n_psfs=1, initial_half_size=100, datapath=nested
        )
        assert pists == measured["pists"]


class TestPlotComparison:
    def test_fresh_instance_with_datapath(self, env, measured, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
        fresh = phasing.SPL("none", "none", "tnfr")
        assert getattr(fresh, "_Qt", None) is None
        fresh.plot_comparison(datapath=measured["folder"], psf_n=0)
        assert shown == [1]
        plt.close("all")

    def test_templates_reloaded_when_lambda_differs(self, env, measured, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        fresh = phasing.SPL("none", "none", "tnfr")
        # cache built for a subset of the measurement wavelengths
        fresh._load_templates(LAMBDAS[:3])
        assert fresh._Qt.shape[1] == 3
        fresh.plot_comparison(datapath=measured["folder"], psf_n=0)
        assert fresh._Qt.shape[1] == LAMBDAS.size
        np.testing.assert_array_equal(fresh._Qt_lambda, LAMBDAS)
        plt.close("all")

    def test_set_tn_fringes_invalidates_template_cache(self, env, measured):
        spl = phasing.SPL("none", "none", "tnfr")
        spl._load_templates(LAMBDAS)
        assert spl._Qt is not None
        spl.set_tn_fringes("other")
        assert spl._Qt is None
        assert spl._Qt_lambda is None

    def test_no_tnfringes_raises_clear_error(self, env, measured):
        spl = phasing.SPL("none", "none", None)
        with pytest.raises(
            ValueError, match="No fringe templates tracking number set"
        ):
            spl.plot_comparison(datapath=measured["folder"], psf_n=0)
        with pytest.raises(
            ValueError, match="No fringe templates tracking number set"
        ):
            spl._load_templates(LAMBDAS)


class TestAnalyzeDatapath:
    def test_matches_spl_analysis(self, env, measured):
        nested = os.path.join(env["data"], "nested", "cam0")
        _copy_measurement(measured["folder"], nested)
        pists = phasing._analyze_datapath(
            nested, "tnfr", n_psfs=1, initial_half_size=100
        )
        assert pists == measured["pists"]
