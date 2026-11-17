"""
Tests for the normalization paths of the IFF processing pipeline.

Covers:

- ``opticalib.dmutils._modes_normalization.resolve``
- the ``normalization`` argument of ``iff_processing.iff_redux`` (numerical
  result of the reduction)
- the forwarding of ``normalization`` from ``iff_processing.process`` down to
  ``iff_redux``
"""

import os
import numpy as np
import numpy.ma as ma
import pytest
from astropy.io import fits
from unittest.mock import patch

from opticalib.dmutils import iff_processing as ifp
from opticalib.analyzer import push_pull_reduction_algorithm


# --------------------------------------------------------------------------- #
# iff_redux() -- numerical behaviour of the normalization
# --------------------------------------------------------------------------- #
SHAPE = (6, 6)
TEMPLATE = [1, -1, 1]  # T = 3
AMPS = np.array([0.1, 0.5])  # one amplitude per mode
MODES = np.array([0, 1])


def _make_images(seed: int, n: int, masked_pixel: tuple | None = None):
    """Generate ``n`` reproducible masked images."""
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        mask = np.zeros(SHAPE, dtype=bool)
        if masked_pixel is not None:
            mask[masked_pixel] = True
        imgs.append(ma.masked_array(rng.normal(size=SHAPE), mask=mask))
    return imgs


class _Env:
    """Fake on-disk layout: file name -> image, and captured saved outputs."""

    def __init__(self, n_rep: int, n_modes: int, T: int, masked_pixel=None):
        self.images: dict[str, ma.MaskedArray] = {}
        self.saved: dict[int, ma.MaskedArray] = {}
        self.headers: dict[int, fits.Header] = {}
        self.fileMat = np.empty((n_rep, n_modes, T), dtype=object)
        seed = 0
        for r in range(n_rep):
            for m in range(n_modes):
                imgs = _make_images(seed, T, masked_pixel if (r, m) == (0, 0) else None)
                seed += 1
                for t in range(T):
                    name = f"img_r{r}_m{m}_t{t}.fits"
                    self.fileMat[r, m, t] = name
                    self.images[name] = imgs[t]

    # patched osutils functions ------------------------------------------- #
    def read_phasemap(self, path, **kw):
        return self.images[path]

    def read_fits_header(self, path, **kw):
        return fits.Header()

    def save_fits(self, path, data, overwrite=False, header=None, **kw):
        mode_id = int(os.path.basename(path).split("_")[1].split(".")[0])
        self.saved[mode_id] = data
        self.headers[mode_id] = header.copy()


def _run_redux(env: _Env, template, amps, modes, n_rep, normalization, **kw):
    """Run ``iff_redux`` against the fake environment."""
    with patch.object(ifp._osu, "read_phasemap", env.read_phasemap), patch.object(
        ifp._osu, "read_fits_header", env.read_fits_header
    ), patch.object(ifp._osu, "save_fits", env.save_fits):
        ifp.iff_redux(
            tn="19700101_000000",
            fileMat=env.fileMat,
            ampVect=amps,
            modeList=modes,
            template=template,
            n_repetitions=n_rep,
            normalization=normalization,
            **kw,
        )


def _raw_reduction(env: _Env, template, n_rep: int, mode_idx: int):
    """Un-normalized reference (normalization factor of exactly 1),
    averaged over the repetitions. Independent of the code under test."""
    T = len(template)
    acc, mask = None, None
    for r in range(n_rep):
        imgs = [env.images[env.fileMat[r, mode_idx, t]] for t in range(T)]
        red = push_pull_reduction_algorithm(imgs, template, normalization=1)
        acc = red.data.astype(np.float32) if acc is None else acc + red.data
        mask = red.mask.copy() if mask is None else mask | red.mask
    return ma.masked_array(acc / n_rep, mask=mask)


class TestIffReduxNormalization:
    """Effect of the ``normalization`` argument on ``iff_redux`` outputs."""

    def _check(self, env, template, n_rep, factors):
        """Assert saved == raw / factors[mode_idx] (data and mask)."""
        for m, mode_id in enumerate(MODES):
            raw = _raw_reduction(env, template, n_rep, m)
            out = env.saved[int(mode_id)]
            np.testing.assert_allclose(
                out.filled(0), (raw / factors[m]).filled(0), rtol=1e-4, atol=1e-6
            )
            np.testing.assert_array_equal(ma.getmaskarray(out), ma.getmaskarray(raw))

    def test_true_uses_default_per_mode_formula(self):
        """True -> each mode divided by 2 * max(T-1,1) * A_mode."""
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 1, True)
        T = len(TEMPLATE)
        self._check(env, TEMPLATE, 1, [2 * max(T - 1, 1) * a for a in AMPS])

    def test_true_is_default_value(self):
        """Omitting ``normalization`` is equivalent to ``True``."""
        env_a, env_b = _Env(1, 2, 3), _Env(1, 2, 3)
        _run_redux(env_a, TEMPLATE, AMPS, MODES, 1, True)
        with patch.object(ifp._osu, "read_phasemap", env_b.read_phasemap), patch.object(
            ifp._osu, "read_fits_header", env_b.read_fits_header
        ), patch.object(ifp._osu, "save_fits", env_b.save_fits):
            ifp.iff_redux(
                tn="x",
                fileMat=env_b.fileMat,
                ampVect=AMPS,
                modeList=MODES,
                template=TEMPLATE,
                n_repetitions=1,
            )
        for m in MODES:
            np.testing.assert_allclose(
                env_a.saved[int(m)].filled(0), env_b.saved[int(m)].filled(0)
            )

    def test_false_applies_no_normalization(self):
        """False -> raw push-pull sum, finite and unscaled."""
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 1, False)
        for out in env.saved.values():
            assert np.all(np.isfinite(out.filled(0)))
        self._check(env, TEMPLATE, 1, [1.0, 1.0])

    @pytest.mark.parametrize("factor", [2, 0.5, 10.0, 1])
    def test_number_is_flat_factor_for_all_modes(self, factor):
        """A number divides every mode by the same value, ignoring A."""
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 1, factor)
        self._check(env, TEMPLATE, 1, [factor, factor])

    def test_true_and_false_and_number_differ(self):
        """Sanity: the three paths give three different scalings of the
        same raw data."""
        outs = {}
        for key, norm in (("true", True), ("false", False), ("num", 4.0)):
            env = _Env(1, 2, len(TEMPLATE))
            _run_redux(env, TEMPLATE, AMPS, MODES, 1, norm)
            outs[key] = env.saved[0].filled(0)
        assert not np.allclose(outs["true"], outs["false"])
        assert not np.allclose(outs["true"], outs["num"])
        assert not np.allclose(outs["false"], outs["num"])
        np.testing.assert_allclose(outs["false"] / 4.0, outs["num"], rtol=1e-4)

    def test_scalar_amplitude_is_broadcast_with_default_norm(self):
        """A scalar amplitude is expanded to all modes before normalizing."""
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, 0.2, MODES, 1, True)
        T = len(TEMPLATE)
        self._check(env, TEMPLATE, 1, [2 * max(T - 1, 1) * 0.2] * 2)

    def test_repetitions_are_averaged_after_normalization(self):
        """With N repetitions the result is the mean of the normalized runs."""
        env = _Env(3, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 3, True)
        T = len(TEMPLATE)
        self._check(env, TEMPLATE, 3, [2 * max(T - 1, 1) * a for a in AMPS])

    def test_repetitions_with_flat_number(self):
        env = _Env(2, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 2, 3.0)
        self._check(env, TEMPLATE, 2, [3.0, 3.0])

    @pytest.mark.parametrize("template", [[1], [1, -1], [1, -1, 1, -1, 1]])
    def test_default_formula_across_template_lengths(self, template):
        """The default factor follows the template length (T=1 clamps to 1)."""
        env = _Env(1, 2, len(template))
        _run_redux(env, template, AMPS, MODES, 1, True)
        T = len(template)
        self._check(env, template, 1, [2 * max(T - 1, 1) * a for a in AMPS])

    def test_mask_is_preserved_for_every_normalization(self):
        """Masked pixels stay masked whatever the normalization is."""
        for norm in (True, False, 2.0):
            env = _Env(1, 2, len(TEMPLATE), masked_pixel=(2, 3))
            _run_redux(env, TEMPLATE, AMPS, MODES, 1, norm)
            assert ma.getmaskarray(env.saved[0])[2, 3]
            assert not ma.getmaskarray(env.saved[1])[2, 3]

    @pytest.mark.parametrize("io_workers,prefetch", [(1, 0), (2, 1), (4, 3)])
    def test_normalization_independent_of_prefetching(self, io_workers, prefetch):
        """Per-mode normalization must follow the right mode when I/O is
        prefetched in parallel."""
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(
            env, TEMPLATE, AMPS, MODES, 1, True, io_workers=io_workers, prefetch=prefetch
        )
        T = len(TEMPLATE)
        self._check(env, TEMPLATE, 1, [2 * max(T - 1, 1) * a for a in AMPS])

    def test_amplitude_header_matches_normalizing_amplitude(self):
        env = _Env(1, 2, len(TEMPLATE))
        _run_redux(env, TEMPLATE, AMPS, MODES, 1, True)
        for m, mode_id in enumerate(MODES):
            assert env.headers[int(mode_id)]["AMP"] == pytest.approx(AMPS[m])

    @pytest.mark.parametrize("bad", ["total", "shlike", None])
    def test_invalid_normalization_raises(self, bad):
        env = _Env(1, 2, len(TEMPLATE))
        with pytest.raises(ValueError):
            _run_redux(env, TEMPLATE, AMPS, MODES, 1, bad)
        assert env.saved == {}

    def test_validation_errors_unchanged_by_normalization(self):
        """Shape-consistency checks still fire before any normalization."""
        env = _Env(1, 2, len(TEMPLATE))
        with pytest.raises(ValueError, match="Template length"):
            _run_redux(env, [1, -1], AMPS, MODES, 1, True)
        with pytest.raises(ValueError, match="repetitions"):
            _run_redux(env, TEMPLATE, AMPS, MODES, 2, True)


# --------------------------------------------------------------------------- #
# process() -- forwarding of the normalization option
# --------------------------------------------------------------------------- #
@pytest.fixture
def patched_process(tmp_path):
    """Patch every collaborator of ``process`` except the forwarding itself."""
    info = {
        "FILES": {
            "modes_list": np.array([0, 1]),
            "template": np.array(TEMPLATE),
            "n_repetitions": 1,
            "registration_modes": np.array([]),
        },
        "IFFUNC": {"amplitude": AMPS, "modes_list": MODES},
    }
    with patch.object(ifp, "_get_acq_par", return_value=info), patch.object(
        ifp, "_get_acq_info", return_value={}
    ), patch.object(ifp, "get_trigger_frame", return_value=0), patch.object(
        ifp, "_check_information_consistency"
    ), patch.object(
        ifp, "get_reg_file_matrix", return_value=np.array([])
    ), patch.object(
        ifp, "get_iff_file_matrix", return_value=np.empty((1, 2, 3), dtype=object)
    ), patch.object(
        ifp, "_modes_matrix_reorganization", side_effect=lambda m, i: m
    ), patch.object(
        ifp, "_IMFOLD", str(tmp_path)
    ), patch.object(
        ifp, "iff_redux"
    ) as redux, patch.object(
        ifp, "save_cube"
    ):
        yield redux


class TestProcessNormalizationForwarding:
    """``process`` must hand its ``normalization`` option to ``iff_redux``."""

    def test_default_is_true(self, patched_process):
        ifp.process("20240101_000000")
        assert patched_process.call_args.kwargs["normalization"] is True

    @pytest.mark.parametrize("norm", [True, False, 3, 0.25])
    def test_value_is_forwarded_untouched(self, patched_process, norm):
        ifp.process("20240101_000000", normalization=norm)
        forwarded = patched_process.call_args.kwargs["normalization"]
        assert forwarded == norm
        assert type(forwarded) is type(norm)

    def test_amplitudes_and_template_reach_redux(self, patched_process):
        """The inputs the default normalization depends on (T and A) are
        passed to ``iff_redux`` together with the option."""
        ifp.process("20240101_000000", normalization=True)
        kw = patched_process.call_args.kwargs
        np.testing.assert_array_equal(kw["ampVect"], AMPS)
        np.testing.assert_array_equal(kw["template"], TEMPLATE)

    def test_normalization_is_keyword_only(self, patched_process):
        """``normalization`` sits after ``*`` in the signature."""
        with pytest.raises(TypeError):
            ifp.process("20240101_000000", False, False, 1, 5.0)

    def test_list_of_tn_forwards_normalization_for_each(self, patched_process):
        ifp.process(["20240101_000000", "20240101_000001"], normalization=2.0)
        assert patched_process.call_count == 2
        for call in patched_process.call_args_list:
            assert call.kwargs["normalization"] == 2.0
