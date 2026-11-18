"""Tests for opticalib.ground._stitching_algorithm (platform guard)."""

import sys

import numpy as np
import pytest

from opticalib.ground import _stitching_algorithm as st


def _cube(n=2, size=8):
    yy, xx = np.mgrid[:size, :size]
    circ = (yy - size / 2 + 0.5) ** 2 + (xx - size / 2 + 0.5) ** 2 > (size / 2 - 1) ** 2
    imgs = [
        np.ma.masked_array(np.random.default_rng(i).normal(size=(size, size)), mask=circ)
        for i in range(n)
    ]
    return np.ma.masked_array(imgs), circ


class TestWindowsGuard:
    def test_win32_cpu_raises_before_any_work(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(st.xp, "on_gpu", False, raising=False)

        def fail(*a, **k):
            raise AssertionError("work started before the guard")

        monkeypatch.setattr(st, "ZernikeFitter", fail)
        # even invalid input must hit the guard first
        with pytest.raises(NotImplementedError, match="Windows"):
            st.map_stitching(None, None, [1, 2, 3])

    def test_win32_gpu_does_not_trigger_guard(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(st.xp, "on_gpu", True, raising=False)

        class Reached(Exception):
            pass

        def reached(*a, **k):
            raise Reached

        monkeypatch.setattr(st, "ZernikeFitter", reached)
        cube, mask = _cube()
        with pytest.raises(Reached):
            st.map_stitching(cube, mask, [1, 2, 3])

    @pytest.mark.parametrize("plat", ["linux", "darwin"])
    def test_posix_cpu_does_not_trigger_guard(self, monkeypatch, plat):
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setattr(st.xp, "on_gpu", False, raising=False)

        class Reached(Exception):
            pass

        def reached(*a, **k):
            raise Reached

        monkeypatch.setattr(st, "ZernikeFitter", reached)
        cube, mask = _cube()
        with pytest.raises(Reached):  # not NotImplementedError
            st.map_stitching(cube, mask, [1, 2, 3])
