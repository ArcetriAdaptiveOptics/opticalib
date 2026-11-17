"""
Tests of the keyword arguments of the simulated DM ``run_cmd_history``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from opticalib.simulator._API.base_simulated_dm import BaseSimulatedDM


class _Recorder(SimpleNamespace):
    """Minimal stand-in for a simulated DM: records the set_shape calls."""

    def get_shape(self):
        return np.zeros(3)

    def set_shape(self, cmd, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def dm(tmp_path, monkeypatch):
    import opticalib.simulator._API.base_simulated_dm as module

    monkeypatch.setattr(module.fp, "OPD_IMAGES_ROOT_FOLDER", str(tmp_path), raising=False)
    monkeypatch.setattr(module._rc, "get_iff_config", lambda key=None: {})
    return _Recorder(
        _cmd_history=np.ones((3, 2)),
        _logger=SimpleNamespace(info=lambda *a: None, error=lambda *a: None),
        calls=[],
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"slaving": True}, True),
        ({"slaving": "zero-force"}, "zero-force"),
        ({"slave": True}, True),  # former keyword, still accepted
        ({}, False),
    ],
)
def test_run_cmd_history_forwards_slaving(dm, kwargs, expected):
    BaseSimulatedDM.run_cmd_history(dm, None, save="20260101_000000", **kwargs)
    # One call per mode with the requested slaving, then the final reset.
    assert [c.get("slaving") for c in dm.calls[:2]] == [expected, expected]
