"""Tests for the ``BaseDeformableMirror`` contract in ``devices/_API/base_devices``."""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

import opticalib.devices._API.base_devices as bd
from opticalib.core.exceptions import DeviceAttributeError, DeviceNotFoundError
from opticalib.devices._API.base_devices import BaseDeformableMirror


def _fill(obj, **over):
    attrs = dict(
        _name="Fake",
        n_acts=4,
        is_segmented=False,
        n_segments=1,
        n_acts_per_segment=4,
        mirror_modes=None,
        act_coord=None,
        ff=None,
        _cmd_history=None,
        _slave_ids=[],
        _border_ids=[],
    )
    attrs.update(over)
    for k, v in attrs.items():
        if isinstance(getattr(type(obj), k, None), property):
            continue
        setattr(obj, k, v)


class _Dm(BaseDeformableMirror):
    """Minimal concrete DM; ``skip`` names attributes left undefined."""

    def __init__(self, skip=(), logger=None, **over):
        _fill(self, **over)
        for a in skip:
            delattr(self, a)
        if logger is not None:
            self._logger = logger

    def set_shape(self, cmd, differential=False, *, slaving=False):
        pass

    def get_shape(self):
        return np.zeros(self.n_acts)

    def upload_cmd_history(self, tcmdhist, *, slaving=False):
        pass

    def run_cmd_history(self, wfs=None, save=None, **kw):
        pass


class TestRequiredAttributes:
    def test_complete_subclass_constructs(self):
        dm = _Dm()
        assert dm.n_acts == 4

    def test_missing_single_attribute(self):
        with pytest.raises(DeviceAttributeError, match="ff"):
            _Dm(skip=("ff",))

    def test_missing_attributes_all_named(self):
        with pytest.raises(DeviceAttributeError) as ei:
            _Dm(skip=("ff", "act_coord", "_slave_ids"))
        msg = str(ei.value)
        for name in ("ff", "act_coord", "_slave_ids"):
            assert name in msg
        assert "_Dm" in msg

    def test_missing_all_attributes(self):
        class Empty(_Dm):
            def __init__(self):
                pass

        with pytest.raises(DeviceAttributeError) as ei:
            Empty()
        for name in BaseDeformableMirror._REQUIRED_ATTRIBUTES:
            assert name in str(ei.value)

    def test_is_attribute_error_subclass(self):
        with pytest.raises(AttributeError):
            _Dm(skip=("n_acts",))

    def test_validation_only_after_outermost_init(self):
        class Child(_Dm):
            def __init__(self):
                super().__init__(skip=("ff",))  # would be incomplete here
                self.ff = None  # completed afterwards

        assert Child().ff is None

    def test_child_still_incomplete_raises(self):
        class Child(_Dm):
            def __init__(self):
                super().__init__(skip=("ff",))

        with pytest.raises(DeviceAttributeError):
            Child()

    def test_property_satisfies_requirement(self):
        class Prop(_Dm):
            def __init__(self):
                super().__init__()

            @property
            def n_acts(self):
                return 4

        assert Prop().n_actuators == 4

    def test_none_values_are_accepted(self):
        _Dm(mirror_modes=None, act_coord=None, ff=None, _cmd_history=None)

    def test_abstract_class_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseDeformableMirror()


class TestSegmentationWarning:
    def test_mismatch_warns_via_logger(self):
        logger = MagicMock()
        _Dm(logger=logger, n_segments=2, n_acts_per_segment=3, n_acts=7)
        logger.warning.assert_called_once()
        assert "n_acts" in logger.warning.call_args[0][0]

    def test_mismatch_without_logger_warns_runtime(self):
        with pytest.warns(RuntimeWarning, match="n_segments"):
            dm = _Dm(n_segments=2, n_acts_per_segment=3, n_acts=7)
        assert dm.n_acts == 7  # no raise

    def test_consistent_does_not_warn(self):
        logger = MagicMock()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _Dm(logger=logger, n_segments=2, n_acts_per_segment=2, n_acts=4)
            _Dm()
        logger.warning.assert_not_called()

    def test_none_logger_falls_back_to_warnings(self):
        with pytest.warns(RuntimeWarning):
            _Dm(n_segments=3, n_acts_per_segment=1, n_acts=4)
            # explicit _logger=None path
        dm = _Dm()
        dm._logger = None
        with pytest.warns(RuntimeWarning):
            dm._warn("x")


class TestInitSlavingIds:
    def _patch(self, monkeypatch, cfg=None, exc=None):
        calls = []

        def fake(section, name):
            calls.append((section, name))
            if exc is not None:
                raise exc
            return cfg

        monkeypatch.setattr(bd._rc, "get_device_config", fake)
        return calls

    def test_present(self, monkeypatch):
        calls = self._patch(monkeypatch, {"slave_ids": [1, 2], "border_ids": (3,)})
        dm = _Dm(_name="Mine")
        dm._init_slaving_ids()
        assert dm.slave_ids == [1, 2]
        assert dm.border_ids == [3]
        assert calls == [("DEFORMABLE.MIRRORS", "Mine")]

    def test_explicit_config_name(self, monkeypatch):
        calls = self._patch(monkeypatch, {})
        _Dm()._init_slaving_ids("Other")
        assert calls == [("DEFORMABLE.MIRRORS", "Other")]

    @pytest.mark.parametrize("exc", [KeyError("x"), DeviceNotFoundError("x")])
    def test_missing_block(self, monkeypatch, exc):
        self._patch(monkeypatch, exc=exc)
        dm = _Dm(_slave_ids=[9], _border_ids=[9])
        dm._init_slaving_ids()
        assert dm.slave_ids == [] and dm.border_ids == []

    @pytest.mark.parametrize("cfg", [None, {}])
    def test_empty_config(self, monkeypatch, cfg):
        self._patch(monkeypatch, cfg)
        dm = _Dm()
        dm._init_slaving_ids()
        assert dm.slave_ids == [] and dm.border_ids == []

    def test_none_values(self, monkeypatch):
        self._patch(monkeypatch, {"slave_ids": None, "border_ids": None})
        dm = _Dm()
        dm._init_slaving_ids()
        assert dm.slave_ids == [] and dm.border_ids == []

    def test_partial_keys(self, monkeypatch):
        self._patch(monkeypatch, {"slave_ids": [4]})
        dm = _Dm()
        dm._init_slaving_ids()
        assert dm.slave_ids == [4] and dm.border_ids == []

    def test_ids_are_copied(self, monkeypatch):
        src = [1, 2]
        self._patch(monkeypatch, {"slave_ids": src})
        dm = _Dm()
        dm._init_slaving_ids()
        dm._slave_ids.append(99)
        assert src == [1, 2]

    def test_other_exceptions_propagate(self, monkeypatch):
        self._patch(monkeypatch, exc=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            _Dm()._init_slaving_ids()


class TestGetSlavingMethod:
    @pytest.mark.parametrize("slave", ["zero-force", "minimum-rms", "anything", ""])
    def test_string_returned_verbatim(self, slave):
        assert _Dm(_slave_ids=[1])._get_slaving_method(slave) == slave

    @pytest.mark.parametrize("slave", [False, None, 0])
    def test_falsy_returns_none(self, slave):
        assert _Dm(_slave_ids=[1])._get_slaving_method(slave) is None

    def test_true_no_slave_ids_warns(self):
        logger = MagicMock()
        dm = _Dm(logger=logger)
        assert dm._get_slaving_method(True) is None
        logger.warning.assert_called_once()

    def test_true_no_slave_ids_but_border_warns(self):
        logger = MagicMock()
        dm = _Dm(logger=logger, _border_ids=[1])
        assert dm._get_slaving_method(True) is None
        logger.warning.assert_called_once()

    def test_true_slave_only(self):
        assert _Dm(_slave_ids=[1])._get_slaving_method(True) == "zero-force"

    def test_true_slave_and_border(self):
        dm = _Dm(_slave_ids=[1], _border_ids=[2])
        assert dm._get_slaving_method(True) == "minimum-rms"

    def test_truthy_non_bool(self):
        assert _Dm(_slave_ids=[1])._get_slaving_method(1) == "zero-force"


class TestApplySlaving:
    @pytest.fixture
    def rec(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(
            bd, "compute_slave_cmd",
            lambda dm, cmd, method=None: calls.setdefault("vec", (cmd, method)) and "VEC",
        )
        monkeypatch.setattr(
            bd, "compute_slaved_command_matrix",
            lambda dm, cmdmat=None, method=None: calls.setdefault("mat", (cmdmat, method)) and "MAT",
        )
        return calls

    def test_no_slaving_returns_input(self, rec):
        cmd = np.zeros(4)
        assert _Dm(_slave_ids=[1])._apply_slaving(cmd, False) is cmd
        assert rec == {}

    def test_no_slaving_matrix_returns_input(self, rec):
        m = np.zeros((4, 2))
        assert _Dm(_slave_ids=[1])._apply_slaving(m, None) is m
        assert rec == {}

    def test_1d_true_resolved_method(self, rec):
        dm = _Dm(_slave_ids=[1], _border_ids=[2])
        assert dm._apply_slaving(np.zeros(4), True) == "VEC"
        assert rec["vec"][1] == "minimum-rms"
        assert "mat" not in rec

    def test_2d_true_resolved_method(self, rec):
        dm = _Dm(_slave_ids=[1])
        assert dm._apply_slaving(np.zeros((4, 3)), True) == "MAT"
        assert rec["mat"][1] == "zero-force"
        assert "vec" not in rec

    def test_2d_string_method(self, rec):
        _Dm(_slave_ids=[1])._apply_slaving(np.zeros((4, 3)), "custom")
        assert rec["mat"][1] == "custom"

    def test_1d_string_method(self, rec):
        _Dm()._apply_slaving(np.zeros(4), "custom")
        assert rec["vec"][1] == "custom"

    def test_true_without_slave_ids_passthrough(self, rec):
        cmd = np.zeros((4, 2))
        assert _Dm(logger=MagicMock())._apply_slaving(cmd, True) is cmd
        assert rec == {}


class TestProperties:
    def test_properties(self):
        h = np.ones((4, 2))
        dm = _Dm(_slave_ids=[1], _border_ids=[2, 3], _cmd_history=h, n_acts=4)
        assert dm.slave_ids == [1]
        assert dm.border_ids == [2, 3]
        assert dm.cmd_history is h
        assert dm.n_actuators == 4

    def test_cmd_history_none(self):
        assert _Dm().cmd_history is None

    def test_cmd_history_read_only(self):
        with pytest.raises(AttributeError):
            _Dm().cmd_history = np.zeros(1)

    @pytest.mark.parametrize("prop", ["slave_ids", "border_ids", "n_actuators"])
    def test_other_properties_read_only(self, prop):
        with pytest.raises(AttributeError):
            setattr(_Dm(), prop, 1)

    def test_n_actuators_tracks_n_acts(self):
        dm = _Dm()
        dm.n_acts = 10
        assert dm.n_actuators == 10
