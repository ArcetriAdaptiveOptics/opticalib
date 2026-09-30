"""
Tests for the CalpyGUI procedure windows (opticalib.gui.procedures).

A fake context records the code the windows would run in the console, so
the generated calls can be checked without a kernel.
"""

import pytest

pytest.importorskip("qtconsole")
pytest.importorskip("qtawesome")


class FakeContext:
    """Stands in for ProcedureContext: records runs and answers queries."""

    def __new__(cls, *args, **kwargs):
        from opticalib.gui.procedures.base import ProcedureContext

        class _Context(ProcedureContext):
            def __init__(self):
                super().__init__(run=self._run, query=self._query, interrupt=self._interrupt)
                self.runs = []
                self.queries = []
                self.interrupted = 0
                self.answers = {}

            def _run(self, code, title, on_done=None, on_error=None):
                from opticalib.gui.kernel import Task

                task = Task(code, title, on_done=on_done, on_error=on_error)
                self.runs.append(task)
                return task

            def _query(self, expressions, callback):
                self.queries.append(expressions)
                callback({k: self.answers.get(k, RuntimeError("no answer")) for k in expressions})

            def _interrupt(self):
                self.interrupted += 1

        return _Context()


@pytest.fixture
def context(qapp):
    ctx = FakeContext()
    ctx.set_workspace(
        [
            {"name": "dm", "type": "AlpaoDm", "module": "m", "kind": "dm", "summary": ""},
            {"name": "interf", "type": "Fake4DInterf", "module": "m", "kind": "interferometer", "summary": ""},
            {"name": "cam", "type": "GigaVision", "module": "m", "kind": "camera", "summary": ""},
        ]
    )
    return ctx


def _finish(task, state="done", error=None):
    task._start("m")
    task._finish(state, error)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestParams:
    def test_numbers(self, qapp):
        from opticalib.gui.procedures.params import FloatParam, IntParam, ParamError

        amp = FloatParam("amp", "Amplitude", 7e-8, minimum=0)
        amp.widget()
        assert amp.code() == "7e-08"
        amp.widget().setText("1.5e-7")
        assert amp.code() == "1.5e-07"
        amp.widget().setText("-1")
        with pytest.raises(ParamError, match="Amplitude"):
            amp.code()
        amp.widget().setText("abc")
        with pytest.raises(ParamError):
            amp.code()
        n = IntParam("n", "Frames", 10, minimum=1, maximum=100)
        n.widget()
        assert n.code() == "10"

    def test_optional_values_render_none(self, qapp):
        from opticalib.gui.procedures.params import ExprParam, FloatParam, TextParam, TnParam

        for param in (FloatParam("a", "A", None, optional=True), TextParam("b", "B", optional=True),
                      ExprParam("c", "C", optional=True), TnParam("d", "D", optional=True)):
            param.widget()
            assert param.code() == "None"

    def test_expressions_are_checked(self, qapp):
        from opticalib.gui.procedures.params import ExprParam, ParamError

        modes = ExprParam("modes", "Modes", "np.arange(10)")
        modes.widget()
        assert modes.code() == "np.arange(10)"
        modes.widget().setText("np.arange(10")
        with pytest.raises(ParamError, match="invalid expression"):
            modes.code()

    def test_variables_and_text(self, qapp):
        from opticalib.gui.procedures.params import ParamError, TextParam, VarParam

        var = VarParam("out", "Result", "tn_iff")
        var.widget()
        assert var.code() == "tn_iff"
        var.widget().setText("2bad")
        with pytest.raises(ParamError):
            var.code()
        text = TextParam("mode", "Mode", "it's")
        text.widget()
        assert text.code() == '"it\'s"'

    def test_device_follows_the_workspace(self, context):
        from opticalib.gui.procedures.params import DeviceParam

        dm = DeviceParam("dm", "Mirror", kinds=("dm",), default="dm")
        dm.widget(context)
        assert dm.code() == "dm"
        combo = dm.widget()
        assert [combo.itemText(i) for i in range(combo.count())] == ["dm"]
        context.set_workspace(context.workspace + [
            {"name": "dm2", "type": "DP", "module": "m", "kind": "dm", "summary": ""}])
        assert combo.count() == 2 and dm.code() == "dm"

    def test_device_required(self, qapp):
        from opticalib.gui.procedures.params import DeviceParam, ParamError

        cam = DeviceParam("cam", "Camera", kinds=("camera",))
        cam.widget()
        with pytest.raises(ParamError, match="choose a device"):
            cam.code()

    def test_tracking_numbers_are_listed(self, context, tmp_path):
        from opticalib.gui.procedures.params import TnParam

        for tn in ("20260101_000000", "20260202_000000"):
            (tmp_path / tn).mkdir()
        context.set_folders({"paths": {"IFFUNCTIONS_ROOT_FOLDER": str(tmp_path)}})
        tn = TnParam("tn", "Tracking number", folder_attr="IFFUNCTIONS_ROOT_FOLDER")
        tn.widget(context)
        combo = tn._combo
        assert [combo.itemText(i) for i in range(combo.count())] == ["20260202_000000", "20260101_000000"]
        tn.set_value("20260101_000000")
        assert tn.code() == "'20260101_000000'"


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------


def _demo_window(context):
    from opticalib.gui.procedures.base import ProcedureWindow, Step
    from opticalib.gui.procedures.params import IntParam, TnParam, VarParam

    class Demo(ProcedureWindow):
        TITLE = "Demo"

        def steps(self):
            return [
                Step(
                    "acquire", "Acquire", "Acquire frames.",
                    [IntParam("n", "Frames", 3, minimum=1), VarParam("out", "Result", "tn_demo")],
                    lambda v: f"{v['out']} = acquire({v['n']})",
                    outputs={"demo_tn": "tn_demo"},
                    confirm="This moves the mirror.",
                ),
                Step(
                    "analyse", "Analyse", "Analyse them.",
                    [TnParam("tn", "Tracking number", state_key="demo_tn")],
                    lambda v: f"analyse({v['tn']})",
                ),
            ]

    return Demo(context)


class TestProcedureWindow:
    def test_preview_and_confirmation(self, context, monkeypatch):
        from opticalib.gui.procedures import base

        window = _demo_window(context)
        assert window._code.toPlainText() == "tn_demo = acquire(3)"
        monkeypatch.setattr(base.QMessageBox, "question", lambda *a: base.QMessageBox.StandardButton.No)
        window.run_current()
        assert context.runs == []
        monkeypatch.setattr(base.QMessageBox, "question", lambda *a: base.QMessageBox.StandardButton.Yes)
        window.run_current()
        assert [t.code for t in context.runs] == ["tn_demo = acquire(3)"]

    def test_results_fill_the_next_step(self, context):
        window = _demo_window(context)
        context.answers["demo_tn"] = repr("20260930_120000")
        task = window.run_step(window.step("acquire"), confirmed=True)
        assert not window._btn_run.isEnabled()  # one step at a time
        _finish(task)
        assert context.queries[-1] == {"demo_tn": "repr(tn_demo)"}
        assert window.state == {"demo_tn": "20260930_120000"}
        assert window._status["acquire"] == "done"
        window.select_step("analyse")
        assert window._code.toPlainText() == "analyse('20260930_120000')"

    def test_failed_step(self, context):
        window = _demo_window(context)
        task = window.run_step(window.step("acquire"), confirmed=True)
        _finish(task, "error", {"ename": "RuntimeError", "evalue": "no light"})
        assert window._status["acquire"] == "error"
        assert "no light" in window._log.item(0).text()
        assert window._btn_run.isEnabled()

    def test_invalid_parameters_disable_run(self, context):
        window = _demo_window(context)
        window.select_step("analyse")  # no tracking number yet
        assert not window._btn_run.isEnabled()
        assert "tracking number is required" in window._error.text()

    def test_stop_interrupts(self, context):
        window = _demo_window(context)
        window.run_step(window.step("acquire"), confirmed=True)
        window._stop()
        assert context.interrupted == 1

    def test_without_context_code_is_preview_only(self, qapp):
        window = _demo_window(None)
        assert window._code.toPlainText() and not window._btn_run.isEnabled()


# ---------------------------------------------------------------------------
# Procedure windows
# ---------------------------------------------------------------------------


def _windows():
    from opticalib.gui.plugins import PLUGIN_WINDOWS

    return list(PLUGIN_WINDOWS.items())


def _fill_required(window):
    """Give every empty required parameter a plausible value."""
    from opticalib.gui.procedures.params import DeviceParam, ParamError, TnListParam, TnParam

    for step in window._steps:
        window._form_for(step)
        for param in step.params:
            try:
                param.code()
            except ParamError:
                if isinstance(param, TnParam):
                    param.set_value("20260101_000000")
                elif isinstance(param, TnListParam):
                    param.set_value(["20260101_000000", "20260102_000000"])
                elif isinstance(param, DeviceParam):
                    param.set_value("motors" if "other" in param.kinds else "dev")


#: Steps that move hardware and must ask for confirmation.
HARDWARE_STEPS = {
    "Deformable Mirror Calibration": {"acquire", "flatten", "loop"},
    "Timeseries": set(),
    "Stitching": {"scan"},
    "Alignment": {"calibrate", "apply"},
    "Segments Phasing": {"dark", "preview", "acquire"},
}


@pytest.mark.parametrize("name, cls", _windows(), ids=[n for n, _ in _windows()])
def test_every_step_generates_valid_code(context, name, cls):
    window = cls(context)
    _fill_required(window)
    for step in window._steps:
        code = step.code()
        compile(code, f"{name}:{step.key}", "exec")
        assert bool(step.confirm) == (step.key in HARDWARE_STEPS[name]), step.key
    window.close()


class TestDMCalibration:
    @pytest.fixture
    def window(self, context):
        from opticalib.gui.procedures.dm_calibration import DeformableMirrorCalibrationWindow

        return DeformableMirrorCalibrationWindow(context)

    def test_defaults_come_from_the_configuration(self, window):
        code = window.step("acquire").code()
        assert code == "tn_iff = ifm.iff_data_acquisition(dm, interf, shuffle=False, n_repetitions=1)"

    def test_explicit_acquisition(self, window):
        step = window.step("acquire")
        window._form_for(step)
        step.param("modes").set_value("np.arange(10)")
        step.param("amplitude").set_value("0.05")
        step.param("base").set_value("hadamard")
        assert "modeslist=np.arange(10), amplitude=0.05, modalbase='hadamard'" in step.code()
        step.param("base_file").set_value("my_base.fits")
        assert "modalbase='my_base.fits'" in step.code()

    def test_tracking_numbers_chain(self, window, context):
        context.answers["iff_tn"] = repr("20260930_100000")
        task = window.run_step(window.step("acquire"), confirmed=True)
        _finish(task)
        process = window.step("process")
        window._form_for(process)
        assert process.param("tn").value() == "20260930_100000"
        _finish(window.run_step(process))
        svd = window.step("svd")
        window._form_for(svd)
        assert svd.param("tn").value() == "20260930_100000"  # same tn in INTMatrices

    def test_closed_loop_is_an_explicit_loop(self, window):
        code = window.step("loop").code()
        assert "for _i in range(3):" in code and "closed_loop_flattening" not in code
        assert "print(f'{_i + 1}/3')" in code  # progress in the activity panel

    def test_threshold_is_a_python_int(self, window):
        step = window.step("flatten")
        window._form_for(step)
        step.param("modes2discard").set_value("2")
        assert "modes2discard=2" in step.code()

    def test_config_check(self, window):
        window._on_checks({"check0": False})
        assert window._warnings.isVisibleTo(window) and "SysConfig" in window._warnings.text()
        window._on_checks({"check0": True})
        assert not window._warnings.isVisibleTo(window)


def test_timeseries_delay_is_at_least_one_second(context):
    from opticalib.gui.procedures.params import ParamError
    from opticalib.gui.procedures.timeseries import TimeseriesWindow

    window = TimeseriesWindow(context)
    step = window.step("acquire")
    window._form_for(step)
    step.param("delay").set_value(0.5)
    with pytest.raises(ParamError):
        step.code()


def test_phasing_sweep_takes_the_dark_first(context):
    from opticalib.gui.procedures.phasing import SegmentsPhasingWindow

    window = SegmentsPhasingWindow(context)
    lines = window.step("acquire").code().splitlines()
    assert lines[0].startswith("spl.acquire_dark_frame(") and ".acquire(" in lines[1]


def test_plugin_panel_opens_every_window(qapp):
    from opticalib.gui.plugins import PLUGIN_WINDOWS, PLUGINS

    assert [name for name, _, _ in PLUGINS] == list(PLUGIN_WINDOWS)


# ---------------------------------------------------------------------------
# Sections, configuration defaults, post-processing
# ---------------------------------------------------------------------------


def test_dm_calibration_sections(context):
    from qtpy.QtCore import Qt

    from opticalib.gui.procedures.dm_calibration import DeformableMirrorCalibrationWindow

    window = DeformableMirrorCalibrationWindow(context)
    rows = [window._step_list.item(i) for i in range(window._step_list.count())]
    labels = [(r.text(), r.data(Qt.ItemDataRole.UserRole)) for r in rows]
    assert labels == [
        ("CALIBRATION", None),
        ("Acquire influence functions", "acquire"),
        ("Process", "process"),
        ("Singular values", "svd"),
        ("Apply flat command", "flatten"),
        ("Closed-loop flattening", "loop"),
        ("POST-PROCESSING", None),
        ("Filter Zernike modes", "filter"),
        ("Stack cubes", "stack"),
        ("ROI processing", "roi"),
    ]
    assert not rows[0].flags() & Qt.ItemFlag.ItemIsSelectable  # headers are not steps
    assert window.current_step().key == "acquire"
    window.select_step("roi")
    assert window.current_step().key == "roi"


def test_configuration_defaults_are_shown(context):
    from opticalib.gui.procedures.dm_calibration import DeformableMirrorCalibrationWindow

    context.answers.update({
        "acquire__modes": "np.arange(0, 88)",
        "acquire__amplitude": "0.05",
        "acquire__template": "[1, -1, 1]",
        "acquire__base": "zonal",
    })
    window = DeformableMirrorCalibrationWindow(context)
    queried = [q for q in context.queries if "acquire__modes" in q][-1]
    assert "get_iff_config('IFFUNC')['modes_list']" in queried["acquire__modes"]
    step = window.step("acquire")
    assert step.param("modes").widget().placeholderText() == "np.arange(0, 88)"
    assert step.param("amplitude").widget().placeholderText() == "0.05"
    base = step.param("base").widget()
    assert base.itemText(0) == "From configuration (zonal)"
    # Left empty, the values are not passed: the library reads the same configuration.
    assert "amplitude" not in step.code()
    # The hints are read again when the configuration is reloaded.
    count = len(context.queries)
    context.answers["acquire__amplitude"] = "0.1"
    context.set_folders({"paths": {}})
    assert len(context.queries) > count
    assert step.param("amplitude").widget().placeholderText() == "0.1"


def test_hints_reach_forms_built_later(context):
    from opticalib.gui.procedures.phasing import SegmentsPhasingWindow

    context.answers.update({"analyse__n_psfs": "6", "setup__camera": "CAMERAS:GigaVision"})
    window = SegmentsPhasingWindow(context)
    window.select_step("analyse")  # its form is created now
    assert window.step("analyse").param("n_psfs").widget().placeholderText() == "6"
    camera = window.step("setup").param("camera")
    assert camera.widget().lineEdit().placeholderText() == "CAMERAS:GigaVision"


def test_post_processing_code(context):
    from opticalib.gui.procedures.dm_calibration import DeformableMirrorCalibrationWindow

    window = DeformableMirrorCalibrationWindow(context)
    stack = window.step("stack")
    window._form_for(stack)
    stack.param("tns").set_value("20260101_000000, 20260102_000000")
    assert stack.code() == "tn_stacked = ifp.stack_cubes(['20260101_000000', '20260102_000000'])"
    roi = window.step("roi")
    window._form_for(roi)
    roi.param("tn").set_value("20260101_000000")
    roi.param("tt_detrend").set_value(True)
    assert roi.code() == (
        "tn_roi = ifp.cube_roi_processing('20260101_000000', 0, tt_detrend=True, "
        "mean_subtraction=False, median_subtraction=False, roinull=False)"
    )


def test_tn_list_param(context, tmp_path):
    from opticalib.gui.procedures.params import ParamError, TnListParam

    for tn in ("20260101_000000", "20260202_000000"):
        (tmp_path / tn).mkdir()
    context.set_folders({"paths": {"INTMAT_ROOT_FOLDER": str(tmp_path)}})
    tns = TnListParam("tns", "Tracking numbers", folder_attr="INTMAT_ROOT_FOLDER", minimum=2)
    tns.widget(context)
    assert tns.available() == ["20260202_000000", "20260101_000000"]
    tns.append("20260101_000000")
    with pytest.raises(ParamError, match="at least 2"):
        tns.code()
    tns.append("20260202_000000")
    assert tns.code() == "['20260101_000000', '20260202_000000']"


def test_device_boxes_are_wide(context):
    from qtpy.QtWidgets import QSizePolicy

    from opticalib.gui.procedures.params import DeviceParam

    combo = DeviceParam("dm", "Mirror", kinds=("dm",)).widget(context)
    assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert combo.minimumContentsLength() >= 24
