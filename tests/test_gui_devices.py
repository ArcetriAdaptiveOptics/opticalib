"""
Tests for the CalpyGUI device registry and device panel
(opticalib.gui.widgets.device_registry, device_panel, connect_dialog).
"""

import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)
pytest.importorskip("qtawesome", exc_type=ImportError)

from opticalib.gui.widgets import device_panel as dp  # noqa: E402
from opticalib.gui.widgets import device_registry as reg  # noqa: E402

CONFIG = """\
SYSTEM:
  data_path: ''
DEVICES:
  # wavefront sensors
  WFS:
    Ingot:
      camera: CAMERAS:GigaVision
  INTERFEROMETERS:
    phasecam6110:  # lower-case name
      ip: 192.168.0.10
      port: 8011
    PhaseCamX:
      ip:
      port:
    Mystery:
      ip: 10.0.0.1
      port: 1
  DEFORMABLE.MIRRORS:
    Alpao820:
      serialNumber: BAX820
    MyPetal:
      ip0: 1.2.3.4
  CAMERAS:
    GigaVision:
      id: CAM1
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "configuration.yaml"
    path.write_text(CONFIG)
    return path


def _entries(path):
    return {e.name: e for e in reg.list_entries(str(path))}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestResolution:
    """How configuration entries are mapped to device classes."""

    @pytest.mark.parametrize(
        "section, name, conf, cls, command",
        [
            ("INTERFEROMETERS", "PhaseCam6110", {"ip": "a", "port": 1}, "PhaseCam", "interf = devices.PhaseCam('6110')"),
            ("INTERFEROMETERS", "phasecam6110", {"ip": "a", "port": 1}, "PhaseCam", "interf = devices.PhaseCam('6110')"),
            ("INTERFEROMETERS", "AccuFiz4020", {"ip": "a", "port": 1}, "AccuFiz", "interf = devices.AccuFiz('4020')"),
            ("INTERFEROMETERS", "4DProcesser1", {"ip": "a", "port": 1}, "Processer4D", "interf = devices.Processer4D('1')"),
            ("INTERFEROMETER", "PhaseCam6110", {"ip": "a", "port": 1}, "PhaseCam", "interf = devices.PhaseCam('6110')"),
            ("DEFORMABLE.MIRRORS", "Alpao820", {"serialNumber": "B"}, "AlpaoDm", "dm = devices.AlpaoDm(820)"),
            ("DEFORMABLE.MIRRORS", "PetalDM", {"ip0": "x"}, "PetalMirror", "dm = devices.PetalMirror()"),
            ("DEFORMABLE.MIRRORS", "Splatt", {"ip": "x", "port": 1}, "SplattDm", "dm = devices.SplattDm()"),
            ("DEFORMABLE.MIRRORS", "AdOpticaDP", {}, "DP", "dm = devices.DP()"),
            ("DEFORMABLE.MIRRORS", "M4AU", {}, "M4AU", "dm = devices.M4AU()"),
            ("DEFORMABLE.MIRRORS", "AdOpticaDM", {}, "AdOpticaDm", "dm = devices.AdOpticaDm()"),
            ("CAMERAS", "GigaVision", {"id": "c"}, "GigaVision", "cam = devices.GigaVision('GigaVision')"),
            ("CAMERAS", "Guide", {"ip": "c"}, "GigaVision", "cam = devices.GigaVision('Guide')"),
            ("WFS", "INGOT", {"camera": "CAMERAS:GigaVision"}, "Ingot", "wfs = devices.Ingot('GigaVision')"),
            ("WFS", "Ingot", {"camera": "CAMERAS:GigaVision"}, "Ingot", "wfs = devices.Ingot('GigaVision')"),
            ("WFS", "ingot", {"camera": "CAMERAS:GigaVision"}, "Ingot", "wfs = devices.Ingot('GigaVision')"),
        ],
    )
    def test_ready_entries(self, section, name, conf, cls, command):
        entry = reg.resolve_entry(section, name, conf)
        assert entry.device_class.name == cls
        assert entry.ready, entry.problems
        code = reg.build_command(entry)
        compile(code, "<connect>", "exec")
        assert code.splitlines() == ["import opticalib.devices as devices", command]

    def test_explicit_class_wins(self):
        entry = reg.resolve_entry("INTERFEROMETERS", "AccuFiz1", {"class": "phasecam", "ip": "a", "port": 1})
        assert entry.source == "explicit" and entry.device_class.name == "PhaseCam"

    def test_unknown_explicit_class(self):
        entry = reg.resolve_entry("CAMERAS", "Cam", {"class": "Nope"})
        assert entry.device_class is None and "Unknown class 'Nope'" in entry.problems[0]

    def test_section_default(self):
        entry = reg.resolve_entry("CAMERAS", "Guider", {"id": "x"})
        assert entry.source == "default" and entry.device_class.name == "GigaVision"

    def test_unknown_class(self):
        entry = reg.resolve_entry("INTERFEROMETERS", "Mystery", {"ip": "a"})
        assert entry.device_class is None and not entry.ready
        code = reg.build_command(entry)
        assert "devices." not in code.splitlines()[-1]
        compile(code, "<connect>", "exec")

    @pytest.mark.parametrize(
        "section, name, conf, problem",
        [
            ("INTERFEROMETERS", "PhaseCamX", {"ip": None, "port": ""}, "Missing: ip, port."),
            ("CAMERAS", "GigaVision", {}, "Missing: id or ip."),
            ("WFS", "Ingot", {}, "Cannot build the Ingot arguments"),
            ("DEFORMABLE.MIRRORS", "AlpaoXXX", {}, "name the entry Alpao<number of actuators>"),
            ("INTERFEROMETERS", "PhaseCam", {"ip": "a", "port": 1}, "name the entry PhaseCam<model>"),
            ("DEFORMABLE.MIRRORS", "MyPetal", {"class": "PetalMirror", "ip0": "x"}, "PetalMirror reads the entry 'PetalDM'"),
        ],
    )
    def test_problems(self, section, name, conf, problem):
        entry = reg.resolve_entry(section, name, conf)
        assert not entry.ready
        assert any(problem in p for p in entry.problems), entry.problems

    def test_optional_config_mismatch_is_a_note(self):
        entry = reg.resolve_entry("DEFORMABLE.MIRRORS", "DP", {})
        assert entry.ready and "AdOpticaDP" in entry.notes[0]

    def test_template_entries_are_all_listed(self):
        from opticalib.core.root import TEMPLATE_CONF_FILE

        entries = reg.list_entries(TEMPLATE_CONF_FILE)
        names = {e.name for e in entries}
        assert {"PhaseCamX", "AccuFizX", "4DProcesser1", "AlpaoXXX", "PetalDM", "GigaVision", "INGOT"} <= names
        for entry in entries:
            compile(reg.build_command(entry), entry.name, "exec")
        interferometers = [e for e in entries if e.section == "INTERFEROMETERS"]
        assert all(e.device_class is not None and "Missing: ip, port." in e.problems for e in interferometers)

    def test_empty_or_invalid_files(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("")
        assert reg.list_entries(str(path)) == []
        path.write_text("SYSTEM:\n  data_path: ''\nDEVICES:\n")
        assert reg.list_entries(str(path)) == []
        path.write_text("DEVICES:\n  CAMERAS:\n    Empty:\n")
        assert [e.name for e in reg.list_entries(str(path))] == ["Empty"]


class TestSetEntryClass:
    """Writing the ``class`` key into the configuration file."""

    def test_insert_and_update_preserve_the_file(self, config_file):
        reg.set_entry_class(str(config_file), "INTERFEROMETERS", "Mystery", "AccuFiz")
        text = config_file.read_text()
        assert "    Mystery:\n      class: AccuFiz\n      ip: 10.0.0.1\n" in text
        assert text.replace("      class: AccuFiz\n", "") == CONFIG
        reg.set_entry_class(str(config_file), "INTERFEROMETERS", "Mystery", "PhaseCam")
        assert config_file.read_text().count("class:") == 1
        assert _entries(config_file)["Mystery"].device_class.name == "PhaseCam"

    def test_empty_entries(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("DEVICES:\n  CAMERAS:\n    A:\n    B: {}\n  WFS:\n    C:\n")
        reg.set_entry_class(str(path), "CAMERAS", "A", "GigaVision")
        reg.set_entry_class(str(path), "CAMERAS", "B", "GigaVision")
        assert path.read_text() == (
            "DEVICES:\n  CAMERAS:\n    A:\n      class: GigaVision\n"
            "    B:\n      class: GigaVision\n  WFS:\n    C:\n"
        )

    def test_same_name_in_another_section(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("DEVICES:\n  CAMERAS:\n    X:\n      id: 1\n  WFS:\n    X:\n      camera: a\n")
        reg.set_entry_class(str(path), "WFS", "X", "Ingot")
        assert path.read_text().endswith("  WFS:\n    X:\n      class: Ingot\n      camera: a\n")

    @pytest.mark.parametrize(
        "text, section, name",
        [
            ("DEVICES:\n  CAMERAS:\n    A: {id: 1}\n", "CAMERAS", "A"),
            ("DEVICES:\n  CAMERAS:\n    A:\n", "CAMERAS", "Missing"),
            ("DEVICES:\n  CAMERAS:\n    A:\n", "WFS", "A"),
        ],
    )
    def test_refused_edits_leave_the_file(self, tmp_path, text, section, name):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            reg.set_entry_class(str(path), section, name, "GigaVision")
        assert path.read_text() == text


# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------


class TestQuickActions:
    """Tests for the device quick actions."""

    def _info(self, kind, simulated=False):
        return dp.DeviceInfo(
            key="k", title="t", kind=kind, var_name="x", class_name="C",
            module_prefix="m", build_code=lambda o: "", simulated=simulated,
        )

    @pytest.mark.parametrize("kind", ["dm", "interferometer", "wfs", "camera", "device"])
    def test_actions_compile(self, kind):
        for simulated in (False, True):
            for label, code, confirm in dp.quick_actions(self._info(kind, simulated)):
                compile(code, label, "exec")

    def test_zero_command_only_for_simulated_dms(self):
        labels = [a[0] for a in dp.quick_actions(self._info("dm"))]
        assert "Reset to zero" not in labels
        actions = {a[0]: a for a in dp.quick_actions(self._info("dm", simulated=True))}
        assert actions["Reset to zero"][2] is True  # needs confirmation


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@pytest.fixture
def panel(qapp, config_file):
    from opticalib.gui.kernel import Task

    tasks = []

    def runner(code, title, on_done, on_error):
        task = Task(code, title, on_done=on_done, on_error=on_error)
        tasks.append(task)
        return task

    widget = dp.DevicePanel(str(config_file), runner=runner)
    widget.tasks = tasks
    return widget


def _ws(name, type_, module, kind):
    return {"name": name, "type": type_, "module": module, "kind": kind, "summary": ""}


class TestDevicePanel:
    """Tests for the card states of the device panel."""

    def test_every_entry_has_a_card(self, panel):
        keys = {c.info.key for c in panel.cards}
        assert {
            "cfg:WFS:Ingot",
            "cfg:INTERFEROMETERS:phasecam6110",
            "cfg:INTERFEROMETERS:PhaseCamX",
            "cfg:INTERFEROMETERS:Mystery",
            "cfg:DEFORMABLE.MIRRORS:Alpao820",
            "cfg:DEFORMABLE.MIRRORS:MyPetal",
            "cfg:CAMERAS:GigaVision",
            "sim:alpao", "sim:dp", "sim:petal", "sim:interf",
        } <= keys

    def test_ready_and_setup_cards(self, panel):
        ready = panel.card("cfg:WFS:Ingot")
        assert ready.display_status == "disconnected"
        assert ready._button.text() == "Connect"
        assert ready.connect_code().endswith("wfs = devices.Ingot('GigaVision')")
        setup = panel.card("cfg:INTERFEROMETERS:PhaseCamX")
        assert setup.display_status == "setup"
        assert setup._button.text() == "Set up…"
        assert "Missing: ip, port." in setup._problem.full_text()

    def test_setup_button_opens_the_dialog(self, panel, monkeypatch):
        opened = []
        monkeypatch.setattr(panel, "open_connect_dialog", lambda card: opened.append(card))
        # Reconnect the signal to the patched method.
        card = panel.card("cfg:INTERFEROMETERS:Mystery")
        card.setup_requested.disconnect()
        card.setup_requested.connect(panel.open_connect_dialog)
        card._button.click()
        assert opened == [card] and panel.tasks == []

    def test_connect_success_and_disconnect(self, panel):
        card = panel.card("sim:petal")
        card._button.click()
        assert card.status == "queued"
        task = panel.tasks[-1]
        assert "dm = PetalMirror()" in task.code
        task._start("msg-1")
        assert card.status == "connecting"
        task._finish("done")
        assert card.status == "connected"
        panel.update_workspace([])
        assert card.status == "disconnected"

    def test_connect_failure(self, panel):
        card = panel.card("sim:petal")
        card._button.click()
        task = panel.tasks[-1]
        task._start("msg-1")
        task._finish("error", {"ename": "OSError", "evalue": "no data"})
        assert card.status == "error"
        assert "OSError: no data" in card.message

    def test_alpao_option(self, panel):
        card = panel.card("sim:alpao")
        card._combo.setCurrentIndex(card._combo.findData(277))
        assert card.connect_code().endswith("AlpaoDm(n_acts=277)")

    def test_interferometer_needs_a_dm(self, panel):
        interf = panel.card("sim:interf")
        panel.update_workspace([])
        assert interf.status == "unavailable"
        assert not interf._button.isEnabled()
        panel.update_workspace([_ws("dm", "PetalMirror", "opticalib.simulator.fake_dms", "dm")])
        assert interf.status == "disconnected"

    def test_manual_creation_detected(self, panel):
        panel.update_workspace([_ws("dm", "DP", "opticalib.simulator.fake_dms", "dm")])
        assert panel.card("sim:dp").status == "connected"
        panel.update_workspace([_ws("dm", "DP", "opticalib.devices.deformable_mirrors", "dm")])
        assert panel.card("sim:dp").status == "disconnected"

    def test_reconnect_moves_ownership(self, panel):
        alpao, dp_card = panel.card("sim:alpao"), panel.card("sim:dp")
        for card in (alpao, dp_card):
            card._button.click()
            task = panel.tasks[-1]
            task._start(f"m-{card.info.key}")
            task._finish("done")
        assert dp_card.status == "connected"
        assert alpao.status == "disconnected"

    def test_reload_keeps_state(self, panel):
        # The stored workspace does not contain `cam` (e.g. its refresh is
        # queued behind a long command): reloading must not disconnect.
        card = panel.card("cfg:CAMERAS:GigaVision")
        card.set_status("connected")
        panel.reload()
        assert panel.card("cfg:CAMERAS:GigaVision") is card
        assert card.status == "connected"

    def test_edit_request(self, panel):
        requested = []
        panel.edit_config_requested.connect(lambda s, n: requested.append((s, n)))
        panel.card("cfg:INTERFEROMETERS:Mystery").edit_requested.emit(panel.card("cfg:INTERFEROMETERS:Mystery"))
        assert requested == [("INTERFEROMETERS", "Mystery")]


class TestConnectDialog:
    """Tests for the connect dialog and how the panel applies it."""

    def _dialog(self, panel, name):
        card = panel.card(f"cfg:INTERFEROMETERS:{name}")
        return card, panel._make_dialog(card.info.entry, card.info.var_name)

    def test_choosing_a_class(self, panel):
        card, dialog = self._dialog(panel, "Mystery")
        assert dialog.class_name() is None
        assert not dialog._btn_connect.isEnabled()
        dialog._class_combo.setCurrentIndex(dialog._class_combo.findData("PhaseCam"))
        assert dialog.class_name() == "PhaseCam"
        # The name has no model suffix: the command is a template to complete.
        assert "Cannot build the PhaseCam arguments" in dialog._problems.text()
        assert dialog.save_class()

    def test_code_edits_survive_class_changes_until_reset(self, panel):
        card, dialog = self._dialog(panel, "phasecam6110")
        assert dialog.code().endswith("interf = devices.PhaseCam('6110')")
        assert not dialog._save_class.isChecked()  # the name already maps to PhaseCam
        dialog._code.setPlainText("interf = devices.PhaseCam('6110', ip='1.1.1.1', port=1)")
        dialog._var_edit.setText("interf2")
        assert "ip='1.1.1.1'" in dialog.code()
        dialog._reset_code()
        assert dialog.code().endswith("interf2 = devices.PhaseCam('6110')")

    @pytest.mark.parametrize("var, ok", [("interf", True), ("my_dev2", True), ("2dev", False), ("class", False), ("", False)])
    def test_variable_validation(self, panel, var, ok):
        card, dialog = self._dialog(panel, "phasecam6110")
        dialog._var_edit.setText(var)
        assert dialog._btn_connect.isEnabled() is ok

    def test_apply_saves_class_and_connects(self, panel, config_file):
        changed = []
        panel.config_changed.connect(lambda: changed.append(1))
        card, dialog = self._dialog(panel, "Mystery")
        dialog._class_combo.setCurrentIndex(dialog._class_combo.findData("AccuFiz"))
        dialog._var_edit.setText("fizeau")
        dialog._code.setPlainText("fizeau = devices.AccuFiz('1', ip='10.0.0.1', port=1)")
        panel.apply_connect_dialog(card, dialog)
        assert "    Mystery:\n      class: AccuFiz\n" in config_file.read_text()
        assert changed == [1]
        card = panel.card("cfg:INTERFEROMETERS:Mystery")
        assert card.info.class_name == "AccuFiz" and card.info.var_name == "fizeau"
        task = panel.tasks[-1]
        assert task.code == "fizeau = devices.AccuFiz('1', ip='10.0.0.1', port=1)"
        task._start("m")
        task._finish("done")
        assert card.status == "connected"
        # Connection detection follows the chosen variable and class.
        panel.update_workspace([_ws("fizeau", "AccuFiz", "opticalib.devices.interferometer", "interferometer")])
        assert card.status == "connected"

    def test_apply_without_saving(self, panel, config_file):
        card, dialog = self._dialog(panel, "Mystery")
        dialog._class_combo.setCurrentIndex(dialog._class_combo.findData("PhaseCam"))
        dialog._save_class.setChecked(False)
        panel.apply_connect_dialog(card, dialog)
        assert config_file.read_text() == CONFIG
        assert panel.card("cfg:INTERFEROMETERS:Mystery").info.class_name == "PhaseCam"


def test_config_editor_goto_entry(qapp, config_file):
    from opticalib.gui.widgets.config_editor import ConfigEditorDialog

    dialog = ConfigEditorDialog(str(config_file))
    assert dialog.goto_entry("INTERFEROMETERS", "Mystery")
    cursor = dialog._text_edit.textCursor()
    assert cursor.selectedText() == "Mystery"
    assert cursor.blockNumber() == CONFIG.splitlines().index("    Mystery:")
    assert not dialog.goto_entry("INTERFEROMETERS", "Nope")


class TestSetEntryClassRobustness:
    def test_trailing_comments(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "DEVICES:\n  CAMERAS:\n    A: {}  # empty for now\n"
            "    B:  # main camera\n      class: Old  # chosen by hand\n      id: 1\n"
        )
        reg.set_entry_class(str(path), "CAMERAS", "A", "GigaVision")
        reg.set_entry_class(str(path), "CAMERAS", "B", "GigaVision")
        assert path.read_text() == (
            "DEVICES:\n  CAMERAS:\n    A: # empty for now\n      class: GigaVision\n"
            "    B:  # main camera\n      class: GigaVision  # chosen by hand\n      id: 1\n"
        )

    def test_invalid_file_is_left_untouched(self, tmp_path):
        text = "DEVICES:\n  CAMERAS:\n    A:\n      id: [1, 2\n"
        path = tmp_path / "c.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            reg.set_entry_class(str(path), "CAMERAS", "A", "GigaVision")
        assert path.read_text() == text

    def test_no_temporary_files_left(self, config_file):
        reg.set_entry_class(str(config_file), "INTERFEROMETERS", "Mystery", "PhaseCam")
        assert sorted(p.name for p in config_file.parent.iterdir()) == ["configuration.yaml"]
