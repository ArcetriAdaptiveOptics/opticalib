"""
End-to-end smoke test of CalpyGUI (opticalib.gui.app).

The whole application -- window, out-of-process IPython kernel and calpy
bootstrap -- runs in a subprocess with an offscreen Qt platform, isolated Qt
settings and a temporary experiment, so nothing leaks into the test process
or into the user's configuration.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)
pytest.importorskip("ipykernel", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)
pytest.importorskip("qtawesome", exc_type=ImportError)

_SCRIPT = textwrap.dedent(
    r"""
    import json, os, sys, time
    import opticalib.gui  # selects the default Qt binding before qtpy
    from qtpy.QtCore import QEventLoop, QSettings
    from qtpy.QtWidgets import QApplication, QMessageBox

    settings_dir, config_path = sys.argv[1], sys.argv[2]
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, settings_dir)
    app = QApplication(sys.argv)
    from opticalib.gui.app import CalpyGUI

    result = {}

    def pump(until, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline and not until():
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        return bool(until())

    def query(expressions):
        box = {}
        window.bridge.query(expressions, box.update)
        pump(lambda: box, 30)
        return box

    window = CalpyGUI(config_path=config_path)
    window.show()
    bridge = window.bridge
    result["ready"] = pump(lambda: bridge.is_ready, 180)
    pump(lambda: False, 0.5)
    from qtpy.QtWidgets import QTabBar

    def dock_state():
        titles = {k: d.titleBarWidget().title() for k, d in window._docks.items()}
        bars = sorted(
            [b.tabText(i) for i in range(b.count())]
            for b in window.findChildren(QTabBar) if b.isVisible()
        )
        return titles, bars

    result["docks"] = dock_state()
    procedures = window._docks["procedures"]
    procedures.titleBarWidget()._float.click()  # detach into its own window
    pump(lambda: False, 0.5)
    result["detached"] = [procedures.isFloating(), procedures.titleBarWidget().title()]
    window._reset_layout()
    pump(lambda: False, 0.5)
    result["docks_after_reset"] = dock_state()
    names = ("opt", "folders", "zern", "osu", "az", "sim", "ifp", "ifm", "oplt", "_gui")
    ns = query({n: f"{n!r} in globals()" for n in names})
    result["names"] = sorted(n for n in names if ns.get(n) is True)
    result["aoconf"] = query({"a": "__import__('os').environ['AOCONF']"}).get("a")

    task = window._run(
        "import numpy as np\n"
        "img = np.ma.masked_array(np.ones((32, 48)), mask=np.eye(32, 48, dtype=bool))\n"
        "figure(); plot([0, 1, 2]); title('ramp')\n"
        "_gui.view(img, 'img')",
        "Plot and view",
    )
    pump(lambda: task.is_final and window.plot_viewer.get_figure_count() >= 2, 60)
    result["task"] = task.state
    result["plots"] = [(i.kind, i.title) for i in window.plot_viewer.items]
    pump(lambda: any(i["name"] == "img" for i in window.workspace_view.items), 30)
    result["workspace"] = sorted(i["name"] for i in window.workspace_view.items)
    result["data_base"] = window.data_browser._base

    task = window._run(
        "import time\nfor i in range(1, 11):\n    print(f'{i}/10', end='\\r', flush=True); time.sleep(0.05)",
        "Progress",
    )
    seen = []
    task.changed.connect(lambda: seen.append(task.progress))
    pump(lambda: task.is_final, 30)
    result["progress_task"] = task.state
    result["progress"] = [p for p in seen if p is not None][-1:]

    task = window._run("import time; time.sleep(60)", "Sleep")
    pump(lambda: task.state == "running", 30)
    time.sleep(0.5)
    bridge.interrupt()
    result["interrupt_ok"] = pump(lambda: task.is_final, 20)
    result["interrupt"] = (task.state, (task.error or {}).get("ename"))

    task = window._run("1/0", "Fail")
    pump(lambda: task.is_final, 20)
    result["error"] = (task.state, (task.error or {}).get("ename"))
    result["error_card"] = any(c.job is task for c in window.activity.cards)

    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._restart_kernel()
    result["restarted"] = pump(lambda: bridge.is_ready, 180)
    after = query({"img": "'img' in globals()", "gui": "'_gui' in globals()"})
    result["after_restart"] = [after.get("img"), after.get("gui")]
    task = window._run("x_after_restart = 1", "After restart")
    pump(lambda: task.is_final, 30)
    result["after_restart_task"] = task.state

    tmp_dir = window._tmp_dir
    window.close()
    pump(lambda: False, 0.5)
    result["kernel_stopped"] = not bridge.kernel_manager.has_kernel
    result["tmp_removed"] = not os.path.exists(tmp_dir)
    settings = QSettings("ArcetriAdaptiveOptics", "CalpyGUI")
    result["layout_saved"] = settings.contains("window/state")
    result["settings_file"] = settings.fileName()
    print("RESULT " + json.dumps(result))
    """
)


@pytest.mark.integration
def test_gui_end_to_end(tmp_path):
    """Run a full CalpyGUI session and check every subsystem."""
    # Locate the template without importing opticalib in the test process
    # (the autouse fixture of conftest.py sets AOCONF to an empty string).
    import importlib.util

    package_dir = os.path.dirname(importlib.util.find_spec("opticalib").origin)
    TEMPLATE_CONF_FILE = os.path.join(package_dir, "core", "_configurations", "configuration.yaml")

    data_path = tmp_path / "data"
    config_path = tmp_path / "experiment" / "configuration.yaml"
    config_path.parent.mkdir()
    with open(TEMPLATE_CONF_FILE) as f:
        template = f.read()
    config_path.write_text(template.replace("data_path: ''", f"data_path: '{data_path}'", 1))
    script = tmp_path / "smoke.py"
    script.write_text(_SCRIPT)
    settings_dir = tmp_path / "settings"

    env = dict(
        os.environ,
        AOCONF=str(config_path),
        QT_QPA_PLATFORM="offscreen",
        CUDA_VISIBLE_DEVICES="",
        XDG_CONFIG_HOME=str(tmp_path / "xdg"),
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(settings_dir), str(config_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=600,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, f"smoke script failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    r = json.loads(lines[-1][len("RESULT "):])

    assert r["ready"]
    # Tabbed panels are named by their tab: their (draggable) title bar shows
    # no text; the console, alone, shows its name. Qt's stale tab bars are hidden.
    expected_docks = [
        {"devices": "", "procedures": "", "workspace": "", "data": "", "console": "Console"},
        [["Devices", "Procedures"], ["Workspace", "Data"]],
    ]
    assert r["docks"] == expected_docks
    assert r["detached"] == [True, "Procedures"]
    assert r["docks_after_reset"] == expected_docks
    assert r["names"] == sorted(["opt", "folders", "zern", "osu", "az", "sim", "ifp", "ifm", "oplt", "_gui"])
    assert r["aoconf"] == str(config_path)
    assert r["task"] == "done"
    assert ["figure", "ramp"] in r["plots"] and ["data", "img"] in r["plots"]
    assert "img" in r["workspace"] and "np" not in r["workspace"]
    assert r["data_base"] == str(data_path)
    assert r["progress_task"] == "done" and r["progress"] == [pytest.approx(1.0)]
    assert r["interrupt_ok"] and r["interrupt"] == ["error", "KeyboardInterrupt"]
    assert r["error"] == ["error", "ZeroDivisionError"] and r["error_card"]
    assert r["restarted"] and r["after_restart"] == [False, True]
    assert r["after_restart_task"] == "done"
    assert r["kernel_stopped"] and r["tmp_removed"] and r["layout_saved"]
    assert r["settings_file"].startswith(str(settings_dir))
