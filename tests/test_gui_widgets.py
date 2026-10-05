"""
Tests for the CalpyGUI panels: theme, background jobs, plot viewer, data
browser and workspace view.
"""

import os
import re

import numpy as np
import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)
pytest.importorskip("qtawesome", exc_type=ImportError)
pytest.importorskip("pyqtgraph", exc_type=ImportError)


def _png(color: str = "red") -> bytes:
    from qtpy.QtCore import QBuffer, QByteArray, QIODevice
    from qtpy.QtGui import QColor, QPixmap

    pixmap = QPixmap(40, 30)
    pixmap.fill(QColor(color))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buf, "PNG")
    return bytes(data)


class TestTheme:
    def test_tokens_and_style_sheets(self, qapp):
        from opticalib.gui import theme as th

        assert set(th.LIGHT_TOKENS) == set(th.DARK_TOKENS)
        for tokens in (th.LIGHT_TOKENS, th.DARK_TOKENS):
            # Every {token} placeholder was substituted.
            assert re.search(r"\{[a-z_]+\}", th.build_style_sheet(tokens)) is None
            assert tokens["surface"] in th.console_style_sheet(tokens)

    def test_modes(self, qapp):
        from opticalib.gui import theme as th

        manager = th.theme()
        mode = manager.mode
        try:
            manager.set_mode("dark")
            assert manager.is_dark and manager.tokens is th.DARK_TOKENS
            manager.set_mode("light")
            assert not manager.is_dark
            with pytest.raises(ValueError):
                manager.set_mode("sepia")
        finally:
            manager.set_mode(mode)


class TestLocalJob:
    def test_success(self, qapp, qt_wait):
        from opticalib.gui.activity import LocalJob

        job = LocalJob(lambda: 6 * 7, "answer").start()
        assert qt_wait(lambda: job.is_final)
        assert job.state == "done" and job.result == 42

    def test_failure(self, qapp, qt_wait):
        from opticalib.gui.activity import LocalJob

        failed = []
        job = LocalJob(lambda: 1 / 0, "boom", on_error=failed.append).start()
        assert qt_wait(lambda: job.is_final)
        assert job.error["ename"] == "ZeroDivisionError" and failed == [job]


class TestPlotViewer:
    def test_prepare_array(self):
        from opticalib.gui.widgets.plot_viewer import prepare_array

        masked = np.ma.masked_array(np.ones((2, 3)), mask=[[1, 0, 0], [0, 0, 0]])
        out = prepare_array(masked)
        assert out.dtype == np.float32 and np.isnan(out[0, 0])
        assert prepare_array(np.zeros((4, 5, 7))).shape == (7, 4, 5)
        with pytest.raises(ValueError):
            prepare_array(np.zeros((1, 1, 1, 1)))

    def test_figures_update_in_place(self, qapp):
        from opticalib.gui.widgets.plot_viewer import PlotViewer

        viewer = PlotViewer()
        viewer.add_figure({"uid": 1, "num": 1, "title": "one", "png": _png()})
        viewer.add_figure({"uid": 2, "num": 2, "title": "two", "png": _png("blue")})
        viewer.add_figure({"uid": 1, "num": 1, "title": "one v2", "png": _png("green")})
        assert [i.title for i in viewer.items] == ["one v2", "two"]
        viewer.show_previous()
        assert viewer.current_key() == "fig:1"
        viewer.remove_current()
        assert [i.key for i in viewer.items] == ["fig:2"]
        viewer.clear()
        assert viewer.get_figure_count() == 0

    @pytest.mark.parametrize("shape", [(50,), (20, 30), (20, 30, 4)])
    def test_data_items(self, qapp, shape):
        from opticalib.gui.widgets.plot_viewer import PlotViewer

        viewer = PlotViewer()
        item = viewer.add_data(np.random.rand(*shape), "data")
        assert viewer.current_key() == item.key
        window = viewer.pop_out_current()
        assert window is not None
        window.close()

    def test_closed_pop_out_is_released(self, qapp, qt_wait):
        # Regression: closing a pop-out, then destroying the viewer, used to
        # crash PySide6 when the deferred deletion ran a stale lambda.
        import gc

        from opticalib.gui.widgets.plot_viewer import PlotViewer

        viewer = PlotViewer()
        viewer.add_data(np.random.rand(20, 30), "data")
        window = viewer.pop_out_current()
        window.close()
        assert viewer._windows == []
        del viewer, window
        gc.collect()
        qt_wait(lambda: False, timeout=0.5)

    def test_save_array_round_trip(self, tmp_path):
        from opticalib.ground.osutils import load_fits
        from opticalib.gui.widgets.plot_viewer import save_array

        masked = np.ma.masked_array(np.arange(6.0).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 0]])
        save_array(str(tmp_path / "a.fits"), masked)
        assert np.ma.getmaskarray(load_fits(str(tmp_path / "a.fits")))[0, 1]
        save_array(str(tmp_path / "a.npy"), masked)
        np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), masked.data)

    def test_load_npz_view_removes_file_on_failure(self, tmp_path):
        from opticalib.gui.widgets.plot_viewer import load_npz_view

        path = tmp_path / "broken.npz"
        path.write_bytes(b"not a zip file")
        with pytest.raises(Exception):
            load_npz_view(str(path))
        assert not path.exists()

    def test_load_npz_view_removes_file(self, tmp_path):
        from opticalib.gui.widgets.plot_viewer import load_npz_view

        path = tmp_path / "v.npz"
        np.savez(path, data=np.ones((2, 2)), mask=np.eye(2, dtype=bool))
        arr = load_npz_view(str(path))
        assert isinstance(arr, np.ma.MaskedArray) and arr.mask[0, 0]
        assert not path.exists()


@pytest.fixture
def data_tree(tmp_path):
    """A small opticalib-like data tree."""
    from opticalib.ground.osutils import save_fits

    images = tmp_path / "OPDImages"
    for tn in ("20260101_120000", "20260102_120000"):
        (images / tn).mkdir(parents=True)
    save_fits(str(images / "20260102_120000" / "img.fits"), np.ones((4, 4)))
    np.save(images / "20260102_120000" / "cube.npy", np.zeros((3, 3, 2)))
    (tmp_path / "IFFunctions").mkdir()
    return {
        "base": str(tmp_path),
        "config": "",
        "categories": [["OPD images", str(images)], ["IF functions", str(tmp_path / "IFFunctions")]],
    }


class TestDataBrowser:
    def test_tree_and_filter(self, qapp, data_tree):
        from opticalib.gui.widgets.data_browser import DataBrowser

        browser = DataBrowser()
        browser.set_folders(data_tree)
        images = browser._tree.topLevelItem(0)
        assert images.text(0) == "OPD images  (2)"
        latest = images.child(0)
        assert latest.text(0) == "20260102_120000"
        latest.setExpanded(True)
        files = sorted(latest.child(i).text(0) for i in range(latest.childCount()))
        assert files == ["cube.npy", "img.fits"]
        assert browser._tracking_number(latest.child(0)) == "20260102_120000"
        browser._filter.setText("0101")
        assert images.child(0).isHidden() and not images.child(1).isHidden()

    def test_new_tracking_number_appears(self, qapp, qt_wait, data_tree):
        from opticalib.gui.widgets.data_browser import DataBrowser

        browser = DataBrowser()
        browser.set_folders(data_tree)
        os.mkdir(os.path.join(data_tree["categories"][1][1], "20260103_000000"))
        assert qt_wait(lambda: browser._tree.topLevelItem(1).text(0) == "IF functions  (1)", timeout=10)

    def test_missing_folders(self, qapp, tmp_path):
        from opticalib.gui.widgets.data_browser import DataBrowser

        browser = DataBrowser()
        browser.set_folders({"base": str(tmp_path), "categories": [["Gone", str(tmp_path / "gone")]]})
        assert browser._tree.topLevelItem(0).text(0) == "Gone  (0)"

    def test_loaders(self, data_tree):
        from opticalib.gui.widgets.data_browser import load_array_file, load_code

        folder = os.path.join(data_tree["categories"][0][1], "20260102_120000")
        assert load_array_file(os.path.join(folder, "img.fits")).shape == (4, 4)
        assert load_array_file(os.path.join(folder, "cube.npy")).shape == (3, 3, 2)
        assert load_code("/a/b.fits") == "data = osu.load_fits('/a/b.fits')"
        assert load_code("/a/it's.npy") == 'data = np.load("/a/it\'s.npy")'


class TestWorkspaceView:
    ITEMS = [
        {"name": "dm", "type": "AlpaoDm", "module": "m", "kind": "dm", "summary": "<dm>"},
        {"name": "img", "type": "MaskedArray", "module": "m", "kind": "array", "summary": "(2, 2)"},
        {"name": "n", "type": "int", "module": "m", "kind": "other", "summary": "3"},
    ]

    def test_groups_filter_and_actions(self, qapp):
        from opticalib.gui.widgets.workspace import WorkspaceView, default_action

        view = WorkspaceView()
        view.set_items(self.ITEMS)
        tree = view._tree
        groups = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert groups == ["Devices (1)", "Arrays (1)", "Other (1)"]
        requested = []
        view.run_requested.connect(lambda code, title: requested.append(code))
        view._on_double_click(tree.topLevelItem(1).child(0), 0)
        assert requested == ["_gui.view(img, 'img')"] == [default_action(self.ITEMS[1])]
        view._filter.setText("zz")
        assert view._empty.isVisibleTo(view)


def test_elided_label(qapp):
    from opticalib.gui.widgets.common import ElidedLabel

    label = ElidedLabel("/a/very/long/path/that/does/not/fit/in/the/label.yaml")
    label.resize(120, 20)
    label.show()
    assert "…" in label.text()
    assert label.full_text().endswith("label.yaml")
    label.close()


def test_plugin_window_is_released_on_close(qapp, qt_wait):
    import gc

    from qtpy.QtWidgets import QMainWindow

    from opticalib.gui.plugins import PLUGIN_WINDOWS

    owner = QMainWindow()
    closed = []
    for window_cls in PLUGIN_WINDOWS.values():
        window = window_cls(parent=owner)
        window.closed.connect(lambda w: (closed.append(w), w.deleteLater()))
        window.show()
        window.close()
    assert len(closed) == len(PLUGIN_WINDOWS)
    closed.clear()
    del owner, window
    gc.collect()
    qt_wait(lambda: False, timeout=0.5)


def test_finished_card_stops_its_spinner(qapp, qt_wait):
    from qtpy.QtWidgets import QMainWindow

    from opticalib.gui.activity import ActivityCenter
    from opticalib.gui.kernel import Task

    window = QMainWindow()
    window.show()
    center = ActivityCenter(window)
    task = Task("x = 1", "spinning")
    card = center.track(task)
    task._start("m1")
    assert card._spin is not None
    qt_wait(lambda: False, timeout=0.1)  # let the animation start
    spin = card._spin
    task._finish("error", {"ename": "E", "evalue": "x"})
    assert card._spin is None
    timer = spin.info[spin.parent_widget][0] if spin.parent_widget in spin.info else None
    assert timer is None or not timer.isActive()
    window.close()


class TestBackendButton:
    """The status-bar switch of the xupy backend."""

    def test_states(self, qapp):
        from opticalib.gui.app import BackendButton

        button = BackendButton()
        assert not button.isEnabled()  # unknown until the kernel answers
        requested = []
        button.switch_requested.connect(requested.append)

        button.set_state(True, True)
        assert button.isEnabled() and button._glow.isEnabled()
        assert "GPU (CuPy)" in button.toolTip()
        button.click()
        assert requested == [False]  # on the GPU: switch to the CPU

        button.set_state(False, True)
        assert not button._glow.isEnabled()
        lit, dead, _ = button._colors()
        assert dead.lightness() != lit.lightness()
        button.click()
        assert requested == [False, True]

        button.set_state(False, False)  # no CuPy: nothing to switch to
        assert not button.isEnabled() and "No GPU available" in button.toolTip()


class TestRamBar:
    """The status-bar gauge of the kernel memory."""

    def test_format_bytes(self, qapp):
        from opticalib.gui.app import RamBar

        assert RamBar.format_bytes(512) == "512 B"
        assert RamBar.format_bytes(2048) == "2 KB"
        assert RamBar.format_bytes(1536 * 1024**2) == "1.5 GB"
        assert RamBar.format_bytes(3 * 1024**4) == "3.0 TB"

    def test_reading_and_levels(self, qapp):
        from opticalib.gui.app import RamBar
        from opticalib.gui.theme import theme

        gb = 1024**3
        bar = RamBar(lambda: None)
        bar.set_reading(2 * gb, 16 * gb, 0.5, gb)
        assert bar._bar.value() == 125 and bar._value.text() == "2.0 GB / 16.0 GB"
        assert "Kernel memory: 2.0 GB" in bar.toolTip() and "50% of 16.0 GB" in bar.toolTip()
        assert theme().tokens["accent"] in bar._bar.styleSheet()
        bar.set_reading(2 * gb, 16 * gb, 0.9)
        assert theme().tokens["warning"] in bar._bar.styleSheet()
        bar.set_reading(2 * gb, 16 * gb, 0.97)
        assert theme().tokens["danger"] in bar._bar.styleSheet()

    def test_hidden_without_kernel(self, qapp):
        from opticalib.gui.app import RamBar

        pid = [None]
        bar = RamBar(lambda: pid[0])
        bar.refresh()
        assert bar.isHidden()
        pid[0] = os.getpid()  # any live process will do
        bar.refresh()
        assert not bar.isHidden() and "/" in bar._value.text()
        bar._timer.stop()


def test_kernel_side_backend(monkeypatch):
    import json

    from opticalib.gui import kernel_side

    state = json.loads(kernel_side.backend())
    assert set(state) == {"on_gpu", "available"}
    assert isinstance(state["on_gpu"], bool) and isinstance(state["available"], bool)


class TestExperiments:
    """Resolving the experiment opened by File → Open experiment."""

    def test_resolve_experiment(self, qapp, tmp_path):
        from opticalib.gui.app import resolve_experiment

        sysconfig = tmp_path / "ExpA" / "SysConfig" / "configuration.yaml"
        sysconfig.parent.mkdir(parents=True)
        sysconfig.write_text("SYSTEM: {}\n")
        flat = tmp_path / "ExpB" / "configuration.yaml"
        flat.parent.mkdir()
        flat.write_text("SYSTEM: {}\n")
        assert resolve_experiment(str(tmp_path / "ExpA")) == str(sysconfig)
        assert resolve_experiment(str(tmp_path / "ExpB")) == str(flat)
        assert resolve_experiment(str(flat)) == str(flat)
        with pytest.raises(FileNotFoundError):
            resolve_experiment(str(tmp_path / "Nothing"))

    def test_experiment_name_skips_sysconfig(self, qapp):
        from opticalib.gui.app import _get_experiment_name

        assert _get_experiment_name("/data/ExpA/SysConfig/configuration.yaml") == "ExpA"
        assert _get_experiment_name("/data/ExpB/configuration.yaml") == "ExpB"
