"""
Tests for the kernel-side helpers of CalpyGUI (opticalib.gui.kernel_side and
opticalib.gui.kernel_backend), and for the pure helpers of the kernel bridge.

The kernel-side code runs inside the IPython kernel in production; here it
is exercised in-process with an injected publisher.
"""

import json
import os

import numpy as np
import pytest

from opticalib.gui import kernel_side as ks


@pytest.fixture
def mpl_backend():
    """Switch pyplot to the CalpyGUI kernel backend for one test."""
    import matplotlib.pyplot as plt

    previous = plt.get_backend()
    plt.close("all")
    plt.switch_backend("module://opticalib.gui.kernel_backend")
    yield plt
    plt.close("all")
    plt.switch_backend(previous)


@pytest.fixture
def published(monkeypatch):
    """Collect the display_data payloads instead of publishing them."""
    messages = []
    monkeypatch.setattr(ks, "_default_publisher", messages.append)
    return messages


class TestFigures:
    """Tests for the figure publication."""

    def test_new_figure_is_published_once(self, mpl_backend, published):
        plt = mpl_backend
        plt.figure()
        plt.plot([1, 2, 3])
        plt.title("ramp")
        assert ks.publish_figures() == 1
        payload = published[-1][ks.FIGURE_MIME]
        assert payload["title"] == "ramp"
        import base64

        assert base64.b64decode(payload["png"]).startswith(b"\x89PNG")
        # Nothing changed: nothing published.
        assert ks.publish_figures() == 0

    def test_changed_figure_is_republished_in_place(self, mpl_backend, published):
        plt = mpl_backend
        fig = plt.figure()
        plt.plot([1, 2])
        ks.publish_figures()
        uid = published[-1][ks.FIGURE_MIME]["uid"]
        plt.plot([2, 1])
        assert ks.publish_figures() == 1
        assert published[-1][ks.FIGURE_MIME]["uid"] == uid
        assert fig._calpy_uid == uid

    def test_empty_figure_is_skipped(self, mpl_backend, published):
        mpl_backend.figure()
        assert ks.publish_figures() == 0

    def test_show_and_pause_publish(self, mpl_backend, published):
        plt = mpl_backend
        plt.figure()
        plt.plot([0, 1])
        plt.show()
        assert len(published) == 1
        plt.plot([1, 0])
        plt.pause(0.01)
        assert len(published) == 2

    def test_each_figure_has_its_own_uid(self, mpl_backend, published):
        plt = mpl_backend
        for i in range(2):
            plt.figure()
            plt.plot([i, i + 1])
        ks.publish_figures()
        uids = {m[ks.FIGURE_MIME]["uid"] for m in published}
        assert len(uids) == 2


class TestView:
    """Tests for :func:`kernel_side.view`."""

    def test_masked_image(self, tmp_path, monkeypatch, published):
        monkeypatch.setenv("CALPY_GUI_TMP", str(tmp_path))
        mask = np.zeros((4, 5), bool)
        mask[0, 0] = True
        img = np.ma.masked_array(np.arange(20.0).reshape(4, 5), mask=mask)
        path = ks.view(img, "img")
        payload = published[-1][ks.IMAGE_MIME]
        assert payload["title"] == "img"
        assert payload["shape"] == [4, 5]
        assert os.path.dirname(path) == str(tmp_path)
        with np.load(path) as npz:
            assert npz["mask"][0, 0]
            np.testing.assert_array_equal(npz["data"], img.data)

    def test_unmasked_cube(self, tmp_path, monkeypatch, published):
        monkeypatch.setenv("CALPY_GUI_TMP", str(tmp_path))
        path = ks.view(np.zeros((3, 4, 2)))
        with np.load(path) as npz:
            assert "mask" not in npz.files
        assert "3-D array" in published[-1][ks.IMAGE_MIME]["title"]

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2, 2)), np.array(["a", "b"])])
    def test_rejects_unviewable(self, bad, published):
        with pytest.raises((ValueError, TypeError)):
            ks.view(bad)


class _FakeDM:
    n_acts = 3

    def set_shape(self, cmd, differential=False):
        pass

    def get_shape(self):
        return np.zeros(3)


class _FakeInterf:
    def acquire_map(self, nframes=1, delay=0, rebin=1):
        return np.zeros((2, 2))


class TestWorkspace:
    """Tests for the workspace description."""

    def test_classification_and_filtering(self):
        import types

        ks._baseline.clear()
        ns = {
            "img": np.zeros((2, 3)),
            "dm": _FakeDM(),
            "interf": _FakeInterf(),
            "n": 42,
            "_private": 1,
            "np": np,
            "fn": lambda: None,
            "cls": dict,
            "mod": types,
        }
        items = {i["name"]: i for i in ks.workspace_items(ns)}
        assert set(items) == {"img", "dm", "interf", "n"}
        assert items["img"]["kind"] == "array"
        assert items["img"]["summary"] == "(2, 3) float64"
        assert items["dm"]["kind"] == "dm"
        assert items["interf"]["kind"] == "interferometer"
        assert items["n"]["kind"] == "other"
        assert items["dm"]["module"] == __name__

    def test_baseline_names_are_hidden(self):
        value = object()
        ks._baseline.clear()
        ks._baseline["preset"] = id(value)
        try:
            names = [i["name"] for i in ks.workspace_items({"preset": value})]
            assert names == []
            # Rebinding the name makes it visible again.
            names = [i["name"] for i in ks.workspace_items({"preset": object()})]
            assert names == ["preset"]
        finally:
            ks._baseline.clear()

    def test_folders(self):
        info = json.loads(ks.folders())
        assert set(info) == {"base", "config", "categories", "paths"}
        assert "IFFUNCTIONS_ROOT_FOLDER" in info["paths"]
        labels = [label for label, _ in info["categories"]]
        assert "OPD images" in labels and "IF functions" in labels


class TestBridgeHelpers:
    """Tests for the pure helpers of opticalib.gui.kernel."""

    @pytest.fixture(autouse=True)
    def _kernel_module(self, qapp):
        pytest.importorskip("qtconsole")
        from opticalib.gui import kernel

        self.kernel = kernel

    @pytest.mark.parametrize(
        "text, fraction, detail",
        [
            ("1/88\r2/88\r3/88\r", 3 / 88, "3/88"),
            ("Computing interaction matrix...\n", None, "Computing interaction matrix..."),
            (" 42%|████      | 42/100 [00:01<00:02]", 0.42, " 42%|████      | 42/100 [00:01<00:02]".strip()),
            ("\x1b[32m7/10\x1b[0m\n", 0.7, "7/10"),
            ("", None, ""),
            ("value 12/0", None, "value 12/0"),
        ],
    )
    def test_parse_progress(self, text, fraction, detail):
        got_fraction, got_detail = self.kernel.parse_progress(text)
        if fraction is None:
            assert got_fraction is None
        else:
            assert got_fraction == pytest.approx(fraction)
        assert got_detail == detail

    def test_parse_user_expressions(self):
        payload = json.dumps([{"name": "x"}])
        replies = {
            "json": {"status": "ok", "data": {"text/plain": repr(payload)}},
            "number": {"status": "ok", "data": {"text/plain": "42"}},
            "text": {"status": "ok", "data": {"text/plain": "<object at 0x1>"}},
            "fail": {"status": "error", "ename": "NameError", "evalue": "x"},
        }
        results = self.kernel._parse_user_expressions(replies)
        assert results["json"] == [{"name": "x"}]
        assert results["number"] == 42
        assert results["text"] == "<object at 0x1>"
        assert isinstance(results["fail"], RuntimeError)

    def test_task_lifecycle(self):
        done, failed = [], []
        task = self.kernel.Task("x = 1", on_done=done.append, on_error=failed.append)
        assert task.state == "queued" and task.title == "x = 1"
        task._start("m1")
        task._update_output("5/10\r")
        assert task.progress == pytest.approx(0.5)
        task._finish("done")
        assert task.is_final and done == [task] and failed == []
        task._finish("error")  # a final task never changes again
        assert task.state == "done"

    def test_kernel_env(self, tmp_path):
        bridge = self.kernel.KernelBridge(str(tmp_path / "c.yaml"), None, str(tmp_path))
        env = bridge.kernel_env()
        assert env["AOCONF"] == str(tmp_path / "c.yaml")
        assert env["MPLBACKEND"] == self.kernel.KERNEL_MPL_BACKEND
        assert env["CALPY_GUI_TMP"] == str(tmp_path)


class _Hostile:
    """Like a network proxy: touching instance attributes is expensive or fails."""

    touched = []

    def __getattr__(self, name):
        _Hostile.touched.append(name)
        raise ConnectionError("remote call")

    def __repr__(self):
        raise RuntimeError("no repr")


class TestWorkspaceRobustness:
    def test_hostile_objects_do_not_break_the_listing(self):
        ks._baseline.clear()
        _Hostile.touched.clear()
        ns = {"proxy": _Hostile(), 3: "non-str key", "ok": 1}
        items = {i["name"]: i for i in ks.workspace_items(ns)}
        assert set(items) == {"proxy", "ok"}
        assert items["proxy"]["kind"] == "other"
        assert _Hostile.touched == []  # classification never probed the instance

    def test_post_execute_skips_other_backends(self, monkeypatch):
        import matplotlib

        calls = []
        monkeypatch.setattr(ks, "publish_figures", lambda: calls.append(1))
        monkeypatch.setattr(matplotlib, "get_backend", lambda: "qtagg")
        ks._post_execute()
        assert calls == []
        monkeypatch.setattr(matplotlib, "get_backend", lambda: ks.BACKEND)
        ks._post_execute()
        assert calls == [1]
