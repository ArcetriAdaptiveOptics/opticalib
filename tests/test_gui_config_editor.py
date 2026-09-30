"""
Tests for the CalpyGUI configuration editor (opticalib.gui.widgets.config_editor).
"""

import os

import pytest

pytest.importorskip("qtconsole")
pytest.importorskip("qtawesome")

from opticalib.gui.widgets import config_editor as ce  # noqa: E402

VALID_CONFIG = (
    "SYSTEM:\n  data_path: ''\nDEVICES:\n  CAMERAS:\n    GigaVision:\n      id: CAM1\n"
)


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid configuration file."""
    path = tmp_path / "configuration.yaml"
    path.write_text(VALID_CONFIG)
    return path


@pytest.mark.parametrize(
    "text, valid",
    [
        (VALID_CONFIG, True),
        ("SYSTEM: [unclosed", False),
        ("- just\n- a list\n", False),
        ("DEVICES: {}\n", False),
        ("", False),
    ],
)
def test_validate_config_text(text, valid):
    assert (ce.validate_config_text(text) is None) == valid
    # Backwards-compatible static method.
    assert (ce.ConfigEditorDialog.validate_config_text(text) is None) == valid


def test_validation_reports_the_line():
    error = ce.validate_config_text("SYSTEM:\n  data_path: ''\n  bad: [1, 2\n")
    assert error.startswith("Line ")


@pytest.mark.parametrize(
    "line, index",
    [
        ("key: value # comment", 11),
        ("# full line", 0),
        ("url: 'a#b' # c", 11),
        ("name: value#notacomment", -1),
        ("plain: 3", -1),
    ],
)
def test_comment_start(line, index):
    assert ce._comment_start(line) == index


class TestConfigEditorDialog:
    """Tests for the configuration editor dialog."""

    @pytest.fixture
    def critical_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ce.QMessageBox, "critical", lambda *args: calls.append(args))
        return calls

    def _make_dialog(self, path, saved):
        return ce.ConfigEditorDialog(str(path), on_saved=lambda: saved.append(1))

    def test_unchanged_text_is_not_written(self, qapp, config_file, critical_calls):
        saved = []
        mtime = os.stat(config_file).st_mtime_ns
        dlg = self._make_dialog(config_file, saved)
        assert not dlg.is_modified()
        assert not dlg._btn_save.isEnabled()
        dlg._save_and_close()
        assert dlg.result() == dlg.DialogCode.Accepted
        assert os.stat(config_file).st_mtime_ns == mtime
        assert saved == []

    def test_valid_change_is_saved(self, qapp, config_file, critical_calls):
        saved = []
        dlg = self._make_dialog(config_file, saved)
        new_text = VALID_CONFIG.replace("CAM1", "CAM2")
        dlg._text_edit.setPlainText(new_text)
        dlg._revalidate()
        assert dlg._btn_save.isEnabled()
        dlg._save_and_close()
        assert config_file.read_text() == new_text
        assert saved == [1]
        assert critical_calls == []
        assert dlg.result() == dlg.DialogCode.Accepted

    def test_invalid_change_disables_save(self, qapp, config_file, critical_calls):
        saved = []
        dlg = self._make_dialog(config_file, saved)
        dlg._text_edit.setPlainText("SYSTEM: [unclosed")
        dlg._revalidate()
        assert not dlg._btn_save.isEnabled()
        assert "Line" in dlg._status.text()
        # Saving anyway (e.g. with Ctrl+S) is refused.
        dlg._save_and_close()
        assert config_file.read_text() == VALID_CONFIG
        assert saved == []
        assert len(critical_calls) == 1
        assert dlg.result() != dlg.DialogCode.Accepted

    @pytest.mark.parametrize("discard, closed", [(True, True), (False, False)])
    def test_close_with_unsaved_changes_asks(self, qapp, config_file, monkeypatch, discard, closed):
        questions = []
        answer = ce.QMessageBox.StandardButton.Discard if discard else ce.QMessageBox.StandardButton.Cancel

        def fake_question(*args):
            questions.append(args)
            return answer

        monkeypatch.setattr(ce.QMessageBox, "question", fake_question)
        dlg = self._make_dialog(config_file, [])
        dlg.show()
        dlg._text_edit.setPlainText(VALID_CONFIG + "# edit\n")
        dlg.reject()
        assert len(questions) == 1
        assert dlg.isVisible() is not closed
        assert config_file.read_text() == VALID_CONFIG
        dlg.done(0)

    def test_unreadable_file(self, qapp, tmp_path):
        dlg = ce.ConfigEditorDialog(str(tmp_path / "missing.yaml"))
        assert dlg._text_edit.isReadOnly()
        assert not dlg.is_modified()
        assert not dlg._btn_save.isEnabled()

    def test_highlighter_follows_theme(self, qapp, config_file):
        from opticalib.gui.theme import theme

        dlg = self._make_dialog(config_file, [])
        before = dlg._highlighter._rules[0][1].foreground().color().name()
        mode = theme().mode
        try:
            theme().set_mode("dark" if not theme().is_dark else "light")
            after = dlg._highlighter._rules[0][1].foreground().color().name()
        finally:
            theme().set_mode(mode)
        assert before != after
