"""
Tests for the CalpyGUI configuration editor (opticalib.gui.widgets.config_editor).
"""

import os

import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)
pytest.importorskip("qtawesome", exc_type=ImportError)

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


def test_tab_error_is_explained():
    error = ce.validate_config_text("SYSTEM:\n\tdata_path: ''\n")
    assert error.startswith("Line 2") and "indent with spaces" in error


class TestCodeEditor:
    """Line numbers and soft tabs of the editor."""

    @staticmethod
    def _key(editor, key, modifiers=None):
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QKeyEvent
        from qtpy.QtCore import QEvent

        modifiers = modifiers or Qt.KeyboardModifier.NoModifier
        editor.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifiers))

    @staticmethod
    def _cursor_at(editor, line, column):
        from qtpy.QtGui import QTextCursor

        cursor = QTextCursor(editor.document().findBlockByNumber(line))
        cursor.movePosition(QTextCursor.MoveOperation.Right, n=column)
        editor.setTextCursor(cursor)

    def test_tab_inserts_spaces(self, qapp):
        from qtpy.QtCore import Qt

        editor = ce.CodeEditor(indent=2)
        editor.setPlainText("SYSTEM:\nkey: 1\n")
        self._cursor_at(editor, 1, 0)
        self._key(editor, Qt.Key.Key_Tab)
        assert editor.toPlainText() == "SYSTEM:\n  key: 1\n"
        self._cursor_at(editor, 1, 1)  # to the next indentation stop
        self._key(editor, Qt.Key.Key_Tab)
        assert editor.toPlainText() == "SYSTEM:\n   key: 1\n"  # one space, up to column 2
        assert "\t" not in editor.toPlainText()
        assert ce.validate_config_text(editor.toPlainText()) is None

    def test_backspace_and_shift_tab_dedent(self, qapp):
        from qtpy.QtCore import Qt

        editor = ce.CodeEditor(indent=2)
        editor.setPlainText("SYSTEM:\n    key: 1\n")
        self._cursor_at(editor, 1, 4)
        self._key(editor, Qt.Key.Key_Backspace)
        assert editor.toPlainText() == "SYSTEM:\n  key: 1\n"
        self._key(editor, Qt.Key.Key_Backtab)
        assert editor.toPlainText() == "SYSTEM:\nkey: 1\n"

    def test_block_indent(self, qapp):
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QTextCursor

        editor = ce.CodeEditor(indent=2)
        editor.setPlainText("a: 1\nb: 2\n\nc: 3\n")
        cursor = editor.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(editor.document().findBlockByNumber(3).position(), QTextCursor.MoveMode.KeepAnchor)
        editor.setTextCursor(cursor)  # lines 1-3; the selection ends at the start of line 4
        self._key(editor, Qt.Key.Key_Tab)
        assert editor.toPlainText() == "  a: 1\n  b: 2\n\nc: 3\n"
        self._key(editor, Qt.Key.Key_Backtab)
        assert editor.toPlainText() == "a: 1\nb: 2\n\nc: 3\n"

    def test_paste_expands_tabs(self, qapp):
        from qtpy.QtCore import QMimeData

        editor = ce.CodeEditor(indent=2)
        data = QMimeData()
        data.setText("SYSTEM:\n\tdata_path: ''\n")
        editor.insertFromMimeData(data)
        assert editor.toPlainText() == "SYSTEM:\n  data_path: ''\n"

    def test_line_numbers(self, qapp):
        editor = ce.CodeEditor()
        editor.resize(400, 300)
        editor.setPlainText("\n".join(f"k{i}: {i}" for i in range(150)))
        width = editor.line_number_width()
        assert editor.viewportMargins().left() == width
        assert width > editor.fontMetrics().horizontalAdvance("999")
        editor.grab()  # paints the gutter without errors
