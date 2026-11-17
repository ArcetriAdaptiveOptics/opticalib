"""
Configuration editor of CalpyGUI
================================

An in-app editor for the YAML configuration file with syntax highlighting
and live validation.  The file is written back only when its text changed
and it is a loadable OptiCalib configuration.
"""

import os
from typing import Callable, List, Optional, Tuple

import yaml
from qtpy.QtCore import QEvent, QRect, QRegularExpression, QSize, Qt, QTimer
from qtpy.QtGui import (
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme


class YamlHighlighter(QSyntaxHighlighter):
    """
    Minimal YAML syntax highlighter using the theme colors.

    Parameters
    ----------
    document : QTextDocument
        The document to highlight.
    """

    _RULES: List[Tuple[str, str, bool]] = [
        # (pattern, color token, bold)
        (r"^\s*-?\s*[^\s#:'\"][^:#]*(?=:(\s|$))", "accent", True),
        (r"\b-?\d+(\.\d+)?([eE][-+]?\d+)?\b", "warning", False),
        (r"\b(true|false|True|False|yes|no|null|None|~)\b", "warning", False),
        (r"'[^']*'|\"[^\"]*\"", "success", False),
        (r"^\s*-(?=\s)", "text_muted", True),
    ]

    def __init__(self, document) -> None:
        """Compile the rules and follow theme changes."""
        super().__init__(document)
        self._rules: List[Tuple[QRegularExpression, QTextCharFormat]] = []
        self._comment = QTextCharFormat()
        self._build_formats()
        theme().changed.connect(self._on_theme_changed)

    def _build_formats(self) -> None:
        t = theme()
        self._rules = []
        for pattern, token, bold in self._RULES:
            fmt = QTextCharFormat()
            fmt.setForeground(t.color(token))
            if bold:
                fmt.setFontWeight(QFont.Weight.DemiBold)
            self._rules.append((QRegularExpression(pattern), fmt))
        self._comment = QTextCharFormat()
        self._comment.setForeground(t.color("text_muted"))
        self._comment.setFontItalic(True)

    def _on_theme_changed(self) -> None:
        self._build_formats()
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt override
        """Highlight one line of YAML."""
        comment_at = _comment_start(text)
        code = text if comment_at < 0 else text[:comment_at]
        for regex, fmt in self._rules:
            iterator = regex.globalMatch(code)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)
        if comment_at >= 0:
            self.setFormat(comment_at, len(text) - comment_at, self._comment)


def _comment_start(line: str) -> int:
    """Return the index of the YAML comment in *line*, or -1."""
    quote = None
    for i, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or line[i - 1].isspace()):
            return i
    return -1


def validate_config_text(text: str) -> Optional[str]:
    """
    Check that *text* is a loadable OptiCalib configuration.

    Parameters
    ----------
    text : str
        YAML content to validate.

    Returns
    -------
    str or None
        A description of the problem, or ``None`` when *text* is valid.
    """
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or str(exc)
        if "'\\t'" in problem:
            problem += " (indent with spaces: YAML does not allow tabs)"
        if mark is not None:
            return f"Line {mark.line + 1}, column {mark.column + 1}: {problem}"
        return f"Invalid YAML syntax: {problem}"
    if not isinstance(config, dict):
        return "The configuration must be a YAML mapping of sections."
    if not isinstance(config.get("SYSTEM"), dict):
        return "The configuration must contain a 'SYSTEM' section."
    return None


class _LineNumberArea(QWidget):
    """Gutter of a :class:`CodeEditor` showing the line numbers."""

    def __init__(self, editor: "CodeEditor") -> None:
        """Attach the gutter to ``editor``."""
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Width of the widest line number."""
        return QSize(self._editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Draw the numbers of the visible lines."""
        self._editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    """
    Plain-text editor with line numbers and soft tabs.

    *Tab* inserts spaces up to the next indentation stop (YAML forbids tab
    characters), *Shift+Tab* removes one level, both acting on every
    selected line when the selection spans several; *Backspace* in the
    indentation removes one level.  Tab characters in pasted text are
    expanded to spaces.

    Parameters
    ----------
    indent : int, optional
        Width of one indentation level, in spaces.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, indent: int = 2, parent: Optional[QWidget] = None) -> None:
        """Create the editor and its line-number gutter."""
        super().__init__(parent)
        self.indent = indent
        self._gutter = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_margins)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        theme().changed.connect(self._on_theme_changed)
        self._update_margins()
        self._highlight_current_line()

    # -- line numbers -------------------------------------------------------

    def line_number_width(self) -> int:
        """Width of the gutter, in pixels."""
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * max(digits, 2)

    def _update_margins(self, *_) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_margins()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Resize the gutter when the font changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._update_margins()
            self.resizeEvent(None)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Keep the gutter as tall as the editor."""
        if event is not None:
            super().resizeEvent(event)
        area = self.contentsRect()
        self._gutter.setGeometry(
            QRect(area.left(), area.top(), self.line_number_width(), area.height())
        )

    def paint_line_numbers(self, event) -> None:
        """Paint the gutter (called by :class:`_LineNumberArea`)."""
        t = theme()
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), t.color("surface_alt"))
        painter.setFont(self.font())
        current = self.textCursor().blockNumber()
        width = self._gutter.width() - 6
        height = self.fontMetrics().height()
        block = self.firstVisibleBlock()
        top = round(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        )
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = block.blockNumber()
                painter.setPen(t.color("text" if number == current else "text_muted"))
                painter.drawText(
                    0, top, width, height, Qt.AlignmentFlag.AlignRight, str(number + 1)
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
        painter.end()

    def _highlight_current_line(self) -> None:
        selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(theme().color("surface_alt"))
            selection.format.setProperty(
                QTextCharFormat.Property.FullWidthSelection, True
            )
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)
        self.setExtraSelections(selections)
        self._gutter.update()

    def _on_theme_changed(self) -> None:
        self._highlight_current_line()

    # -- soft tabs ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Turn Tab / Shift+Tab / Backspace into indentation edits."""
        key = event.key()
        if self.isReadOnly():
            super().keyPressEvent(event)
        elif key == Qt.Key.Key_Backtab or (
            key == Qt.Key.Key_Tab
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._shift_lines(-1)
        elif key == Qt.Key.Key_Tab:
            if self._spans_lines():
                self._shift_lines(1)
            else:
                column = self.textCursor().positionInBlock()
                self.insertPlainText(" " * (self.indent - column % self.indent))
        elif key == Qt.Key.Key_Backspace and self._in_indentation():
            cursor = self.textCursor()
            column = cursor.positionInBlock()
            cursor.movePosition(
                QTextCursor.MoveOperation.Left,
                QTextCursor.MoveMode.KeepAnchor,
                (column - 1) % self.indent + 1,
            )
            cursor.removeSelectedText()
        else:
            super().keyPressEvent(event)

    def insertFromMimeData(self, source) -> None:  # noqa: N802 - Qt override
        """Paste with tab characters expanded to spaces."""
        if source.hasText():
            self.insertPlainText(source.text().expandtabs(self.indent))
        else:
            super().insertFromMimeData(source)

    def _spans_lines(self) -> bool:
        cursor = self.textCursor()
        doc = self.document()
        return cursor.hasSelection() and (
            doc.findBlock(cursor.selectionStart()).blockNumber()
            != doc.findBlock(cursor.selectionEnd()).blockNumber()
        )

    def _in_indentation(self) -> bool:
        cursor = self.textCursor()
        if cursor.hasSelection() or cursor.positionInBlock() == 0:
            return False
        before = cursor.block().text()[: cursor.positionInBlock()]
        return before.strip(" ") == ""

    def _shift_lines(self, direction: int) -> None:
        """Indent (+1) or dedent (-1) the lines touched by the cursor."""
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart())
        last = doc.findBlock(cursor.selectionEnd())
        if (
            cursor.hasSelection()
            and last != first
            and cursor.selectionEnd() == last.position()
        ):
            last = (
                last.previous()
            )  # a selection ending at a line start excludes that line
        edit = QTextCursor(doc)
        edit.beginEditBlock()
        block = first
        while block.isValid():
            edit.setPosition(block.position())
            if direction > 0:
                if block.text().strip():
                    edit.insertText(" " * self.indent)
            else:
                text = block.text()
                count = min(len(text) - len(text.lstrip(" ")), self.indent)
                edit.movePosition(
                    QTextCursor.MoveOperation.Right,
                    QTextCursor.MoveMode.KeepAnchor,
                    count,
                )
                edit.removeSelectedText()
            if block == last:
                break
            block = block.next()
        edit.endEditBlock()


class ConfigEditorDialog(QDialog):
    """
    Dialog to view and edit the YAML configuration file.

    The file is written back only when its text was modified and it parses as
    a valid OptiCalib configuration; the validation runs while typing and
    the *Save* button stays disabled while the text is invalid.  Closing
    with unsaved changes asks for confirmation.

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.
    on_saved : callable, optional
        Function called without arguments after the file has been saved.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        config_path: str,
        on_saved: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Open the config file and display its content."""
        super().__init__(parent)
        self.setWindowTitle(f"Configuration – {os.path.basename(config_path)}")
        self.resize(760, 680)

        self._config_path = config_path
        self._on_saved = on_saved
        self._original_text = ""
        self._error: Optional[str] = None

        path_label = QLabel(config_path)
        path_label.setProperty("muted", True)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._text_edit = CodeEditor(indent=2)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSizeF(max(font.pointSizeF(), 10.5))
        self._text_edit.setFont(font)
        self._text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = YamlHighlighter(self._text_edit.document())
        try:
            with open(config_path, "r") as f:
                self._original_text = f.read()
            self._text_edit.setPlainText(self._original_text)
        except OSError as exc:
            self._text_edit.setPlainText(f"Could not read file:\n{exc}")
            self._text_edit.setReadOnly(True)

        self._status = QLabel()
        self._status.setWordWrap(True)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.reject)
        self._btn_save = QPushButton("Save and Close")
        self._btn_save.setProperty("accent", True)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_and_close)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._save_and_close)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(self._btn_close)
        btn_row.addWidget(self._btn_save)
        layout = QVBoxLayout(self)
        layout.addWidget(path_label)
        layout.addWidget(self._text_edit, 1)
        layout.addLayout(btn_row)

        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(250)
        self._validate_timer.timeout.connect(self._revalidate)
        self._text_edit.textChanged.connect(self._on_text_changed)
        theme().changed.connect(self._update_status)
        self._revalidate()

    def goto_entry(self, section: str, name: str) -> bool:
        """
        Move the cursor to a ``DEVICES`` entry and select its name.

        Parameters
        ----------
        section : str
            ``DEVICES`` subsection, as written in the file.
        name : str
            Entry name.

        Returns
        -------
        bool
            Whether the entry was found.
        """
        from .device_registry import locate_entry

        lines = self._text_edit.toPlainText().splitlines(keepends=True)
        try:
            index, indent = locate_entry(lines, section, name)
        except ValueError:
            return False
        block = self._text_edit.document().findBlockByNumber(index)
        cursor = QTextCursor(block)
        cursor.movePosition(
            QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, indent
        )
        cursor.movePosition(
            QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(name)
        )
        self._text_edit.setTextCursor(cursor)
        self._text_edit.centerCursor()
        self._text_edit.setFocus()
        return True

    # Kept for backwards compatibility with the phase-0 API.
    validate_config_text = staticmethod(validate_config_text)

    def is_modified(self) -> bool:
        """
        Return whether the edited text differs from the file on disk.

        Returns
        -------
        bool
            ``True`` when there are unsaved changes.
        """
        if self._text_edit.isReadOnly():
            return False
        return self._text_edit.toPlainText() != self._original_text

    def _on_text_changed(self) -> None:
        self._btn_save.setEnabled(self.is_modified() and self._error is None)
        self._validate_timer.start()

    def _revalidate(self) -> None:
        if self._text_edit.isReadOnly():
            self._error = "The configuration file could not be read."
        else:
            self._error = validate_config_text(self._text_edit.toPlainText())
        self._btn_save.setEnabled(self.is_modified() and self._error is None)
        self._update_status()

    def _update_status(self) -> None:
        t = theme()
        if self._error is None:
            state = "unsaved changes" if self.is_modified() else "saved"
            self._status.setText(
                f"<span style='color:{t.tokens['success']}'>✓</span> Valid configuration · {state}"
            )
        else:
            self._status.setText(
                f"<span style='color:{t.tokens['danger']}'>✗</span> {self._error}"
            )

    def _save_and_close(self) -> None:
        """Validate and save the edited configuration, then close."""
        if not self.is_modified():
            self.accept()
            return

        new_content = self._text_edit.toPlainText()
        error = validate_config_text(new_content)
        if error is not None:
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                f"The configuration was not saved.\n\n{error}",
            )
            return

        try:
            with open(self._config_path, "w") as f:
                f.write(new_content)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"Could not save changes:\n{exc}",
            )
            return

        self._original_text = new_content
        if self._on_saved is not None:
            self._on_saved()
        self.accept()

    def reject(self) -> None:
        """Close the dialog, asking for confirmation if there are unsaved changes."""
        if self.is_modified():
            answer = QMessageBox.question(
                self,
                "Discard Changes",
                "The configuration has unsaved changes. Discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        super().reject()
