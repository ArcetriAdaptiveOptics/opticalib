"""
Connect dialog of CalpyGUI
==========================

Opened from a device card when an entry cannot be connected in one click
(unknown class, missing fields, ...) or with *Connect with options…*.  It
lets the user pick the device class and the variable name, shows what the
chosen class needs from the configuration entry, and previews the command,
which can be edited before running.  The chosen class can be saved in the
entry as ``class: <name>`` so the next connection takes one click.
"""

import keyword
from typing import Optional

from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme
from .device_registry import (
    DEVICE_CLASSES,
    DeviceEntry,
    build_command,
    classes_for,
    resolve_entry,
)


class ConnectDialog(QDialog):
    """
    Choose how to connect a configuration entry.

    Parameters
    ----------
    entry : DeviceEntry
        The resolved configuration entry.
    var_name : str, optional
        Variable name to propose (defaults to the section variable).
    parent : QWidget, optional
        Parent widget.

    Notes
    -----
    After ``exec()`` returns :attr:`EDIT_CONFIG`, the caller should open the
    configuration editor on the entry.
    """

    #: ``exec()`` result asking to edit the configuration entry.
    EDIT_CONFIG = 2

    def __init__(
        self,
        entry: DeviceEntry,
        var_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build the dialog for *entry*."""
        super().__init__(parent)
        self.setWindowTitle(f"Connect {entry.name}")
        self.resize(620, 480)
        self._entry = entry
        self._code_edited = False
        self._setting_code = False
        self._current: DeviceEntry = entry

        heading = QLabel(f"{entry.name}")
        heading.setProperty("heading", True)
        where = QLabel(f"Configuration entry in DEVICES → {entry.section}")
        where.setProperty("muted", True)

        self._class_combo = QComboBox()
        own = classes_for(entry.section)
        others = [c for c in DEVICE_CLASSES if c not in own]
        for device_class in own + others:
            self._class_combo.addItem(
                f"{device_class.name} — {device_class.description}", device_class.name
            )
        if own and others:
            self._class_combo.insertSeparator(len(own))
        if entry.device_class is not None:
            self._class_combo.setCurrentIndex(
                self._class_combo.findData(entry.device_class.name)
            )
        else:
            self._class_combo.setCurrentIndex(-1)
            self._class_combo.setPlaceholderText("Choose the device class")
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)

        self._var_edit = QLineEdit(var_name or entry.var_name)
        self._var_edit.textChanged.connect(self._on_var_changed)

        self._problems = QLabel()
        self._problems.setWordWrap(True)

        self._code = QPlainTextEdit()
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._code.setFont(font)
        self._code.textChanged.connect(self._on_code_edited)
        self._btn_reset = QPushButton("Reset code")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_code)

        self._save_class = QCheckBox()
        self._save_class.toggled.connect(self._refresh_buttons)

        form = QFormLayout()
        form.addRow("Device class", self._class_combo)
        form.addRow("Variable", self._var_edit)

        code_header = QHBoxLayout()
        code_label = QLabel("Command")
        code_label.setProperty("section", True)
        code_header.addWidget(code_label)
        code_header.addStretch()
        code_header.addWidget(self._btn_reset)

        self._btn_edit = QPushButton("Edit configuration…")
        self._btn_edit.clicked.connect(lambda: self.done(self.EDIT_CONFIG))
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setProperty("accent", True)
        self._btn_connect.setDefault(True)
        self._btn_connect.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self._btn_edit)
        buttons.addStretch()
        buttons.addWidget(self._btn_cancel)
        buttons.addWidget(self._btn_connect)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(where)
        layout.addSpacing(6)
        layout.addLayout(form)
        layout.addWidget(self._problems)
        layout.addLayout(code_header)
        layout.addWidget(self._code, 1)
        layout.addWidget(self._save_class)
        layout.addLayout(buttons)

        theme().changed.connect(self._refresh_problems)
        self._on_class_changed()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def class_name(self) -> Optional[str]:
        """The chosen device class name (``None`` if none is chosen)."""
        return self._class_combo.currentData()

    def var_name(self) -> str:
        """The variable the device is bound to."""
        return self._var_edit.text().strip()

    def code(self) -> str:
        """The command to run (possibly edited by the user)."""
        return self._code.toPlainText()

    def save_class(self) -> bool:
        """Whether to write ``class: <name>`` into the configuration entry."""
        return self._save_class.isEnabled() and self._save_class.isChecked()

    def resolved(self) -> DeviceEntry:
        """The entry resolved with the chosen class."""
        return self._current

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_class_changed(self, *args) -> None:
        name = self.class_name()
        entry = self._entry
        if name is None:
            self._current = entry
        else:
            conf = dict(entry.conf)
            conf["class"] = name
            self._current = resolve_entry(entry.section, entry.name, conf)
        self._refresh_problems()
        if not self._code_edited:
            self._reset_code()
        explicit = str(entry.conf.get("class") or "")
        by_name = (
            entry.device_class.name
            if entry.device_class and entry.source != "explicit"
            else None
        )
        self._save_class.setText(
            f"Save 'class: {name}' in the configuration entry"
            if name
            else "Save the class in the configuration entry"
        )
        already = name is not None and explicit.split(".")[-1].lower() == name.lower()
        self._save_class.setEnabled(name is not None and not already)
        self._save_class.setChecked(
            name is not None and not already and name != by_name
        )
        self._refresh_buttons()

    def _on_var_changed(self, *args) -> None:
        if not self._code_edited:
            self._reset_code()
        self._refresh_buttons()

    def _reset_code(self) -> None:
        self._setting_code = True
        try:
            self._code.setPlainText(
                build_command(self._current, self.var_name() or None)
            )
        finally:
            self._setting_code = False
        self._code_edited = False
        self._btn_reset.setEnabled(False)

    def _on_code_edited(self) -> None:
        if not self._setting_code:
            self._code_edited = True
            self._btn_reset.setEnabled(True)
        self._refresh_buttons()

    def _refresh_problems(self) -> None:
        t = theme()
        entry = self._current
        if self.class_name() is None:
            text = f"<span style='color:{t.tokens['warning']}'>Choose the class of this device.</span>"
        elif entry.problems:
            items = "".join(f"<li>{p}</li>" for p in entry.problems)
            text = (
                f"<span style='color:{t.tokens['warning']}'>This entry is not ready for "
                f"{entry.device_class.name}:</span><ul style='margin:2px 0'>{items}</ul>"
                "You can fix the configuration, or edit the command below."
            )
        else:
            notes = "".join(f"<br>{n}" for n in entry.notes)
            text = f"<span style='color:{t.tokens['success']}'>✓ Ready to connect.</span>{notes}"
        self._problems.setText(text)

    def _refresh_buttons(self) -> None:
        var = self.var_name()
        valid_var = var.isidentifier() and not keyword.iskeyword(var)
        self._var_edit.setToolTip(
            "" if valid_var else "Not a valid Python variable name"
        )
        self._btn_connect.setEnabled(
            valid_var and bool(self.code().strip()) and self.class_name() is not None
        )
