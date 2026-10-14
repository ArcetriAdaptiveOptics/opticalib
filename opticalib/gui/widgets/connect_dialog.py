"""
Connect dialog of CalpyGUI
==========================

Opened from a device card when an entry cannot be connected in one click
(unknown class, missing fields, ...) or with *Connect with options…*.  It
lets the user pick the device class and the variable name, shows what the
chosen class needs from the configuration entry, and previews the command,
which can be edited before running.  The chosen class can be saved in the
entry as ``class: <name>`` so the next connection takes one click.

Besides the opticalib classes, the experiment's own classes can be used:
*Custom … class* asks for the class name, the module (or ``.py`` file of the
experiment folder) and the constructor arguments, and can declare the class
in the ``DEVICE.CLASSES`` section of the configuration, so that it is known
to every later session.
"""

import keyword
import os
import re
from typing import Dict, List, Optional

from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
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
    CUSTOM_CLASSES_SECTION,
    DEVICE_CLASSES,
    SECTION_KINDS,
    DeviceClass,
    DeviceEntry,
    build_command,
    classes_for,
    custom_class,
    module_name,
    normalize_section,
    resolve_entry,
)

#: Combo-box data of the *Custom class* choice.
CUSTOM = "__custom__"

#: How the devices of each kind are called in the dialog.
_KIND_LABELS: Dict[str, str] = {
    "dm": "DM",
    "interferometer": "interferometer",
    "camera": "camera",
    "wfs": "WFS",
    "motor": "motor",
}


def find_class_file(experiment: str, class_name: str) -> str:
    """
    Look for the ``.py`` file of the experiment folder defining a class.

    Only the files directly in the folder are searched.

    Parameters
    ----------
    experiment : str
        The experiment folder.
    class_name : str
        Class name.

    Returns
    -------
    str
        The file name (relative to the folder), or ``''`` if not found.
    """
    if not experiment or not class_name.isidentifier():
        return ""
    pattern = re.compile(rf"^class\s+{re.escape(class_name)}\b", re.MULTILINE)
    try:
        names = sorted(os.listdir(experiment))
    except OSError:
        return ""
    for name in names:
        path = os.path.join(experiment, name)
        if not name.endswith(".py") or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > 2_000_000:
                continue
            with open(path, "r", errors="replace") as f:
                if pattern.search(f.read()):
                    return name
        except OSError:
            continue
    return ""


class ConnectDialog(QDialog):
    """
    Choose how to connect a configuration entry.

    Parameters
    ----------
    entry : DeviceEntry
        The resolved configuration entry.
    var_name : str, optional
        Variable name to propose (defaults to the section variable).
    classes : list of DeviceClass, optional
        Classes to offer, custom ones included (default:
        :data:`DEVICE_CLASSES`; see
        :func:`~.device_registry.known_classes`).
    experiment : str, optional
        The experiment folder, where custom-class files are looked for.
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
        *,
        classes: Optional[List[DeviceClass]] = None,
        experiment: str = "",
    ) -> None:
        """Build the dialog for *entry*."""
        super().__init__(parent)
        self.setWindowTitle(f"Connect {entry.name}")
        self.resize(640, 540)
        self._entry = entry
        self._classes = list(DEVICE_CLASSES if classes is None else classes)
        self._experiment = experiment
        self._code_edited = False
        self._setting_code = False
        self._current: DeviceEntry = entry
        self._custom_error: Optional[str] = None
        # Choice the "save" default was last set for (kept while typing).
        self._save_default_for: object = object()
        kind = _KIND_LABELS.get(
            SECTION_KINDS.get(normalize_section(entry.section), ""), "device"
        )

        heading = QLabel(f"{entry.name}")
        heading.setProperty("heading", True)
        where = QLabel(f"Configuration entry in DEVICES → {entry.section}")
        where.setProperty("muted", True)

        self._class_combo = QComboBox()
        own = classes_for(entry.section, self._classes)
        others = [c for c in self._classes if c not in own]
        for device_class in own + others:
            self._class_combo.addItem(
                f"{device_class.name} — {device_class.description}", device_class.name
            )
        if own and others:
            self._class_combo.insertSeparator(len(own))
        self._class_combo.insertSeparator(self._class_combo.count())
        self._class_combo.addItem(f"Custom {kind} class…", CUSTOM)
        if entry.device_class is not None:
            self._class_combo.setCurrentIndex(
                self._class_combo.findData(entry.device_class.name)
            )
        else:
            self._class_combo.setCurrentIndex(-1)
            self._class_combo.setPlaceholderText("Choose the device class")

        self._var_edit = QLineEdit(var_name or entry.var_name)
        self._var_edit.textChanged.connect(self._on_var_changed)

        # Custom class definition.
        name_guess = re.sub(r"[^0-9A-Za-z_]+", "", re.sub(r"\d+$", "", entry.name))
        if not name_guess.isidentifier():
            name_guess = ""
        file_guess = find_class_file(experiment, name_guess)
        suffix = entry.name[len(name_guess):] if name_guess else ""
        self._custom_name = QLineEdit(name_guess)
        self._custom_name.setPlaceholderText("e.g. MyDM")
        self._custom_module = QLineEdit(file_guess)
        self._custom_module.setPlaceholderText(
            "module name, or .py file of the experiment folder"
        )
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.clicked.connect(self._browse_module)
        self._custom_args = QLineEdit("{suffix}" if suffix else "")
        self._custom_args.setPlaceholderText("constructor arguments, e.g. {suffix}, n=2")
        hint = QLabel(
            "In the arguments, {name} is the entry name and {suffix} what follows "
            f"the class name (here {suffix or 'nothing'!r}). The experiment folder"
            + (f" ({experiment})" if experiment else "")
            + " is importable from the kernel."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        module_row = QHBoxLayout()
        module_row.setContentsMargins(0, 0, 0, 0)
        module_row.addWidget(self._custom_module, 1)
        module_row.addWidget(self._btn_browse)
        custom_form = QFormLayout()
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.addRow("Class name", self._custom_name)
        custom_form.addRow("Module", module_row)
        custom_form.addRow("Arguments", self._custom_args)
        custom_form.addRow(hint)
        self._custom_box = QWidget()
        self._custom_box.setLayout(custom_form)
        for edit in (self._custom_name, self._custom_module, self._custom_args):
            edit.textChanged.connect(self._on_class_changed)

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
        layout.addWidget(self._custom_box)
        layout.addWidget(self._problems)
        layout.addLayout(code_header)
        layout.addWidget(self._code, 1)
        layout.addWidget(self._save_class)
        layout.addLayout(buttons)

        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        theme().changed.connect(self._refresh_problems)
        self._on_class_changed()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def is_custom(self) -> bool:
        """Whether a custom class is being defined."""
        return self._class_combo.currentData() == CUSTOM

    def class_name(self) -> Optional[str]:
        """The chosen device class name (``None`` if none is chosen)."""
        if self.is_custom():
            name = self._custom_name.text().strip()
            return name if name.isidentifier() else None
        return self._class_combo.currentData()

    def var_name(self) -> str:
        """The variable the device is bound to."""
        return self._var_edit.text().strip()

    def code(self) -> str:
        """The command to run (possibly edited by the user)."""
        return self._code.toPlainText()

    def save_class(self) -> bool:
        """
        Whether to save the class in the configuration.

        For an opticalib class, ``class: <name>`` goes in the entry; for a
        custom class, its definition goes in ``DEVICE.CLASSES`` (and the
        entry names it, when its name does not).
        """
        return self._save_class.isEnabled() and self._save_class.isChecked()

    def custom_definition(self) -> Optional[Dict[str, str]]:
        """
        The custom class being defined.

        Returns
        -------
        dict or None
            ``name``, ``module`` (as typed), ``section`` and ``args``; ``None``
            when an opticalib class is chosen or the definition is invalid.
        """
        if not self.is_custom() or self._custom_error is not None:
            return None
        return {
            "name": self._custom_name.text().strip(),
            "module": self._custom_module.text().strip(),
            "section": self._entry.section,
            "args": self._custom_args.text().strip(),
        }

    def resolved(self) -> DeviceEntry:
        """The entry resolved with the chosen class."""
        return self._current

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _custom_class(self) -> Optional[DeviceClass]:
        """Validate the custom-class fields (sets :attr:`_custom_error`)."""
        name = self._custom_name.text().strip()
        module = self._custom_module.text().strip()
        self._custom_error = None
        if not name.isidentifier() or keyword.iskeyword(name):
            self._custom_error = "Enter the class name."
        elif not module:
            self._custom_error = "Enter the module, or the .py file, defining the class."
        else:
            try:
                dotted = module_name(module, self._experiment)
            except ValueError as exc:
                self._custom_error = str(exc)
            else:
                return custom_class(
                    name, dotted, self._entry.section, self._custom_args.text().strip()
                )
        return None

    def _browse_module(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Module defining the class", self._experiment, "Python files (*.py)"
        )
        if not path:
            return
        if self._experiment:
            relative = os.path.relpath(path, self._experiment)
            if not relative.startswith(os.pardir):
                path = relative
        self._custom_module.setText(path)
        if not self._custom_name.text().strip():
            try:
                with open(os.path.join(self._experiment, path), "r", errors="replace") as f:
                    found = re.search(r"^class\s+([A-Za-z_]\w*)", f.read(), re.MULTILINE)
            except OSError:
                found = None
            if found:
                self._custom_name.setText(found.group(1))

    def _on_class_changed(self, *args) -> None:
        entry = self._entry
        custom = self.is_custom()
        self._custom_box.setVisible(custom)
        name = self.class_name()
        if custom:
            device_class = self._custom_class()
            if device_class is None:
                self._current = entry
            else:
                conf = dict(entry.conf)
                conf["class"] = device_class.name
                self._current = resolve_entry(
                    entry.section, entry.name, conf, [device_class] + self._classes
                )
        elif name is None:
            self._current = entry
        else:
            conf = dict(entry.conf)
            conf["class"] = name
            self._current = resolve_entry(entry.section, entry.name, conf, self._classes)
        self._refresh_problems()
        if not self._code_edited:
            self._reset_code()
        choice = self._class_combo.currentData()
        set_default = choice != self._save_default_for
        self._save_default_for = choice
        if custom:
            self._save_class.setText(
                f"Save the class in the configuration ({CUSTOM_CLASSES_SECTION}), "
                "for the next sessions"
            )
            self._save_class.setEnabled(self._custom_error is None)
            if set_default:
                self._save_class.setChecked(True)
        else:
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
            if set_default:
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
            if self.is_custom() and self._custom_error is not None:
                self._code.setPlainText("")
            else:
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
        warn = t.tokens["warning"]
        if self.is_custom() and self._custom_error is not None:
            text = f"<span style='color:{warn}'>{self._custom_error}</span>"
        elif self.class_name() is None:
            text = f"<span style='color:{warn}'>Choose the class of this device.</span>"
        elif entry.problems:
            items = "".join(f"<li>{p}</li>" for p in entry.problems)
            text = (
                f"<span style='color:{warn}'>This entry is not ready for "
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
