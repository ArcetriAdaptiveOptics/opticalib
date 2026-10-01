"""
Parameters of procedure steps
=============================

Each :class:`Param` describes one argument of a procedure step: it builds
its input widget, validates the value and renders it as Python source, so
steps can generate the code that is run (and echoed) in the console.

Values are rendered as literals (``repr``) except for the parameters that
name kernel objects (devices, output variables) or are Python expressions.
"""

import keyword
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

from ..theme import theme


class ParamError(ValueError):
    """Invalid parameter value (the message is shown to the user)."""


class Param(QObject):
    """
    Base class of step parameters.

    Parameters
    ----------
    name : str
        Identifier used by the step template.
    label : str
        Label shown in the form.
    help : str, optional
        Tooltip.
    optional : bool, optional
        Whether an empty value is accepted (rendered as ``None``).
    config_default : str, optional
        Kernel expression giving the value used when the parameter is left
        empty (e.g. read from the configuration file); the window shows it
        as a hint (see :meth:`set_config_hint`).

    Signals
    -------
    changed
        The value changed.
    """

    changed = Signal()

    def __init__(
        self,
        name: str,
        label: str,
        help: str = "",
        optional: bool = False,
        config_default: str = "",
    ) -> None:
        """Store the description; the widget is created by :meth:`widget`."""
        super().__init__()
        self.name = name
        self.label = label
        self.help = help
        self.optional = optional
        self.config_default = config_default
        self._widget: Optional[QWidget] = None
        self.context = None

    def widget(self, context=None) -> QWidget:
        """
        Return the input widget (created on first call).

        Parameters
        ----------
        context : ProcedureContext, optional
            Access to the kernel workspace and data folders.

        Returns
        -------
        QWidget
            The widget.
        """
        if self._widget is None:
            self.context = context
            self._widget = self._make_widget()
            if self.help:
                self._widget.setToolTip(self.help)
            self._setup()
        return self._widget

    def _make_widget(self) -> QWidget:  # pragma: no cover - abstract
        raise NotImplementedError

    def _setup(self) -> None:
        """Connect the widget to the context (called once it is stored)."""

    def set_config_hint(self, text: str) -> None:
        """
        Show the default value used when the parameter is left empty.

        Parameters
        ----------
        text : str
            The value, as read from the configuration (e.g. ``'0.05'``).
        """
        edit = self._hint_edit()
        if edit is not None:
            edit.setPlaceholderText(text)
            edit.setToolTip("\n".join(filter(None, [self.help, f"Default from the configuration: {text}"])))

    def _hint_edit(self):
        """The line edit showing the default hint (``None`` if there is none)."""
        widget = self.widget()
        if isinstance(widget, QLineEdit):
            return widget
        if isinstance(widget, QComboBox) and widget.isEditable():
            return widget.lineEdit()
        return None

    def code(self) -> str:
        """
        Return the value as Python source.

        Returns
        -------
        str
            Source code of the value.

        Raises
        ------
        ParamError
            If the value is invalid.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    def value(self) -> Any:
        """The current (Python) value, for display and state."""
        raise NotImplementedError  # pragma: no cover - abstract

    def set_value(self, value: Any) -> None:
        """Set the value programmatically."""
        raise NotImplementedError  # pragma: no cover - abstract

    def _error(self, message: str) -> ParamError:
        return ParamError(f"{self.label}: {message}")


class IntParam(Param):
    """Integer value, in a spin box."""

    def __init__(self, name, label, default=0, minimum=0, maximum=10**9, **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.default, self.minimum, self.maximum = int(default), int(minimum), int(maximum)

    def _make_widget(self) -> QWidget:
        box = QSpinBox()
        box.setRange(self.minimum, self.maximum)
        box.setValue(self.default)
        box.valueChanged.connect(lambda *_: self.changed.emit())
        return box

    def value(self) -> int:
        """The integer value."""
        return self.widget().value()

    def set_value(self, value) -> None:
        """Set the value."""
        self.widget().setValue(int(value))

    def code(self) -> str:
        """The integer literal."""
        return str(self.value())


class FloatParam(Param):
    """Float value typed as text (scientific notation accepted, e.g. ``7e-8``)."""

    def __init__(self, name, label, default=0.0, minimum=None, maximum=None, **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.default, self.minimum, self.maximum = default, minimum, maximum

    def _make_widget(self) -> QWidget:
        edit = QLineEdit("" if self.default is None else repr(float(self.default)))
        edit.setPlaceholderText("None" if self.optional else "")
        edit.textChanged.connect(lambda *_: self.changed.emit())
        return edit

    def value(self) -> Optional[float]:
        """The float value (``None`` when empty and optional)."""
        text = self.widget().text().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("a number is required")
        try:
            number = float(text)
        except ValueError:
            raise self._error(f"'{text}' is not a number") from None
        if self.minimum is not None and number < self.minimum:
            raise self._error(f"must be ≥ {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise self._error(f"must be ≤ {self.maximum}")
        return number

    def set_value(self, value) -> None:
        """Set the value."""
        self.widget().setText("" if value is None else repr(float(value)))

    def code(self) -> str:
        """The float literal (or ``None``)."""
        number = self.value()
        return "None" if number is None else repr(number)


class BoolParam(Param):
    """Boolean value, in a check box."""

    def __init__(self, name, label, default=False, **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.default = bool(default)

    def _make_widget(self) -> QWidget:
        box = QCheckBox()
        box.setChecked(self.default)
        box.toggled.connect(lambda *_: self.changed.emit())
        return box

    def value(self) -> bool:
        """The boolean value."""
        return self.widget().isChecked()

    def set_value(self, value) -> None:
        """Set the value."""
        self.widget().setChecked(bool(value))

    def code(self) -> str:
        """``True`` or ``False``."""
        return repr(self.value())


class ChoiceParam(Param):
    """
    One value among fixed choices.

    Parameters
    ----------
    choices : sequence of (value, label)
        The choices; values are rendered with ``repr``.
    """

    def __init__(self, name, label, choices: Sequence[Tuple[Any, str]], default=None, **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.choices = list(choices)
        self.default = default if default is not None else self.choices[0][0]

    def _make_widget(self) -> QWidget:
        combo = QComboBox()
        for value, text in self.choices:
            combo.addItem(text, value)
        index = next((i for i, (v, _) in enumerate(self.choices) if v == self.default), 0)
        combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        return combo

    def value(self) -> Any:
        """The chosen value."""
        return self.widget().currentData()

    def set_value(self, value) -> None:
        """Select *value*."""
        combo = self.widget()
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def set_config_hint(self, text: str) -> None:
        """Show the configured value in the label of the ``None`` choice."""
        combo = self.widget()
        for i, (value, label) in enumerate(self.choices):
            if value is None:
                combo.setItemText(i, f"{label} ({text})")

    def code(self) -> str:
        """The literal of the chosen value."""
        return repr(self.value())


class TextParam(Param):
    """Free text, rendered as a string literal."""

    def __init__(self, name, label, default="", placeholder="", **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.default, self.placeholder = default, placeholder

    def _make_widget(self) -> QWidget:
        edit = QLineEdit(self.default)
        edit.setPlaceholderText(self.placeholder or ("None" if self.optional else ""))
        edit.textChanged.connect(lambda *_: self.changed.emit())
        return edit

    def value(self) -> Optional[str]:
        """The text (``None`` when empty and optional)."""
        text = self.widget().text().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("a value is required")
        return text

    def set_value(self, value) -> None:
        """Set the text."""
        self.widget().setText("" if value is None else str(value))

    def code(self) -> str:
        """The string literal (or ``None``)."""
        return repr(self.value())


class ExprParam(TextParam):
    """
    A Python expression evaluated in the kernel (e.g. ``np.arange(10)``).

    The expression is inserted verbatim; its syntax is checked before running.
    """

    def value(self) -> Optional[str]:
        """The expression source (``None`` when empty and optional)."""
        text = self.widget().text().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("an expression is required")
        try:
            compile(text, self.name, "eval")
        except SyntaxError as exc:
            raise self._error(f"invalid expression ({exc.msg})") from None
        return text

    def code(self) -> str:
        """The expression (or ``None``)."""
        text = self.value()
        return "None" if text is None else text


class VarParam(TextParam):
    """Name of the kernel variable receiving a result."""

    def value(self) -> str:
        """The variable name."""
        text = self.widget().text().strip()
        if not text.isidentifier() or keyword.iskeyword(text):
            raise self._error(f"'{text}' is not a valid variable name")
        return text

    def code(self) -> str:
        """The variable name (not quoted)."""
        return self.value()


class DeviceParam(Param):
    """
    A device connected in the kernel, chosen among the workspace variables.

    Parameters
    ----------
    kinds : sequence of str
        Accepted workspace kinds (``'dm'``, ``'interferometer'``, ``'wfs'``,
        ``'camera'``, ``'other'``...).
    default : str, optional
        Preferred variable name (e.g. ``'dm'``).
    """

    def __init__(self, name, label, kinds: Sequence[str], default: str = "", **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.kinds = tuple(kinds)
        self.default = default

    def _make_widget(self) -> QWidget:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setMinimumContentsLength(24)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.lineEdit().setPlaceholderText("None" if self.optional else "connect a device first")
        combo.currentTextChanged.connect(lambda *_: self.changed.emit())
        return combo

    def _setup(self) -> None:
        if self.context is not None:
            self.context.workspace_changed.connect(self._refresh)
            self._refresh(self.context.workspace)

    def _refresh(self, items: List[Dict[str, str]]) -> None:
        combo = self.widget()
        current = combo.currentText().strip()
        names = [i["name"] for i in items if i.get("kind") in self.kinds]
        combo.blockSignals(True)
        combo.clear()
        for item in items:
            if item.get("kind") in self.kinds:
                combo.addItem(f"{item['name']}", item["name"])
                combo.setItemData(
                    combo.count() - 1, f"{item['name']}: {item.get('type', '')}", Qt.ItemDataRole.ToolTipRole
                )
        choice = current if current else (self.default if self.default in names else (names[0] if names else ""))
        combo.setEditText(choice)
        combo.blockSignals(False)
        self.changed.emit()

    def value(self) -> Optional[str]:
        """The variable name (``None`` when empty and optional)."""
        text = self.widget().currentText().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("choose a device (connect it in the Devices panel)")
        if not text.isidentifier():
            raise self._error(f"'{text}' is not a variable name")
        return text

    def set_value(self, value) -> None:
        """Select the device variable *value*."""
        self.widget().setEditText("" if value is None else str(value))

    def code(self) -> str:
        """The variable name (or ``None``)."""
        name = self.value()
        return "None" if name is None else name


class DeviceListParam(TextParam):
    """Comma-separated device variables, rendered as a Python list."""

    def value(self) -> List[str]:
        """The variable names."""
        names = [n.strip() for n in self.widget().text().split(",") if n.strip()]
        if not names and not self.optional:
            raise self._error("at least one device is required")
        for name in names:
            if not name.isidentifier():
                raise self._error(f"'{name}' is not a variable name")
        return names

    def code(self) -> str:
        """``[a, b]``."""
        return "[" + ", ".join(self.value()) + "]"


class TnParam(Param):
    """
    A tracking number, typed or chosen among the folders of a data category.

    Parameters
    ----------
    folder_attr : str, optional
        Attribute of :mod:`opticalib.core.root` whose folder lists the
        tracking numbers (e.g. ``'IFFUNCTIONS_ROOT_FOLDER'``).
    state_key : str, optional
        Procedure state entry that fills the value when a previous step
        produced it (e.g. ``'iff_tn'``).
    """

    def __init__(self, name, label, folder_attr: str = "", state_key: str = "", **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.folder_attr = folder_attr
        self.state_key = state_key

    def _make_widget(self) -> QWidget:
        box = QWidget()
        self._combo = QComboBox()
        self._combo.setEditable(True)
        self._combo.lineEdit().setPlaceholderText("None" if self.optional else "YYYYMMDD_HHMMSS")
        self._combo.currentTextChanged.connect(lambda *_: self.changed.emit())
        refresh = QToolButton()
        refresh.setIcon(theme().icon("refresh"))
        refresh.setToolTip("List the tracking numbers on disk")
        refresh.clicked.connect(self.refresh)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._combo, 1)
        row.addWidget(refresh)
        return box

    def _setup(self) -> None:
        if self.context is not None:
            self.context.folders_changed.connect(self._on_folders_changed)
        self.refresh()

    def _on_folders_changed(self, info) -> None:
        self.refresh()

    def folder(self) -> str:
        """The folder listing the tracking numbers (empty if unknown)."""
        if self.context is None or not self.folder_attr:
            return ""
        return self.context.folder(self.folder_attr)

    def refresh(self) -> None:
        """List the tracking numbers of :meth:`folder`, newest first."""
        folder = self.folder()
        if not folder or not os.path.isdir(folder):
            return
        try:
            tns = sorted((e.name for e in os.scandir(folder) if e.is_dir()), reverse=True)
        except OSError:
            return
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(tns[:200])
        self._combo.setEditText(current)
        self._combo.blockSignals(False)

    def value(self) -> Optional[str]:
        """The tracking number (``None`` when empty and optional)."""
        self.widget()
        text = self._combo.currentText().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("a tracking number is required")
        return text

    def set_value(self, value) -> None:
        """Set the tracking number."""
        self.widget()
        self._combo.setEditText("" if value is None else str(value))

    def code(self) -> str:
        """The string literal (or ``None``)."""
        tn = self.value()
        return "None" if tn is None else repr(tn)


class TnListParam(Param):
    """
    Several tracking numbers, typed (comma or space separated) or picked
    from the folders of a data category; rendered as a list of strings.

    Parameters
    ----------
    folder_attr : str, optional
        Attribute of :mod:`opticalib.core.root` whose folder lists the
        tracking numbers offered by the *Add* menu.
    minimum : int, optional
        Minimum number of tracking numbers.
    """

    def __init__(self, name, label, folder_attr: str = "", minimum: int = 1, **kwargs) -> None:
        """Create the parameter."""
        super().__init__(name, label, **kwargs)
        self.folder_attr = folder_attr
        self.minimum = minimum

    def _make_widget(self) -> QWidget:
        box = QWidget()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("YYYYMMDD_HHMMSS, YYYYMMDD_HHMMSS, …")
        self._edit.textChanged.connect(lambda *_: self.changed.emit())
        self._add = QToolButton()
        self._add.setIcon(theme().icon("plus"))
        self._add.setToolTip("Add a tracking number from disk")
        self._add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add.setMenu(QMenu(self._add))
        self._add.menu().aboutToShow.connect(self._fill_menu)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._edit, 1)
        row.addWidget(self._add)
        return box

    def _hint_edit(self):
        self.widget()
        return self._edit

    def available(self) -> List[str]:
        """Tracking numbers found on disk, newest first."""
        folder = self.context.folder(self.folder_attr) if self.context is not None and self.folder_attr else ""
        if not folder or not os.path.isdir(folder):
            return []
        try:
            return sorted((e.name for e in os.scandir(folder) if e.is_dir()), reverse=True)
        except OSError:
            return []

    def _fill_menu(self) -> None:
        menu = self._add.menu()
        menu.clear()
        tns = self.available()
        if not tns:
            menu.addAction("No tracking numbers found").setEnabled(False)
        for tn in tns[:50]:
            menu.addAction(tn).triggered.connect(lambda checked=False, t=tn: self.append(t))

    def append(self, tn: str) -> None:
        """Add *tn* at the end of the list."""
        current = self._edit.text().strip().rstrip(",")
        self._edit.setText(f"{current}, {tn}" if current else tn)

    def value(self) -> List[str]:
        """The tracking numbers."""
        self.widget()
        tns = [t for t in self._edit.text().replace(",", " ").split() if t]
        if len(tns) < self.minimum:
            raise self._error(f"at least {self.minimum} tracking number(s) required")
        return tns

    def set_value(self, value) -> None:
        """Set the tracking numbers (a list or a comma-separated string)."""
        self.widget()
        if isinstance(value, (list, tuple)):
            value = ", ".join(map(str, value))
        self._edit.setText("" if value is None else str(value))

    def code(self) -> str:
        """``['tn1', 'tn2']``."""
        return repr(self.value())
