"""
Procedure windows: common framework
===================================

A procedure window guides one bench procedure as a sequence of
:class:`Step` objects (e.g. *acquire*, *process*, *analyse*).  Each step is
declarative: parameters (:mod:`~opticalib.gui.procedures.params`), a code
template, the kernel variables it produces and, for steps that move
hardware, a confirmation message.

Like every GUI action, running a step executes **visible Python code** in
the console, so the procedure stays reproducible and scriptable: the code
preview shows exactly what will run, and *Copy code* gives it to you.

Results (e.g. the tracking number of an acquisition) are read back from the
kernel into the procedure *state*, which pre-fills the parameters of the
next steps.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QFontDatabase, QGuiApplication
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..activity import format_elapsed
from ..theme import theme
from .params import Param, ParamError, TnParam


class ProcedureContext(QObject):
    """
    What procedure windows need from the application.

    Parameters
    ----------
    run : callable
        ``run(code, title, on_done, on_error) -> Task`` executing code in the
        console (see :meth:`~opticalib.gui.kernel.KernelBridge.run`).
    query : callable
        ``query(expressions, callback)`` evaluating expressions silently.
    interrupt : callable
        Interrupts the running code.
    view_file : callable, optional
        ``view_file(path)`` previews a data file in the plot viewer.
    parent : QObject, optional
        Parent object.

    Signals
    -------
    workspace_changed(list)
        The kernel variables changed (see
        :func:`~opticalib.gui.kernel_side.workspace_items`).
    folders_changed(dict)
        The data folders changed (see :func:`~opticalib.gui.kernel_side.folders`).
    """

    workspace_changed = Signal(list)
    folders_changed = Signal(dict)

    def __init__(self, run, query, interrupt, view_file=None, parent=None) -> None:
        """Store the application callbacks."""
        super().__init__(parent)
        self.run = run
        self.query = query
        self.interrupt = interrupt
        self.view_file = view_file
        self.workspace: List[Dict[str, str]] = []
        self.folders: Dict[str, Any] = {}

    def set_workspace(self, items: List[Dict[str, str]]) -> None:
        """Update the kernel variables (called by the application)."""
        self.workspace = list(items)
        self.workspace_changed.emit(self.workspace)

    def set_folders(self, info: Dict[str, Any]) -> None:
        """Update the data folders (called by the application)."""
        self.folders = dict(info)
        self.folders_changed.emit(self.folders)

    def folder(self, attr: str) -> str:
        """
        Return a data folder by its :mod:`opticalib.core.root` attribute name.

        Parameters
        ----------
        attr : str
            E.g. ``'IFFUNCTIONS_ROOT_FOLDER'``.

        Returns
        -------
        str
            The folder path (empty if unknown).
        """
        return self.folders.get("paths", {}).get(attr, "")


def kwargs_code(values: Dict[str, str], *names: str) -> str:
    """
    Render ``name=value`` pairs, leaving out optional values set to ``None``.

    Leaving them out lets the library defaults (often read from the
    configuration file) apply.

    Parameters
    ----------
    values : dict
        Parameter sources, as passed to :attr:`Step.template`.
    *names : str
        Parameters to render, in order.

    Returns
    -------
    str
        E.g. ``"nframes=5, delay=1.0"``.
    """
    return ", ".join(f"{name}={values[name]}" for name in names if values[name] != "None")


@dataclass
class Step:
    """
    One operation of a procedure.

    Attributes
    ----------
    key : str
        Identifier.
    title : str
        Short name shown in the step list.
    description : str
        What the step does (shown above its parameters).
    params : list of Param
        The parameters.
    template : callable
        ``template(values) -> str`` returning the code, where *values* maps
        each parameter name to its Python source.
    outputs : dict
        Procedure state entries produced by the step: ``{state_key:
        expression}``, evaluated in the kernel after the step succeeded.
        Expressions are formatted with the parameter sources, e.g.
        ``{'iff_tn': '{out}'}`` reads the variable named by parameter ``out``.
    confirm : str, optional
        Confirmation message shown before running (steps moving hardware).
    after : callable, optional
        ``after(window, results, values)`` called after the outputs were
        read, with the parameter sources of the run.
    section : str, optional
        Heading of the group of steps this step belongs to (e.g.
        ``'Post-processing'``); consecutive steps with the same section are
        listed together under it.
    """

    key: str
    title: str
    description: str
    params: List[Param]
    template: Callable[[Dict[str, str]], str]
    outputs: Dict[str, str] = field(default_factory=dict)
    confirm: Optional[str] = None
    after: Optional[Callable[["ProcedureWindow", Dict[str, Any], Dict[str, str]], None]] = None
    section: str = ""

    def param(self, name: str) -> Param:
        """Return the parameter called *name*."""
        return next(p for p in self.params if p.name == name)

    def values(self) -> Dict[str, str]:
        """
        Return the Python source of every parameter.

        Raises
        ------
        ParamError
            If a parameter is invalid.
        """
        return {p.name: p.code() for p in self.params}

    def code(self) -> str:
        """
        Return the code of the step with the current parameter values.

        Raises
        ------
        ParamError
            If a parameter is invalid.
        """
        return self.template(self.values())


class ProcedureWindow(QMainWindow):
    """
    Window guiding a procedure through its :class:`Step` objects.

    Subclasses define :attr:`TITLE`, :attr:`DESCRIPTION` and :meth:`steps`.

    Parameters
    ----------
    context : ProcedureContext, optional
        Access to the kernel; without it the window only previews code.
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    closed(object)
        The window was closed (the window is passed); its owner deletes it.
    state_changed(dict)
        The procedure state changed.
    """

    closed = Signal(object)
    state_changed = Signal(dict)

    TITLE = "Procedure"
    DESCRIPTION = ""
    #: Pre-flight checks, as ``(expression, warning)``: when the expression
    #: evaluates to ``False`` in the kernel, the warning is shown on top.
    CHECKS: List[tuple] = []

    def __init__(self, context: Optional[ProcedureContext] = None, parent: Optional[QWidget] = None) -> None:
        """Build the window."""
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(1100, 720)
        self.context = context
        self.state: Dict[str, Any] = {}
        self._steps: List[Step] = self.steps()
        self._forms: Dict[str, QWidget] = {}
        self._status: Dict[str, str] = {s.key: "idle" for s in self._steps}
        self._config_hints: Dict[tuple, str] = {}
        self._task = None
        self._task_step: Optional[Step] = None
        self._build_ui()
        theme().changed.connect(self._refresh_step_list)
        self.select_step(self._steps[0].key if self._steps else "")
        self.run_checks()
        self.refresh_config_defaults()
        if context is not None:
            # The configuration may have been reloaded: read it again.
            context.folders_changed.connect(self._on_config_reloaded)

    # ------------------------------------------------------------------
    # To be defined by subclasses
    # ------------------------------------------------------------------

    def steps(self) -> List[Step]:
        """Return the steps of the procedure."""
        return []

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        heading = QLabel(self.TITLE)
        heading.setProperty("title", True)
        subtitle = QLabel(self.DESCRIPTION)
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        self._warnings = QLabel()
        self._warnings.setWordWrap(True)
        self._warnings.hide()

        self._step_list = QListWidget()
        self._step_list.setMinimumWidth(220)
        section = None
        for step in self._steps:
            if step.section and step.section != section:
                header = QListWidgetItem(step.section.upper())
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                font = header.font()
                font.setBold(True)
                font.setPointSizeF(font.pointSizeF() * 0.85)
                header.setFont(font)
                self._step_list.addItem(header)
            section = step.section
            item = QListWidgetItem(step.title)
            item.setData(Qt.ItemDataRole.UserRole, step.key)
            self._step_list.addItem(item)
        self._step_list.currentRowChanged.connect(self._show_step)

        self._step_title = QLabel()
        self._step_title.setProperty("heading", True)
        self._step_description = QLabel()
        self._step_description.setWordWrap(True)
        self._step_description.setProperty("muted", True)
        self._form_area = QScrollArea()
        self._form_area.setWidgetResizable(True)
        self._form_area.setFrameShape(QFrame.Shape.NoFrame)

        code_label = QLabel("CODE")
        code_label.setProperty("section", True)
        self._code = QPlainTextEdit()
        self._code.setReadOnly(True)
        self._code.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._code.setMinimumHeight(60)
        self._error = QLabel()
        self._error.setWordWrap(True)

        self._btn_copy = QPushButton("Copy code")
        self._btn_copy.clicked.connect(self._copy_code)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("Interrupt the running step (like Ctrl+C in the console)")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_run = QPushButton("Run step")
        self._btn_run.setProperty("accent", True)
        self._btn_run.clicked.connect(self.run_current)
        buttons = QHBoxLayout()
        buttons.addWidget(self._btn_copy)
        buttons.addStretch()
        buttons.addWidget(self._btn_stop)
        buttons.addWidget(self._btn_run)

        log_label = QLabel("RESULTS")
        log_label.setProperty("section", True)
        self._log = QListWidget()
        self._log.setMinimumHeight(50)

        # Parameters, code and results in a vertical splitter: the form gets
        # most of the room, and the user can resize the parts.
        form_part = QWidget()
        form_column = QVBoxLayout(form_part)
        form_column.setContentsMargins(0, 0, 0, 0)
        form_column.addWidget(self._step_title)
        form_column.addWidget(self._step_description)
        form_column.addWidget(self._form_area, 1)
        code_part = QWidget()
        code_column = QVBoxLayout(code_part)
        code_column.setContentsMargins(0, 6, 0, 0)
        code_column.addWidget(code_label)
        code_column.addWidget(self._code, 1)
        code_column.addWidget(self._error)
        code_column.addLayout(buttons)
        log_part = QWidget()
        log_column = QVBoxLayout(log_part)
        log_column.setContentsMargins(0, 6, 0, 0)
        log_column.addWidget(log_label)
        log_column.addWidget(self._log, 1)
        self._parts = QSplitter(Qt.Orientation.Vertical)
        self._parts.setChildrenCollapsible(False)
        for part, stretch in ((form_part, 5), (code_part, 2), (log_part, 1)):
            self._parts.addWidget(part)
            self._parts.setStretchFactor(self._parts.count() - 1, stretch)
        self._parts.setSizes([430, 170, 110])

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(12, 0, 0, 0)
        column.addWidget(self._parts)

        splitter = QSplitter()
        splitter.addWidget(self._step_list)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 860])

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.addWidget(heading)
        root.addWidget(subtitle)
        root.addWidget(self._warnings)
        root.addSpacing(8)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._refresh_step_list()
        self._refresh_buttons()

    def _form_for(self, step: Step) -> QWidget:
        if step.key not in self._forms:
            form_widget = QWidget()
            form = QFormLayout(form_widget)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            for param in step.params:
                widget = param.widget(self.context)
                form.addRow(param.label, widget)
                param.changed.connect(self._refresh_code)
            if not step.params:
                note = QLabel("This step has no parameters.")
                note.setProperty("muted", True)
                form.addRow(note)
            self._forms[step.key] = form_widget
            self._apply_state(step)
            self._apply_config_hints(step)
        return self._forms[step.key]

    def current_step(self) -> Optional[Step]:
        """The step shown in the window."""
        item = self._step_list.currentItem()
        key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return next((s for s in self._steps if s.key == key), None)

    def _step_items(self):
        """(item, step) pairs of the step list, headers excluded."""
        for row in range(self._step_list.count()):
            item = self._step_list.item(row)
            key = item.data(Qt.ItemDataRole.UserRole)
            if key is not None:
                yield item, self.step(key)

    def step(self, key: str) -> Step:
        """Return the step called *key*."""
        return next(s for s in self._steps if s.key == key)

    def select_step(self, key: str) -> None:
        """Show the step called *key*."""
        for item, step in self._step_items():
            if step.key == key:
                self._step_list.setCurrentItem(item)
                return

    def _show_step(self, row: int) -> None:
        step = self.current_step()
        if step is None:
            return
        self._step_title.setText(step.title)
        self._step_description.setText(step.description)
        self._form_area.takeWidget()
        self._form_area.setWidget(self._form_for(step))
        self._refresh_code()
        self._refresh_buttons()

    def _refresh_code(self) -> None:
        step = self.current_step()
        if step is None:
            return
        try:
            code = step.code()
        except ParamError as exc:
            self._code.setPlainText("")
            self._error.setText(f"<span style='color:{theme().tokens['warning']}'>{exc}</span>")
            self._btn_run.setEnabled(False)
            return
        self._code.setPlainText(code)
        self._error.setText("")
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        running = self._task is not None and not self._task.is_final
        valid = bool(self._code.toPlainText().strip())
        self._btn_run.setEnabled(valid and not running and self.context is not None)
        self._btn_stop.setEnabled(running)
        self._btn_copy.setEnabled(valid)

    def _refresh_step_list(self) -> None:
        t = theme()
        icons = {
            "idle": ("circle-outline", "text_muted"),
            "running": ("progress-clock", "accent"),
            "done": ("check-circle", "success"),
            "error": ("alert-circle", "danger"),
        }
        for item, step in self._step_items():
            name, token = icons[self._status[step.key]]
            item.setIcon(t.icon(name, token))
        for row in range(self._step_list.count()):
            item = self._step_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) is None:
                item.setForeground(t.color("text_muted"))

    def refresh_config_defaults(self) -> None:
        """
        Read the defaults of the parameters from the kernel (configuration).

        Every parameter with a ``config_default`` expression shows the value
        it would get when left empty (e.g. ``0.05``) as placeholder text.
        """
        if self.context is None:
            return
        from ..kernel import KERNEL_SIDE

        expressions = {}
        for step in self._steps:
            for param in step.params:
                if param.config_default:
                    key = f"{step.key}__{param.name}"
                    # JSON-encoded, so the reply stays a string (the bridge would
                    # otherwise decode e.g. "0.05" into a float).
                    expressions[key] = f"__import__('json').dumps({KERNEL_SIDE}.describe({param.config_default}))"
        if expressions:
            self.context.query(expressions, self._on_config_defaults)

    def _on_config_defaults(self, results: Dict[str, Any]) -> None:
        for key, text in results.items():
            if isinstance(text, str):
                step_key, name = key.split("__", 1)
                self._config_hints[(step_key, name)] = text
        for step in self._steps:
            if step.key in self._forms:
                self._apply_config_hints(step)

    def _apply_config_hints(self, step: Step) -> None:
        for param in step.params:
            hint = self._config_hints.get((step.key, param.name))
            if hint is not None:
                param.set_config_hint(hint)

    def _on_config_reloaded(self, info: Dict[str, Any]) -> None:
        self.run_checks()
        self.refresh_config_defaults()

    def run_checks(self) -> None:
        """Evaluate :attr:`CHECKS` in the kernel and show the failed ones."""
        if self.context is None or not self.CHECKS:
            return
        expressions = {f"check{i}": expr for i, (expr, _) in enumerate(self.CHECKS)}
        self.context.query(expressions, self._on_checks)

    def _on_checks(self, results: Dict[str, Any]) -> None:
        failed = [
            message
            for i, (_, message) in enumerate(self.CHECKS)
            if results.get(f"check{i}") is False
        ]
        self.show_warnings(failed)

    def show_warnings(self, messages: List[str]) -> None:
        """
        Show warnings above the steps (hidden when *messages* is empty).

        Parameters
        ----------
        messages : list of str
            The warnings.
        """
        color = theme().tokens["warning"]
        self._warnings.setText("<br>".join(f"<span style='color:{color}'>⚠ {m}</span>" for m in messages))
        self._warnings.setVisible(bool(messages))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        """
        Store a procedure result and pre-fill the parameters that use it.

        Parameters
        ----------
        key : str
            State entry (e.g. ``'iff_tn'``).
        value : Any
            Its value.
        """
        self.state[key] = value
        for step in self._steps:
            if step.key in self._forms:
                self._apply_state(step, only=key)
        self.state_changed.emit(dict(self.state))

    def _apply_state(self, step: Step, only: Optional[str] = None) -> None:
        for param in step.params:
            key = getattr(param, "state_key", "")
            if key and key in self.state and (only is None or key == only):
                if isinstance(param, TnParam):
                    param.refresh()
                param.set_value(self.state[key])

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run_current(self) -> None:
        """Run the step shown, after confirmation if it moves hardware."""
        step = self.current_step()
        if step is not None:
            self.run_step(step)

    def run_step(self, step: Step, confirmed: bool = False):
        """
        Run *step* in the console.

        Parameters
        ----------
        step : Step
            The step.
        confirmed : bool, optional
            Skip the confirmation of hardware-moving steps.

        Returns
        -------
        Task or None
            The task, or ``None`` if the step was not run.
        """
        if self.context is None or (self._task is not None and not self._task.is_final):
            return None
        try:
            values = step.values()
            code = step.template(values)
        except ParamError as exc:
            QMessageBox.warning(self, step.title, str(exc))
            return None
        outputs = {key: expr.format(**values) for key, expr in step.outputs.items()}
        if step.confirm and not confirmed:
            answer = QMessageBox.question(
                self,
                step.title,
                f"{step.confirm}\n\nCode to run:\n\n{code}\n\nContinue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        self._task_step = step
        self._set_status(step, "running")
        started = time.monotonic()
        task = self.context.run(
            code,
            f"{self.TITLE}: {step.title}",
            lambda t, s=step, t0=started: self._on_step_done(s, t0, outputs, values),
            lambda t, s=step, t0=started: self._on_step_failed(s, t, t0),
        )
        self._task = task
        self._refresh_buttons()
        return task

    def _on_step_done(self, step: Step, started: float, outputs: Dict[str, str], values: Dict[str, str]) -> None:
        elapsed = format_elapsed(time.monotonic() - started)
        if not outputs:
            self._finish_step(step, {}, elapsed, values)
            return
        self.context.query(
            {key: f"repr({expr})" for key, expr in outputs.items()},
            lambda results, s=step, e=elapsed, v=values: self._finish_step(s, results, e, v),
        )

    def _finish_step(self, step: Step, results: Dict[str, Any], elapsed: str, values: Dict[str, str]) -> None:
        import ast

        state_values: Dict[str, Any] = {}
        for key, text in results.items():
            if isinstance(text, Exception):
                continue
            try:
                state_values[key] = ast.literal_eval(text) if isinstance(text, str) else text
            except (ValueError, SyntaxError):
                state_values[key] = text
        for key, value in state_values.items():
            self.set_state(key, value)
        shown = ", ".join(f"{k} = {v!r}" for k, v in state_values.items())
        self._log_line("check-circle", "success", f"{step.title} ({elapsed}){': ' + shown if shown else ''}")
        self._set_status(step, "done")
        self._task = None
        self._refresh_buttons()
        if step.after is not None:
            step.after(self, state_values, values)

    def _on_step_failed(self, step: Step, task, started: float) -> None:
        error = task.error or {}
        elapsed = format_elapsed(time.monotonic() - started)
        self._log_line(
            "alert-circle", "danger",
            f"{step.title} failed ({elapsed}): {error.get('ename', 'Error')}: {error.get('evalue', '')}",
        )
        self._set_status(step, "error")
        self._task = None
        self._refresh_buttons()

    def _set_status(self, step: Step, status: str) -> None:
        self._status[step.key] = status
        self._refresh_step_list()

    def _log_line(self, icon: str, token: str, text: str) -> None:
        item = QListWidgetItem(theme().icon(icon, token), text)
        item.setToolTip(text)
        self._log.insertItem(0, item)

    def _stop(self) -> None:
        if self._task is not None and not self._task.is_final and self.context is not None:
            self.context.interrupt()

    def _copy_code(self) -> None:
        QGuiApplication.clipboard().setText(self._code.toPlainText())

    # Qt event handler: the camelCase name is required for Qt to call it.
    def closeEvent(self, event) -> None:  # noqa: N802
        """Notify the owner, which deletes the window."""
        super().closeEvent(event)
        self.closed.emit(self)
