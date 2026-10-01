"""
Activity feedback for CalpyGUI
==============================

Non-blocking feedback for long operations: nothing in CalpyGUI freezes the
window, so the user needs to see what is running instead.

* :class:`ActivityCenter` - a floating panel with one card per running or
  queued operation, with a spinner or a progress bar, the elapsed time, the
  latest output line and an *Interrupt* / *Cancel* button.  Finished
  operations turn into a short-lived success toast, or an error card with
  the traceback one click away.  The panel sits in the top-left corner of
  the plot area by default; it can be dragged anywhere by its header, pins
  to the nearest window corner, and can be locked or collapsed.
* :class:`StartupOverlay` - a translucent overlay listing the startup steps
  while the kernel boots and the calpy environment loads.

Both work with :class:`~opticalib.gui.kernel.Task` objects (kernel commands)
and with :class:`LocalJob` objects (work done in a GUI background thread).
"""

import time
from typing import Callable, Dict, List, Optional

from qtpy.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSettings,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtGui import QAction, QColor, QFont, QPainter
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .kernel import strip_ansi
from .theme import SETTINGS_APP, SETTINGS_ORG, repolish, theme
from .widgets.common import ElidedLabel

#: Seconds a success toast stays visible.
TOAST_SECONDS = 4.0
#: Maximum number of cards shown at once.
MAX_CARDS = 4


def format_elapsed(seconds: float) -> str:
    """
    Format a duration as ``m:ss`` (or ``h:mm:ss``).

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        The formatted duration.
    """
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class LocalJob(QObject):
    """
    A function run in a GUI background thread, tracked like a kernel task.

    Parameters
    ----------
    fn : callable
        Function to run; it must not touch Qt widgets.
    title : str
        Description shown in the activity panel.
    on_done : callable, optional
        Called in the GUI thread with the job when *fn* returned; the result
        is in :attr:`result`.
    on_error : callable, optional
        Called in the GUI thread with the job when *fn* raised.

    Signals
    -------
    changed, finished
        Same meaning as for :class:`~opticalib.gui.kernel.Task`.
    """

    changed = Signal()
    finished = Signal()
    _completed = Signal(object, object)

    def __init__(
        self,
        fn: Callable[[], object],
        title: str,
        on_done: Optional[Callable[["LocalJob"], None]] = None,
        on_error: Optional[Callable[["LocalJob"], None]] = None,
    ) -> None:
        """Create the job; start it with :meth:`start`."""
        super().__init__()
        self.title = title
        self.quiet = False
        self.state = "queued"
        self.progress: Optional[float] = None
        self.detail = ""
        self.error: Optional[Dict[str, object]] = None
        self.result: object = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._fn = fn
        self._on_done = on_done
        self._on_error = on_error
        self._completed.connect(self._complete)

    @property
    def elapsed(self) -> float:
        """Seconds spent running."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def is_final(self) -> bool:
        """Whether the job has finished."""
        return self.state in ("done", "error", "cancelled")

    def start(self) -> "LocalJob":
        """
        Run the job in the global thread pool.

        Returns
        -------
        LocalJob
            The job itself.
        """
        self.state = "running"
        self.started_at = time.monotonic()
        self.changed.emit()
        job = self

        class _Runner(QRunnable):
            def run(self) -> None:
                try:
                    result = job._fn()
                except Exception as exc:  # reported in the GUI thread
                    job._completed.emit(None, exc)
                else:
                    job._completed.emit(result, None)

        QThreadPool.globalInstance().start(_Runner())
        return self

    def _complete(self, result: object, exc: Optional[BaseException]) -> None:
        self.finished_at = time.monotonic()
        if exc is None:
            self.result = result
            self.state = "done"
            self.progress = 1.0
        else:
            self.state = "error"
            self.error = {
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": [],
            }
        self.changed.emit()
        self.finished.emit()
        callback = self._on_done if self.state == "done" else self._on_error
        if callback is not None:
            callback(self)


class TracebackDialog(QDialog):
    """
    Dialog showing the error of a failed operation.

    Parameters
    ----------
    title : str
        The operation that failed.
    error : dict
        ``{'ename', 'evalue', 'traceback'}``.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, title: str, error: Dict[str, object], parent=None) -> None:
        """Build the dialog."""
        super().__init__(parent)
        self.setWindowTitle(f"Error – {title}")
        self.resize(760, 420)
        layout = QVBoxLayout(self)
        heading = QLabel(f"{error.get('ename', 'Error')}: {error.get('evalue', '')}")
        heading.setWordWrap(True)
        heading.setProperty("heading", True)
        layout.addWidget(heading)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Monospace", 9))
        traceback = error.get("traceback") or []
        text.setPlainText(strip_ansi("\n".join(traceback)) or "(no traceback)")
        layout.addWidget(text)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        layout.addLayout(row)


class ActivityCard(QFrame):
    """
    Floating card showing one operation.

    Parameters
    ----------
    job : Task or LocalJob
        The operation.
    interrupt : callable, optional
        Called to stop the running operation (kernel interrupt).
    cancel : callable, optional
        Called with the job to remove it from the queue.
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    dismissed
        The card should be removed.
    """

    dismissed = Signal()

    def __init__(self, job, interrupt=None, cancel=None, parent=None) -> None:
        """Build the card for *job*."""
        super().__init__(parent)
        self.job = job
        self._interrupt = interrupt
        self._cancel = cancel
        self.setProperty("floating", True)
        self.setFixedWidth(340)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 70))
        self.setGraphicsEffect(shadow)

        import qtawesome as qta

        self._icon = qta.IconWidget()
        self._icon.setIconSize(QSize(20, 20))
        self._spin = None
        self._title = ElidedLabel(job.title)
        self._title.setProperty("heading", True)
        self._title.setToolTip(job.title)
        self._elapsed = QLabel("")
        self._elapsed.setProperty("muted", True)
        self._detail = QLabel("")
        self._detail.setProperty("muted", True)
        self._detail.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._action = QPushButton()
        self._action.clicked.connect(self._on_action)
        self._close = QToolButton()
        self._close.setToolTip("Dismiss")
        self._close.clicked.connect(self.dismissed.emit)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        top.addWidget(self._title, 1)
        top.addWidget(self._elapsed, 0, Qt.AlignmentFlag.AlignTop)
        top.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)
        bottom = QHBoxLayout()
        bottom.addWidget(self._detail, 1)
        bottom.addWidget(self._action, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self._bar)
        layout.addLayout(bottom)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_elapsed)
        self._timer.start()
        job.changed.connect(self.refresh)
        theme().changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Update the card from the job state."""
        import qtawesome as qta

        job = self.job
        state = job.state
        self.setProperty("state", state)
        repolish(self)
        self._close.setIcon(theme().icon("close", "text_muted"))
        self._close.setVisible(job.is_final)
        if state == "queued":
            self._icon.setIcon(theme().icon("timer-sand", "text_muted"))
            self._bar.setRange(0, 0)
            self._bar.setVisible(False)
            self._detail.setText("Waiting for the kernel…")
            self._set_action("Cancel", self._cancel is not None)
        elif state == "running":
            if self._spin is None:
                self._spin = qta.Spin(self._icon, autostart=True)
                self._icon.setIcon(theme().icon("loading", "accent", animation=self._spin))
            self._bar.setVisible(True)
            if job.progress is None:
                self._bar.setRange(0, 0)
            else:
                self._bar.setRange(0, 1000)
                self._bar.setValue(int(job.progress * 1000))
            self._detail.setText(job.detail or "Running…")
            self._set_action("Interrupt", self._interrupt is not None)
        elif state == "done":
            self._stop_spinner("check-circle", "success")
            self._bar.setVisible(False)
            self._detail.setText("Completed")
            self._set_action("", False)
        else:
            self._stop_spinner("alert-circle", "danger")
            self._bar.setVisible(False)
            error = job.error or {}
            if state == "cancelled":
                self._detail.setText("Cancelled")
                self._set_action("", False)
            else:
                self._detail.setText(
                    f"{error.get('ename', 'Error')}: {error.get('evalue', '')}"[:160]
                )
                self._set_action("Details", True)
        self._refresh_elapsed()

    def _stop_spinner(self, icon: str, token: str) -> None:
        # Stop the animation timer, otherwise it keeps repainting the icon.
        if self._spin is not None:
            self._spin.stop()
            self._spin = None
        self._icon.setIcon(theme().icon(icon, token))

    def _set_action(self, text: str, visible: bool) -> None:
        self._action.setText(text)
        self._action.setVisible(visible and bool(text))

    def _refresh_elapsed(self) -> None:
        job = self.job
        if job.state == "queued":
            self._elapsed.setText("queued")
        else:
            self._elapsed.setText(format_elapsed(job.elapsed))
        if job.is_final:
            self._timer.stop()

    def _on_action(self) -> None:
        job = self.job
        if job.state == "queued" and self._cancel is not None:
            self._cancel(job)
        elif job.state == "running" and self._interrupt is not None:
            self._interrupt()
        elif job.state == "error":
            dialog = TracebackDialog(job.title, job.error or {}, self.window())
            dialog.exec()
            dialog.deleteLater()


class _DragHeader(QFrame):
    """
    Header of the activity panel; dragging it moves the panel.

    Signals
    -------
    drag_started(QPoint), drag_moved(QPoint), drag_finished()
        Mouse positions are global coordinates.
    """

    drag_started = Signal(QPoint)
    drag_moved = Signal(QPoint)
    drag_finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the header."""
        super().__init__(parent)
        self.draggable = True
        self._dragging = False

    # Qt event handlers: the camelCase names are required for Qt to call them.
    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Start dragging the panel."""
        if self.draggable and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Move the panel with the mouse."""
        if self._dragging:
            self.drag_moved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Drop the panel and pin it to the nearest corner."""
        if self._dragging:
            self._dragging = False
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self.draggable else Qt.CursorShape.ArrowCursor
            )
            self.drag_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ActivityCenter(QWidget):
    """
    Floating, movable panel stacking one :class:`ActivityCard` per operation.

    The panel appears while operations are running or recently finished.  By
    default it sits in the top-left corner of the plot area; it can be
    dragged anywhere in the window by its header, and is then pinned to the
    nearest window corner (keeping its distance from it when the window or
    the panels are resized).  The header also offers *pin* (lock the
    position), *collapse* (show only a summary) and *clear finished*.  The
    position is remembered between sessions.

    Parameters
    ----------
    parent : QMainWindow
        The main window.
    interrupt : callable, optional
        Function interrupting the kernel.
    cancel : callable, optional
        Function removing a queued task from the kernel queue.
    """

    #: Corners the panel can be pinned to.
    ANCHORS = ("top-left", "top-right", "bottom-left", "bottom-right")
    #: Distance from the plot area corner of the default position.
    DEFAULT_OFFSET = QPoint(16, 56)
    _MARGIN = 8

    def __init__(self, parent: QWidget, interrupt=None, cancel=None) -> None:
        """Create the (initially hidden) activity panel."""
        super().__init__(parent)
        self._interrupt = interrupt
        self._cancel = cancel
        self._cards: List[ActivityCard] = []
        self._drag_origin: Optional[QPoint] = None

        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        anchor = settings.value("activity/anchor", "")
        self._anchor: Optional[str] = anchor if anchor in self.ANCHORS else None
        self._offset = QPoint(
            int(settings.value("activity/offset_x", 0)),
            int(settings.value("activity/offset_y", 0)),
        )
        self._pinned = _as_bool(settings.value("activity/pinned", False))
        self._collapsed = _as_bool(settings.value("activity/collapsed", False))

        self._header = _DragHeader()
        self._header.setProperty("floating", True)
        self._header.setFixedWidth(340)
        self._grip = QLabel()
        self._summary = ElidedLabel("Activity")
        self._summary.setProperty("heading", True)
        self._btn_clear = self._header_button("Clear finished", self.clear_finished)
        self._btn_collapse = self._header_button("", lambda: self.set_collapsed(not self._collapsed))
        self._btn_pin = self._header_button("", lambda: self.set_pinned(not self._pinned))
        row = QHBoxLayout(self._header)
        row.setContentsMargins(10, 4, 4, 4)
        row.setSpacing(4)
        row.addWidget(self._grip)
        row.addWidget(self._summary, 1)
        row.addWidget(self._btn_clear)
        row.addWidget(self._btn_collapse)
        row.addWidget(self._btn_pin)
        self._header.drag_started.connect(self._on_drag_started)
        self._header.drag_moved.connect(self._on_drag_moved)
        self._header.drag_finished.connect(self._on_drag_finished)
        self._header.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        reset = QAction("Reset position", self._header)
        reset.triggered.connect(self.reset_position)
        self._header.addAction(reset)

        self._cards_box = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_box)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addWidget(self._header)
        self._layout.addWidget(self._cards_box)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        parent.installEventFilter(self)
        central = getattr(parent, "centralWidget", None)
        if callable(central) and central() is not None:
            central().installEventFilter(self)
        theme().changed.connect(self._refresh_header)
        self._refresh_header()
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cards(self) -> List[ActivityCard]:
        """The cards currently shown."""
        return list(self._cards)

    @property
    def anchor(self) -> Optional[str]:
        """Corner the panel is pinned to (``None``: default position)."""
        return self._anchor

    @property
    def offset(self) -> QPoint:
        """Distance of the panel from its :attr:`anchor` corner."""
        return QPoint(self._offset)

    @property
    def pinned(self) -> bool:
        """Whether the position is locked."""
        return self._pinned

    @property
    def collapsed(self) -> bool:
        """Whether only the header is shown."""
        return self._collapsed

    def track(self, job) -> Optional[ActivityCard]:
        """
        Show a card for *job* (a kernel task or a local job).

        Parameters
        ----------
        job : Task or LocalJob
            The operation to follow; quiet jobs are ignored.

        Returns
        -------
        ActivityCard or None
            The card, or ``None`` for quiet jobs.
        """
        if getattr(job, "quiet", False):
            return None
        cancel = self._cancel if hasattr(job, "msg_id") else None
        interrupt = self._interrupt if hasattr(job, "msg_id") else None
        card = ActivityCard(job, interrupt=interrupt, cancel=cancel, parent=self._cards_box)
        card.dismissed.connect(lambda c=card: self._remove(c))
        job.finished.connect(lambda c=card: self._on_finished(c))
        job.changed.connect(self._refresh_header)
        job.changed.connect(self._schedule_reposition)
        self._cards.append(card)
        self._cards_layout.addWidget(card)
        self._trim()
        self.show()
        self.raise_()
        self._refresh_header()
        self._reposition()
        self._schedule_reposition()
        if job.is_final:
            self._on_finished(card)
        return card

    def clear_finished(self) -> None:
        """Dismiss every finished card."""
        for card in [c for c in self._cards if c.job.is_final]:
            self._remove(card)

    def set_pinned(self, pinned: bool) -> None:
        """
        Lock or unlock the panel position.

        Parameters
        ----------
        pinned : bool
            ``True`` to prevent dragging.
        """
        self._pinned = bool(pinned)
        self._save("activity/pinned", self._pinned)
        self._refresh_header()

    def set_collapsed(self, collapsed: bool) -> None:
        """
        Show only the header (with a summary), or the full cards.

        Parameters
        ----------
        collapsed : bool
            ``True`` to hide the cards.
        """
        self._collapsed = bool(collapsed)
        self._save("activity/collapsed", self._collapsed)
        self._refresh_header()
        self._reposition()

    def move_to(self, anchor: str, offset: QPoint) -> None:
        """
        Pin the panel to a window corner.

        Parameters
        ----------
        anchor : str
            One of :attr:`ANCHORS`.
        offset : QPoint
            Distance of the panel from that corner.
        """
        if anchor not in self.ANCHORS:
            raise ValueError(f"Unknown anchor {anchor!r}; use one of {self.ANCHORS}.")
        self._anchor = anchor
        self._offset = QPoint(max(offset.x(), 0), max(offset.y(), 0))
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue("activity/anchor", anchor)
        settings.setValue("activity/offset_x", self._offset.x())
        settings.setValue("activity/offset_y", self._offset.y())
        self._reposition()

    def reset_position(self) -> None:
        """Go back to the default position (top-left of the plot area)."""
        self._anchor = None
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        for key in ("activity/anchor", "activity/offset_x", "activity/offset_y"):
            settings.remove(key)
        self._reposition()

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------

    def _on_finished(self, card: ActivityCard) -> None:
        if card.job.state in ("done", "cancelled"):
            QTimer.singleShot(int(TOAST_SECONDS * 1000), lambda: self._remove(card))
        self._refresh_header()
        self._reposition()

    def _trim(self) -> None:
        """Drop the oldest finished cards beyond :data:`MAX_CARDS`."""
        while len(self._cards) > MAX_CARDS:
            finished = [c for c in self._cards if c.job.is_final]
            if not finished:
                break
            self._remove(finished[0])

    def _remove(self, card: ActivityCard) -> None:
        if card not in self._cards:
            return
        self._cards.remove(card)
        self._cards_layout.removeWidget(card)
        card.deleteLater()
        if not self._cards:
            self.hide()
        self._refresh_header()
        self._reposition()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    @staticmethod
    def _save(key: str, value) -> None:
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(key, value)

    def _header_button(self, tip: str, slot) -> QToolButton:
        button = QToolButton()
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.clicked.connect(slot)
        return button

    def summary_text(self) -> str:
        """
        Return the header summary, e.g. ``'Activity · 2 running · 1 failed'``.

        Returns
        -------
        str
            The summary.
        """
        states = [c.job.state for c in self._cards]
        parts = []
        for label, names in (
            ("running", ("running",)),
            ("queued", ("queued",)),
            ("failed", ("error",)),
            ("done", ("done", "cancelled")),
        ):
            count = sum(s in names for s in states)
            if count:
                parts.append(f"{count} {label}")
        return " · ".join(["Activity"] + parts)

    def _refresh_header(self) -> None:
        t = theme()
        self._summary.setText(self.summary_text())
        self._summary.setToolTip(self.summary_text())
        self._grip.setPixmap(
            t.icon("drag", "text_muted" if self._pinned else "text").pixmap(QSize(16, 16))
        )
        self._grip.setVisible(not self._pinned)
        self._header.draggable = not self._pinned
        self._header.setCursor(
            Qt.CursorShape.ArrowCursor if self._pinned else Qt.CursorShape.OpenHandCursor
        )
        self._header.setToolTip(
            "Position locked" if self._pinned else "Drag to move; the panel pins to the nearest corner"
        )
        self._btn_clear.setIcon(t.icon("broom", "text_muted"))
        self._btn_clear.setEnabled(any(c.job.is_final for c in self._cards))
        self._btn_collapse.setIcon(t.icon("chevron-down" if self._collapsed else "chevron-up"))
        self._btn_collapse.setToolTip("Show the operations" if self._collapsed else "Collapse")
        self._btn_pin.setIcon(t.icon("pin" if self._pinned else "pin-outline", "accent" if self._pinned else "text"))
        self._btn_pin.setToolTip("Unlock the position" if self._pinned else "Lock the position")
        self._cards_box.setVisible(not self._collapsed)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _area(self) -> QRect:
        """Window area available to the panel (between menu and status bar)."""
        parent = self.parentWidget()
        rect = QRect(0, 0, parent.width(), parent.height())
        menu = getattr(parent, "menuBar", None)
        if callable(menu) and menu().isVisible():
            rect.setTop(menu().geometry().bottom() + 1)
        status = getattr(parent, "statusBar", None)
        if callable(status) and status().isVisible():
            rect.setBottom(status().geometry().top() - 1)
        return rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)

    def _default_top_left(self) -> QPoint:
        """Top-left corner of the default position (plot area, below its toolbar)."""
        parent = self.parentWidget()
        central = getattr(parent, "centralWidget", None)
        if callable(central) and central() is not None:
            origin = central().geometry().topLeft()
        else:
            origin = self._area().topLeft()
        return origin + self.DEFAULT_OFFSET

    def _size(self) -> QSize:
        for card in self._cards:
            card.layout().activate()
            card.setFixedHeight(card.layout().sizeHint().height())
        self._cards_layout.activate()
        self._layout.activate()
        return self.sizeHint()

    def _schedule_reposition(self) -> None:
        QTimer.singleShot(0, self._reposition)

    def _reposition(self) -> None:
        if self.parentWidget() is None or self._drag_origin is not None:
            return
        size = self._size()
        area = self._area()
        if self._anchor is None:
            pos = self._default_top_left()
        else:
            x = area.left() + self._offset.x()
            y = area.top() + self._offset.y()
            if self._anchor.endswith("right"):
                x = area.left() + area.width() - self._offset.x() - size.width()
            if self._anchor.startswith("bottom"):
                y = area.top() + area.height() - self._offset.y() - size.height()
            pos = QPoint(x, y)
        self.setGeometry(QRect(self._clamp(pos, size), size))
        self.raise_()

    def _clamp(self, pos: QPoint, size: QSize) -> QPoint:
        area = self._area()
        x = min(max(pos.x(), area.left()), max(area.left(), area.left() + area.width() - size.width()))
        y = min(max(pos.y(), area.top()), max(area.top(), area.top() + area.height() - size.height()))
        return QPoint(x, y)

    def nearest_anchor(self, rect: QRect):
        """
        Return the window corner nearest to *rect* and the distance from it.

        Parameters
        ----------
        rect : QRect
            Panel geometry in window coordinates.

        Returns
        -------
        anchor : str
            One of :attr:`ANCHORS`.
        offset : QPoint
            Distance of *rect* from that corner.
        """
        area = self._area()
        center = rect.center()
        vertical = "top" if center.y() < area.center().y() else "bottom"
        horizontal = "left" if center.x() < area.center().x() else "right"
        dx = rect.left() - area.left()
        dy = rect.top() - area.top()
        if horizontal == "right":
            dx = area.left() + area.width() - (rect.left() + rect.width())
        if vertical == "bottom":
            dy = area.top() + area.height() - (rect.top() + rect.height())
        return f"{vertical}-{horizontal}", QPoint(max(dx, 0), max(dy, 0))

    def _on_drag_started(self, global_pos: QPoint) -> None:
        self._drag_origin = global_pos - self.mapToGlobal(QPoint(0, 0))

    def _on_drag_moved(self, global_pos: QPoint) -> None:
        if self._drag_origin is None:
            return
        target = self.parentWidget().mapFromGlobal(global_pos - self._drag_origin)
        self.move(self._clamp(target, self.size()))

    def _on_drag_finished(self) -> None:
        if self._drag_origin is None:
            return
        self._drag_origin = None
        anchor, offset = self.nearest_anchor(self.geometry())
        self.move_to(anchor, offset)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        """Follow the window and plot area geometry."""
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move) and self.isVisible():
            self._schedule_reposition()
        return False


def _as_bool(value) -> bool:
    """Convert a QSettings value (possibly the string ``'true'``) to bool."""
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


class StartupOverlay(QWidget):
    """
    Translucent overlay shown over the main window while the kernel starts.

    Parameters
    ----------
    steps : list of (str, str)
        Startup steps as ``(key, label)``.
    title : str
        Heading of the overlay card.
    subtitle : str
        Secondary text (e.g. the experiment name).
    parent : QWidget
        The main window.

    Signals
    -------
    closed
        The overlay was dismissed.
    """

    closed = Signal()

    def __init__(
        self,
        steps,
        title: str,
        subtitle: str,
        parent: QWidget,
    ) -> None:
        """Build the overlay with every step pending."""
        super().__init__(parent)
        self._rows: Dict[str, tuple] = {}
        self._spins: Dict[str, object] = {}
        self._has_error = False
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        card = QFrame(self)
        card.setProperty("floating", True)
        card.setFixedWidth(420)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 90))
        card.setGraphicsEffect(shadow)
        self._card = card

        import qtawesome as qta

        heading = QLabel(title)
        heading.setProperty("title", True)
        sub = QLabel(subtitle)
        sub.setProperty("muted", True)
        sub.setWordWrap(True)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(6)
        layout.addWidget(heading)
        layout.addWidget(sub)
        layout.addSpacing(12)
        for key, label in steps:
            icon = qta.IconWidget()
            text = QLabel(label)
            message = QLabel("")
            message.setProperty("muted", True)
            message.setWordWrap(True)
            message.hide()
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(icon)
            row.addWidget(text, 1)
            layout.addLayout(row)
            layout.addWidget(message)
            self._rows[key] = (icon, text, message)
            self.set_step(key, "pending")
        layout.addSpacing(10)
        self._button = QPushButton("Continue")
        self._button.setProperty("accent", True)
        self._button.clicked.connect(self.dismiss)
        self._button.hide()
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self._button)
        layout.addLayout(button_row)

        parent.installEventFilter(self)
        self._fit_parent()
        self.show()
        self.raise_()

    @property
    def has_error(self) -> bool:
        """Whether a startup step failed."""
        return self._has_error

    def set_step(self, key: str, status: str, message: str = "") -> None:
        """
        Update one startup step.

        Parameters
        ----------
        key : str
            Step key.
        status : str
            ``'pending'``, ``'running'``, ``'done'`` or ``'error'``.
        message : str, optional
            Extra text (error description).
        """
        if key not in self._rows:
            return
        import qtawesome as qta

        icon, text, label = self._rows[key]
        t = theme()
        spin = self._spins.pop(key, None)
        if spin is not None:
            spin.stop()
        if status == "running":
            self._spins[key] = qta.Spin(icon, autostart=True)
            icon.setIcon(t.icon("loading", "accent", animation=self._spins[key]))
        elif status == "done":
            icon.setIcon(t.icon("check-circle", "success"))
        elif status == "error":
            self._has_error = True
            icon.setIcon(t.icon("alert-circle", "danger"))
            self._button.setText("Continue anyway")
            self._button.show()
        else:
            icon.setIcon(t.icon("circle-outline", "text_muted"))
        text.setProperty("muted", status == "pending")
        repolish(text)
        label.setText(message)
        label.setVisible(bool(message))

    def finish(self) -> None:
        """Close the overlay, unless an error is shown."""
        if self._has_error:
            return
        QTimer.singleShot(250, self.dismiss)

    def dismiss(self) -> None:
        """Close the overlay."""
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def _fit_parent(self) -> None:
        parent = self.parentWidget()
        self.setGeometry(0, 0, parent.width(), parent.height())
        hint = self._card.sizeHint()
        self._card.setGeometry(
            (self.width() - hint.width()) // 2,
            (self.height() - hint.height()) // 2,
            hint.width(),
            hint.height(),
        )

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        """Follow the parent window geometry."""
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._fit_parent()
        return False

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Paint the translucent backdrop."""
        painter = QPainter(self)
        backdrop = QColor(theme().tokens["bg"])
        backdrop.setAlpha(215)
        painter.fillRect(self.rect(), backdrop)
