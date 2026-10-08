"""
Small widgets shared by the CalpyGUI panels.
"""

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """
    Single-line label that elides its text in the middle when too long.

    Useful for file paths: the full text stays available in :meth:`full_text`
    (and should be set as tooltip by the caller if needed).

    Signals
    -------
    clicked
        The label was clicked.
    """

    clicked = Signal()

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        """Create the label with *text*."""
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def full_text(self) -> str:
        """The text before elision."""
        return self._full_text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        """Set the (unelided) text."""
        self._full_text = text
        self._elide()

    def sizeHint(self):  # noqa: N802 - Qt override
        """Prefer the width of the full text."""
        hint = super().sizeHint()
        hint.setWidth(self.fontMetrics().horizontalAdvance(self._full_text) + 8)
        return hint

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        """Allow shrinking down to a few characters."""
        hint = super().minimumSizeHint()
        hint.setWidth(min(hint.width(), 40))
        return hint

    def _elide(self) -> None:
        width = max(self.width() - 2, 20)
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, width
        )
        super().setText(elided)

    # Qt event handlers: the camelCase names are required for Qt to call them.
    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-elide on resize."""
        super().resizeEvent(event)
        self._elide()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Emit :attr:`clicked`."""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class DockTitleBar(QWidget):
    """
    Slim title bar for the panels (dock widgets) of the main window.

    It shows the panel title (empty for tabbed panels, whose tab names
    them) and float / close buttons.  Mouse events on the bar itself are
    left to the ``QDockWidget``, so panels can still be dragged out of the
    window, re-docked, and floated with a double-click.

    Parameters
    ----------
    dock : QDockWidget
        The panel.
    """

    def __init__(self, dock) -> None:
        """Build the title bar of *dock*."""
        from qtpy.QtWidgets import QHBoxLayout, QToolButton

        from ..theme import theme

        super().__init__(dock)
        self._dock = dock
        self.setObjectName("DockTitleBar")
        self._label = ElidedLabel(dock.windowTitle())
        self._label.setProperty("muted", True)
        self._float = QToolButton()
        self._float.setAutoRaise(True)
        self._float.setToolTip(
            "Detach the panel into its own window (or double-click the bar)"
        )
        self._float.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self._close = QToolButton()
        self._close.setAutoRaise(True)
        self._close.setToolTip("Close the panel (reopen it from the View menu)")
        self._close.clicked.connect(dock.close)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 2, 2, 2)
        row.setSpacing(2)
        row.addWidget(self._label, 1)
        row.addWidget(self._float)
        row.addWidget(self._close)
        theme().changed.connect(self._refresh_icons)
        self._refresh_icons()

    def set_title(self, text: str) -> None:
        """
        Set the text shown in the bar.

        Parameters
        ----------
        text : str
            The title (an empty string shows none).
        """
        self._label.setText(text)

    def title(self) -> str:
        """The text shown in the bar."""
        return self._label.full_text()

    def _refresh_icons(self) -> None:
        from ..theme import theme

        t = theme()
        self._float.setIcon(
            t.icon(
                "dock-window" if not self._dock.isFloating() else "dock-left",
                "text_muted",
            )
        )
        self._close.setIcon(t.icon("close", "text_muted"))
