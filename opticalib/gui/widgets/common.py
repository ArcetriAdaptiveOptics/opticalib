"""
Small widgets shared by the CalpyGUI panels.
"""

from typing import Optional

from qtpy.QtCore import QSettings, Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import SETTINGS_APP, SETTINGS_ORG, theme


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


class CollapsibleSection(QWidget):
    """
    A titled group of widgets that can be folded away.

    The header shows a chevron, the title and a one-line summary (visible
    also when the group is folded, e.g. "3 devices · 1 connected"); clicking
    it folds or unfolds the *body*.  Widgets that belong to the header (a
    refresh button, ...) go to its right with :meth:`add_header_widget`.
    The state is remembered between sessions when a *key* is given; it is
    stored with the preferences shared by every CalpyGUI application (theme,
    activity panel).

    Parameters
    ----------
    title : str
        Title of the group.
    key : str, optional
        Name under which the state is saved (empty: not saved).
    collapsed : bool, optional
        Initial state when nothing was saved.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    body : QWidget
        Container of the content: give it a layout, or add widgets to the
        layout set with :meth:`set_body_layout`.

    Signals
    -------
    toggled(bool)
        The group was folded (``True``) or unfolded (``False``).
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        key: str = "",
        collapsed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Create the section, restoring its saved state."""
        super().__init__(parent)
        self._key = key
        self._collapsed = collapsed
        if key:
            saved = QSettings(SETTINGS_ORG, SETTINGS_APP).value(self._setting())
            if saved is not None:
                self._collapsed = str(saved).lower() == "true"

        self._button = QToolButton()
        self._button.setProperty("sectionToggle", True)
        self._button.setText(title)
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.clicked.connect(self.toggle)
        self._summary = QLabel()
        self._summary.setProperty("muted", True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._header = QHBoxLayout()
        self._header.setContentsMargins(0, 0, 0, 0)
        self._header.addWidget(self._button)
        self._header.addWidget(self._summary)
        self._header.addStretch()

        self.body = QWidget()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addLayout(self._header)
        column.addWidget(self.body)

        theme().changed.connect(self._refresh)
        self._refresh()

    def _setting(self) -> str:
        return f"ui/collapsed_{self._key}"

    @property
    def title(self) -> str:
        """The title of the group."""
        return self._button.text()

    @property
    def collapsed(self) -> bool:
        """Whether the body is folded away."""
        return self._collapsed

    def set_body_layout(self, layout) -> None:
        """Use *layout* (e.g. a ``QVBoxLayout``) for the body."""
        self.body.setLayout(layout)

    def add_header_widget(self, widget: QWidget) -> None:
        """Add *widget* at the right end of the header."""
        self._header.addWidget(widget)

    def set_summary(self, text: str) -> None:
        """
        Show a short description next to the title.

        Parameters
        ----------
        text : str
            Plain or rich text (empty hides it).
        """
        self._summary.setText(text)
        self._summary.setVisible(bool(text))

    def toggle(self) -> None:
        """Fold the group if unfolded, and the other way round."""
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """
        Fold or unfold the group.

        Parameters
        ----------
        collapsed : bool
            ``True`` to fold it away.
        """
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if self._key:
            QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(
                self._setting(), "true" if collapsed else "false"
            )
        self._refresh()
        self.toggled.emit(collapsed)

    def _refresh(self) -> None:
        self.body.setVisible(not self._collapsed)
        self._button.setIcon(
            theme().icon("chevron-right" if self._collapsed else "chevron-down", "text_muted")
        )
        self._button.setToolTip(
            "Click to show" if self._collapsed else "Click to hide"
        )
