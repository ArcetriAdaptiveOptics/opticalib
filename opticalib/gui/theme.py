"""
Theming for CalpyGUI
====================

Colors are defined once as *tokens* for a light and a dark theme; the Qt
palette, the application style sheet, the icons and the embedded widgets
(console, plots) are all derived from them.

The theme follows the system color scheme by default (Qt >= 6.5) and can be
forced to light or dark; the choice is stored in the application settings.
Widgets that draw with theme colors connect to :attr:`ThemeManager.changed`.
"""

from typing import Dict, Optional

from qtpy.QtCore import QObject, QSettings, Qt, Signal
from qtpy.QtGui import QColor, QGuiApplication, QIcon, QPalette
from qtpy.QtWidgets import QApplication

#: Settings organization and application names, shared by the whole GUI.
SETTINGS_ORG = "ArcetriAdaptiveOptics"
SETTINGS_APP = "CalpyGUI"

#: Theme modes selectable by the user.
THEME_MODES = ("system", "light", "dark")

LIGHT_TOKENS: Dict[str, str] = {
    "bg": "#f4f5f8",
    "surface": "#ffffff",
    "surface_alt": "#eceef3",
    "surface_hover": "#e3e6ee",
    "border": "#d6dae3",
    "text": "#1c2230",
    "text_muted": "#5f6878",
    "accent": "#3867d6",
    "accent_hover": "#2f59bd",
    "accent_text": "#ffffff",
    "accent_soft": "#dfe8fb",
    "success": "#1c9a5f",
    "warning": "#b87a0b",
    "danger": "#d24545",
    "selection": "#cddcfa",
    "plot_bg": "#ffffff",
    "plot_fg": "#3a4252",
}

DARK_TOKENS: Dict[str, str] = {
    "bg": "#15171c",
    "surface": "#1d2027",
    "surface_alt": "#252932",
    "surface_hover": "#2d323d",
    "border": "#333845",
    "text": "#e5e8ef",
    "text_muted": "#98a1b3",
    "accent": "#6d96ff",
    "accent_hover": "#86a8ff",
    "accent_text": "#0f1320",
    "accent_soft": "#26324d",
    "success": "#3fcf8e",
    "warning": "#f0b44a",
    "danger": "#ff6b6b",
    "selection": "#2f4270",
    "plot_bg": "#1d2027",
    "plot_fg": "#c9cfdb",
}

_STYLE_SHEET = """
QWidget {{
    color: {text};
}}
QMainWindow, QDialog {{
    background: {bg};
}}
QMainWindow::separator {{
    background: {bg};
    width: 6px;
    height: 6px;
}}
QDockWidget {{
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background: {bg};
    padding: 6px 8px;
    text-align: left;
}}
QDockWidget > QWidget {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QToolTip {{
    background: {surface_alt};
    color: {text};
    border: 1px solid {border};
    padding: 4px 6px;
}}
QMenuBar {{
    background: {bg};
}}
QMenuBar::item:selected, QMenu::item:selected {{
    background: {selection};
    border-radius: 4px;
}}
QMenu {{
    background: {surface};
    border: 1px solid {border};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 18px 5px 10px;
}}
QMenu::separator {{
    height: 1px;
    background: {border};
    margin: 4px 6px;
}}
QPushButton {{
    background: {surface_alt};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 5px 12px;
}}
QPushButton:hover {{
    background: {surface_hover};
}}
QPushButton:pressed {{
    background: {selection};
}}
QPushButton:disabled {{
    color: {text_muted};
    background: {surface};
}}
QPushButton[accent="true"] {{
    background: {accent};
    border: 1px solid {accent};
    color: {accent_text};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {accent_hover};
}}
QPushButton[accent="true"]:disabled {{
    background: {surface_alt};
    border: 1px solid {border};
    color: {text_muted};
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px;
}}
QToolButton:hover {{
    background: {surface_hover};
    border: 1px solid {border};
}}
QToolButton:checked {{
    background: {accent_soft};
}}
QToolButton::menu-indicator {{
    image: none;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {selection};
    selection-color: {text};
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {accent};
}}
QComboBox QAbstractItemView {{
    background: {surface};
    border: 1px solid {border};
    selection-background-color: {selection};
}}
QTreeView, QListView, QTableView {{
    background: {surface};
    alternate-background-color: {surface_alt};
    border: none;
    selection-background-color: {selection};
    selection-color: {text};
    outline: 0;
}}
QTreeView::item, QListView::item {{
    padding: 3px 2px;
}}
QTreeView::item:hover, QListView::item:hover {{
    background: {surface_hover};
}}
QTreeView::item:selected, QListView::item:selected {{
    background: {selection};
}}
QHeaderView::section {{
    background: {surface};
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border};
    padding: 4px 6px;
}}
QTabWidget::pane {{
    border: none;
}}
QTabBar::tab {{
    background: transparent;
    color: {text_muted};
    padding: 6px 12px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {text};
    border-bottom: 2px solid {accent};
}}
QTabBar::tab:hover {{
    color: {text};
}}
/* Panel tabs (below the panels): the selected tab names the panel shown,
   so it is highlighted like a title. */
QTabBar::tab:bottom {{
    border-bottom: none;
    border-top: 3px solid transparent;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
    padding: 6px 14px 7px 14px;
    margin-right: 2px;
}}
QTabBar::tab:bottom:selected {{
    color: {text};
    background: {accent_soft};
    border-top: 3px solid {accent};
    font-weight: 600;
}}
QTabBar::tab:bottom:hover:!selected {{
    background: {surface_hover};
}}
QWidget#DockTitleBar {{
    background: {bg};
    border: none;
    border-radius: 0px;
}}
QStatusBar {{
    background: {bg};
    color: {text_muted};
}}
QStatusBar::item {{
    border: none;
}}
QProgressBar {{
    background: {surface_alt};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 3px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle {{
    background: {border};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background: {text_muted};
}}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}
QSplitter::handle {{
    background: {bg};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QLabel[muted="true"] {{
    color: {text_muted};
}}
QLabel[title="true"] {{
    font-size: 16pt;
    font-weight: 600;
}}
QLabel[heading="true"] {{
    font-size: 11pt;
    font-weight: 600;
}}
QLabel[section="true"] {{
    color: {text_muted};
    font-size: 8.5pt;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QFrame[card="true"] {{
    background: {surface_alt};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame[card="true"]:hover {{
    border: 1px solid {text_muted};
}}
QFrame[floating="true"] {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 10px;
}}
QFrame[floating="true"][state="error"] {{
    border: 1px solid {danger};
}}
QFrame[floating="true"][state="done"] {{
    border: 1px solid {success};
}}
QFrame#Toolbar {{
    background: {surface};
    border: none;
    border-bottom: 1px solid {border};
}}
"""


def _system_is_dark() -> bool:
    """Return whether the operating system uses a dark color scheme."""
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is not None:
        try:
            return scheme() == Qt.ColorScheme.Dark
        except AttributeError:
            pass
    window = QGuiApplication.palette().color(QPalette.ColorRole.Window)
    return window.lightness() < 128


def build_style_sheet(tokens: Dict[str, str]) -> str:
    """
    Return the application style sheet for the given color *tokens*.

    Parameters
    ----------
    tokens : dict
        One of :data:`LIGHT_TOKENS` or :data:`DARK_TOKENS`.

    Returns
    -------
    str
        The Qt style sheet.
    """
    return _STYLE_SHEET.format(**tokens)


def console_style_sheet(tokens: Dict[str, str]) -> str:
    """
    Return the style sheet of the qtconsole widget for the given *tokens*.

    Parameters
    ----------
    tokens : dict
        One of :data:`LIGHT_TOKENS` or :data:`DARK_TOKENS`.

    Returns
    -------
    str
        Style sheet using qtconsole's prompt classes.
    """
    return (
        "QPlainTextEdit, QTextEdit {{ background-color: {surface}; color: {text};"
        " selection-background-color: {selection}; border: none; }}"
        " .inverted {{ background-color: {text}; color: {surface}; }}"
        " .error {{ color: {danger}; }}"
        " .in-prompt {{ color: {accent}; }}"
        " .in-prompt-number {{ font-weight: bold; }}"
        " .out-prompt {{ color: {danger}; }}"
        " .out-prompt-number {{ font-weight: bold; }}"
    ).format(**tokens)


def build_palette(tokens: Dict[str, str]) -> QPalette:
    """
    Return a Qt palette matching the given color *tokens*.

    Parameters
    ----------
    tokens : dict
        One of :data:`LIGHT_TOKENS` or :data:`DARK_TOKENS`.

    Returns
    -------
    QPalette
        Palette for the Fusion style.
    """
    role = QPalette.ColorRole
    palette = QPalette()
    colors = {
        role.Window: tokens["bg"],
        role.WindowText: tokens["text"],
        role.Base: tokens["surface"],
        role.AlternateBase: tokens["surface_alt"],
        role.ToolTipBase: tokens["surface_alt"],
        role.ToolTipText: tokens["text"],
        role.Text: tokens["text"],
        role.Button: tokens["surface_alt"],
        role.ButtonText: tokens["text"],
        role.BrightText: tokens["danger"],
        role.Highlight: tokens["accent"],
        role.HighlightedText: tokens["accent_text"],
        role.Link: tokens["accent"],
        role.PlaceholderText: tokens["text_muted"],
    }
    for color_role, value in colors.items():
        palette.setColor(color_role, QColor(value))
    disabled = QPalette.ColorGroup.Disabled
    for color_role in (role.Text, role.ButtonText, role.WindowText):
        palette.setColor(disabled, color_role, QColor(tokens["text_muted"]))
    return palette


class ThemeManager(QObject):
    """
    Application-wide theme state.

    Use :func:`theme` to get the shared instance.

    Signals
    -------
    changed
        Emitted after the theme has been (re)applied.
    """

    changed = Signal()

    def __init__(self) -> None:
        """Initialise with the mode stored in the settings."""
        super().__init__()
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        mode = settings.value("ui/theme", "system")
        self._mode = mode if mode in THEME_MODES else "system"
        self._dark = False
        self._tokens = LIGHT_TOKENS
        self._hooked = False

    @property
    def mode(self) -> str:
        """The selected mode: ``'system'``, ``'light'`` or ``'dark'``."""
        return self._mode

    @property
    def is_dark(self) -> bool:
        """Whether the dark theme is active."""
        return self._dark

    @property
    def tokens(self) -> Dict[str, str]:
        """The color tokens of the active theme."""
        return self._tokens

    def color(self, token: str) -> QColor:
        """
        Return the color of *token* in the active theme.

        Parameters
        ----------
        token : str
            Token name, e.g. ``'accent'``.

        Returns
        -------
        QColor
            The color.
        """
        return QColor(self._tokens[token])

    def icon(self, name: str, token: str = "text", **options) -> QIcon:
        """
        Return a Material Design icon drawn in a theme color.

        Parameters
        ----------
        name : str
            Icon name without prefix, e.g. ``'play'`` for ``mdi6.play``.
        token : str, optional
            Color token of the icon.
        **options
            Extra ``qtawesome.icon`` options (e.g. ``animation``).

        Returns
        -------
        QIcon
            The icon; an empty icon when qtawesome is unavailable.
        """
        try:
            import qtawesome as qta
        except ImportError:
            return QIcon()
        options.setdefault("color", self._tokens[token])
        options.setdefault("color_disabled", self._tokens["text_muted"])
        return qta.icon(f"mdi6.{name}", **options)

    def set_mode(self, mode: str) -> None:
        """
        Select and apply a theme mode, and store it in the settings.

        Parameters
        ----------
        mode : str
            One of :data:`THEME_MODES`.
        """
        if mode not in THEME_MODES:
            raise ValueError(f"Unknown theme mode {mode!r}; use one of {THEME_MODES}.")
        self._mode = mode
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("ui/theme", mode)
        self.apply()

    def apply(self, app: Optional[QApplication] = None) -> None:
        """
        Apply the current theme to the application.

        Parameters
        ----------
        app : QApplication, optional
            The application; defaults to the running instance.
        """
        app = app or QApplication.instance()
        if app is None:
            return
        if not self._hooked:
            self._hooked = True
            app.setStyle("Fusion")
            hints = QGuiApplication.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        if self._mode == "system":
            self._dark = _system_is_dark()
        else:
            self._dark = self._mode == "dark"
        self._tokens = DARK_TOKENS if self._dark else LIGHT_TOKENS
        app.setPalette(build_palette(self._tokens))
        app.setStyleSheet(build_style_sheet(self._tokens))
        self.changed.emit()

    def _on_system_scheme_changed(self, *args) -> None:
        """Re-apply the theme when the system scheme changes."""
        if self._mode == "system":
            self.apply()


_THEME: Optional[ThemeManager] = None


def theme() -> ThemeManager:
    """
    Return the shared :class:`ThemeManager`.

    Returns
    -------
    ThemeManager
        The application theme manager.
    """
    global _THEME
    if _THEME is None:
        _THEME = ThemeManager()
    return _THEME


def repolish(widget) -> None:
    """
    Re-evaluate the style sheet of *widget* after a dynamic property change.

    Parameters
    ----------
    widget : QWidget
        Widget whose property (e.g. ``state``) changed.
    """
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
