import os
import re
import sys
import shutil
import argparse
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Optional

docs = """
CALPY DOCUMENTATION
`calpy` is a command-line tool that calls an interactive Python 
shell (IPython) with the option to pass the path to a configuration
file for the `opticalib` package.

Options:
--------
no option : Initialize an IPython shell executing the `opticalib` init script.
            This will load `opticalib` loading a pre-configured environment and 
            configuration file, in `~/.tmp_opticalib/SysConfig/configuration.yaml`.

-f <path> : Option to pass the path to a configuration file to be read 
            (e.g., '../opticalibConf/configuration.yaml'). Used to initiate
            the opticalib package.

-f <path> --gui : Launch the CalpyGUI graphical interface loaded with the
                  configuration file at <path>.  The embedded IPython terminal
                  is initialised identically to a plain `calpy -f <path>` session.

-f <path> --create : Create the configuration file in the specified path, 
                     as well as the complete data folder tree, and enters 
                     an ipython session importing opticalib. The created
                     configuration file is already updated with the provided
                     data path.
                     
-c|--create <path> : Create the configuration file in the specified path, as well as 
                     the complete  data folder tree, and exit. The created
                     configuration file is already updated with the provided
                     data path.

--gui : Launch the CalpyGUI graphical interface with the default configuration
        file (equivalent to running `calpy` without arguments but in GUI mode).

--install-launcher [-f <path>] : Create the `OptiCalib` desktop launcher for the
        GUI: a `.desktop` entry (applications menu and desktop) on Linux, Desktop
        and Start-menu shortcuts to `OptiCalib.exe` on Windows. With -f, the
        launcher opens that experiment.

--uninstall-launcher [-f <path>] : Remove the launcher created by
        --install-launcher (with the same -f, if any).

-h |--help : Shows this help message

"""


def check_dir(config_path: str) -> str:
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        if not os.path.isdir(config_path):
            raise OSError(f"Invalid Path: {config_path}")
    config_path = os.path.join(config_path, "configuration.yaml")
    return config_path


def _resolve_init_file() -> Optional[str]:
    """
    Resolve the path of the IPython bootstrap script for calpy.

    Returns
    -------
    str | None
        Absolute path to the init script when found, otherwise ``None``.
    """
    packaged_path = os.path.join(
        os.path.dirname(__file__), "opticalib", "__init_script__", "initCalpy.py"
    )
    local_dev_path = os.path.join(
        os.path.dirname(__file__), "__init_script__", "initCalpy.py"
    )

    for candidate in (packaged_path, local_dev_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _launch_gui(config_path: Optional[str] = None, report=print) -> None:
    """
    Launch the CalpyGUI graphical interface.

    Parameters
    ----------
    config_path : str or None
        Absolute path to the ``configuration.yaml`` file to load, or
        *None* to use the opticalib default.
    report : callable, optional
        Function showing error messages (e.g. a message box when there is
        no console).
    """
    problem = _gui_display_problem()
    if problem is not None:
        report(f"Error: {problem}")
        sys.exit(1)
    try:
        from opticalib.gui import launch_gui
    except ImportError as exc:
        report(
            f"Error: could not import the CalpyGUI module ({exc}).\n"
            "Make sure the GUI dependencies are installed:\n"
            f"  pip install {' '.join(GUI_REQUIREMENTS)}"
        )
        sys.exit(1)
    launch_gui(config_path=config_path)


#: Packages needed by the CalpyGUI graphical interface.
GUI_REQUIREMENTS = (
    "PySide6-Essentials",
    "QtPy",
    "qtconsole",
    "ipykernel",
    "pyqtgraph",
    "qtawesome",
)


def _gui_display_problem() -> Optional[str]:
    """
    Detect a missing display before Qt aborts the process.

    On Linux, Qt terminates the interpreter (it cannot be caught) when no
    display server is reachable, e.g. over SSH without X forwarding.

    Returns
    -------
    str or None
        A description of the problem, or ``None`` if the GUI can start.
    """
    if not sys.platform.startswith("linux"):
        return None
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    return (
        "no graphical display found (neither DISPLAY nor WAYLAND_DISPLAY is set).\n"
        "Run calpy --gui from a desktop session, or use 'ssh -X' to forward the display."
    )


def _resolve_config_path(path: str) -> str:
    """
    Expand and absolutize a raw config path string.

    When *path* does not end in ``.yaml`` it is treated as a directory;
    ``check_dir`` appends ``/configuration.yaml`` and creates the directory
    if necessary.

    Parameters
    ----------
    path : str
        Raw path string as supplied on the command line.

    Returns
    -------
    str
        Absolute path to the resolved ``configuration.yaml`` file.
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if ".yaml" not in path:
        path = check_dir(path)
    return path


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the calpy CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready to call ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        prog="calpy",
        description=(
            "Interactive Python shell for the opticalib package,\n"
            "with optional GUI and configuration file management."
        ),
        epilog=docs,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        metavar="PATH",
        dest="config_path",
        default=None,
        help=(
            "Path to a configuration file (or directory). "
            "Starts an IPython session with opticalib loaded using this config."
        ),
    )
    parser.add_argument(
        "-c",
        "--create",
        metavar="PATH",
        # nargs='?' allows --create to be used as a bare flag (modifier to -f)
        # or as --create <path> / -c <path> for the standalone create-and-exit mode.
        # const=True signals that the flag was given without a path argument.
        # default=None means the flag was not provided at all.
        nargs="?",
        const=True,
        default=None,
        dest="create",
        help=(
            "Standalone (with PATH): create the configuration file at PATH "
            "together with the full data folder tree, then exit. "
            "Combined with -f (no PATH): create the config at the -f path, "
            "then start an IPython session."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Launch the CalpyGUI graphical interface instead of a plain "
            "IPython terminal.  Can be combined with -f to load a specific "
            "configuration file."
        ),
    )
    launcher = parser.add_mutually_exclusive_group()
    launcher.add_argument(
        "--install-launcher",
        action="store_true",
        help=(
            f"Create the {APP_NAME} desktop launcher of the GUI (applications "
            "menu and desktop). With -f, the launcher opens that experiment."
        ),
    )
    launcher.add_argument(
        "--uninstall-launcher",
        action="store_true",
        help="Remove the launcher created by --install-launcher.",
    )
    return parser

def update_env_var(config_path: str) -> None:
    """
    Update the AOCONF environment variable to point to the specified config path.

    Sets the variable in the current process so it is inherited by any
    subsequently spawned child processes (IPython, GUI).  This is
    cross-platform; the previous ``export AOCONF=...`` shell call only
    worked on Unix and was a no-op for the parent process even there.

    Parameters
    ----------
    config_path : str
        Absolute path to the configuration file to set in the environment.
    """
    os.environ["AOCONF"] = config_path


def _prefer_sysconfig(config_path: str) -> str:
    """
    If *config_path* does not exist, try the standard
    ``<parent>/SysConfig/configuration.yaml`` layout.

    Parameters
    ----------
    config_path : str
        Candidate path to ``configuration.yaml``.

    Returns
    -------
    str
        An existing configuration path when found, otherwise *config_path*.
    """
    if os.path.exists(config_path):
        return config_path
    alt = os.path.join(
        os.path.dirname(config_path), "SysConfig", "configuration.yaml"
    )
    if os.path.exists(alt):
        return alt
    return config_path


# ---------------------------------------------------------------------------
# Desktop launcher (OptiCalib)
# ---------------------------------------------------------------------------

#: Name of the GUI executable and of its desktop launcher.
APP_NAME = "OptiCalib"


def _alert(message: str) -> None:
    """
    Report an error of the GUI launcher.

    ``OptiCalib.exe`` runs without a console (``pythonw``), and desktop
    launchers have no terminal either: on Windows the message is shown in a
    message box, on Linux sent as a desktop notification (``notify-send``)
    when no terminal shows it.
    """
    print(message)  # no-op without a console (sys.stdout is None)
    try:
        if sys.platform.startswith("win"):
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
        elif (sys.stdout is None or not sys.stdout.isatty()) and shutil.which("notify-send"):
            subprocess.run(["notify-send", APP_NAME, message], check=False)
    except Exception:
        pass


def gui_main() -> None:
    """
    Entry point of the ``OptiCalib`` executable: start the GUI.

    ``OptiCalib [-f PATH]`` is equivalent to ``calpy [-f PATH] --gui``; on
    Windows pip turns it into ``OptiCalib.exe``, which runs without a
    console window.
    """
    parser = argparse.ArgumentParser(
        prog=APP_NAME, description="Start the OptiCalib graphical interface (CalpyGUI)."
    )
    parser.add_argument(
        "-f", metavar="PATH", dest="config_path", default=None,
        help="Experiment folder or configuration file to open.",
    )
    args = parser.parse_args()
    config_path = None
    if args.config_path is not None:
        config_path = _prefer_sysconfig(_resolve_config_path(args.config_path))
        update_env_var(config_path)
    _launch_gui(config_path=config_path, report=_alert)


def _existing_config(path: str) -> str:
    """
    Resolve an existing configuration file from an experiment path.

    Unlike ``-f`` for a session, nothing is created: a folder must contain
    ``SysConfig/configuration.yaml`` or ``configuration.yaml``.

    Parameters
    ----------
    path : str
        Experiment folder or configuration file.

    Returns
    -------
    str
        Absolute path of the configuration file.

    Raises
    ------
    FileNotFoundError
        If no configuration file is found.
    """
    path = os.path.abspath(os.path.expanduser(path))
    candidates = [path] if path.endswith((".yaml", ".yml")) else [
        os.path.join(path, "SysConfig", "configuration.yaml"),
        os.path.join(path, "configuration.yaml"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No configuration file found at {path}")


def _experiment_name(config_path: str) -> str:
    """Experiment name of a configuration file (its folder, skipping SysConfig)."""
    folder = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(folder) == "SysConfig":
        folder = os.path.dirname(folder)
    return os.path.basename(folder) or "experiment"


def _launcher_names(config_path: Optional[str]):
    """(display name, file stem) of the launcher for *config_path*."""
    if config_path is None:
        return APP_NAME, APP_NAME
    experiment = _experiment_name(config_path)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment).strip("_") or "experiment"
    return f"{APP_NAME} ({experiment})", f"{APP_NAME}-{slug}"


def _icon_file(extension: str) -> Optional[str]:
    """Path of the OptiCalib icon shipped with the package (``png`` or ``ico``)."""
    spec = importlib.util.find_spec("opticalib")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        icon = os.path.join(location, "gui", "resources", f"opticalib.{extension}")
        if os.path.isfile(icon):
            return icon
    return None


def _calpy_command() -> List[str]:
    """
    Absolute command starting ``calpy`` of this environment.

    Desktop launchers do not activate conda or virtual environments, so the
    script installed next to this interpreter is used.
    """
    folder = Path(sys.executable).parent
    for candidate in (folder / "calpy", folder / "Scripts" / "calpy.exe", folder / "calpy.exe"):
        if candidate.is_file():
            return [str(candidate)]
    found = shutil.which("calpy")
    if found:
        return [os.path.abspath(found)]
    return [sys.executable, "-m", "setup_calpy"]


def _desktop_quote(arg: str) -> str:
    """
    Quote one argument of the ``Exec`` key of a ``.desktop`` file.

    Follows the Desktop Entry Specification: reserved characters require
    double quotes; inside them ``"``, `````, ``$`` and ``\\`` are escaped with a
    backslash, then the general string escaping doubles every backslash,
    and ``%`` (field codes) is doubled.
    """
    arg = arg.replace("%", "%%")
    if arg and not re.search(r'[\s"\'\\`$<>~|&;*?#()]', arg):
        return arg
    quoted = re.sub(r'(["`$\\])', r"\\\1", arg)
    return '"' + quoted.replace("\\", "\\\\") + '"'


def desktop_entry(command: List[str], name: str = APP_NAME, icon: Optional[str] = None) -> str:
    """
    Return the content of a freedesktop ``.desktop`` launcher.

    Parameters
    ----------
    command : list of str
        Command line to run.
    name : str, optional
        Name shown in the menus.
    icon : str, optional
        Icon file.

    Returns
    -------
    str
        The ``.desktop`` file content.
    """
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.5",
        f"Name={name}",
        "GenericName=Optical calibration",
        "Comment=Graphical interface of opticalib (CalpyGUI)",
        "Exec=" + " ".join(_desktop_quote(a) for a in command),
        f"Icon={icon}" if icon else "Icon=applications-science",
        "Terminal=false",
        "Categories=Science;Physics;",
        "Keywords=optics;calibration;deformable mirror;interferometer;",
        "StartupNotify=true",
        "StartupWMClass=CalpyGUI",
    ]
    return "\n".join(lines) + "\n"


def _linux_launcher_files(stem: str) -> List[Path]:
    """Applications-menu entry and (if there is one) the desktop copy."""
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(Path.home(), ".local", "share")
    files = [Path(data_home) / "applications" / f"{stem}.desktop"]
    desktop = None
    if shutil.which("xdg-user-dir"):
        try:
            out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=False)
            desktop = out.stdout.strip() or None
        except OSError:
            desktop = None
    if not desktop:
        desktop = str(Path.home() / "Desktop")
    if os.path.isdir(desktop) and os.path.abspath(desktop) != str(Path.home()):
        files.append(Path(desktop) / f"{stem}.desktop")
    return files


def _opticalib_exe() -> Optional[str]:
    """``OptiCalib.exe`` generated by pip in this environment (Windows)."""
    folder = Path(sys.executable).parent
    for candidate in (folder / "Scripts" / f"{APP_NAME}.exe", folder / f"{APP_NAME}.exe"):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(APP_NAME)
    return os.path.abspath(found) if found else None


def _powershell_quote(text: str) -> str:
    """Single-quoted PowerShell string literal."""
    return "'" + text.replace("'", "''") + "'"


def windows_shortcut_script(name: str, target: str, arguments: str = "", icon: Optional[str] = None, remove: bool = False) -> str:
    """
    Return the PowerShell script creating (or removing) the Windows shortcuts.

    Shortcuts are placed on the Desktop and in the Start menu (Programs);
    the script prints the path of each one.

    Parameters
    ----------
    name : str
        Shortcut name (without ``.lnk``).
    target : str
        Executable to start (``OptiCalib.exe``).
    arguments : str, optional
        Command-line arguments.
    icon : str, optional
        ``.ico`` file.
    remove : bool, optional
        Remove the shortcuts instead of creating them.

    Returns
    -------
    str
        The script.
    """
    folders = "@([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))"
    lnk = f"(Join-Path $folder {_powershell_quote(name + '.lnk')})"
    if remove:
        body = f"$p = {lnk}; if (Test-Path -LiteralPath $p) {{ Remove-Item -LiteralPath $p; Write-Output $p }}"
    else:
        body = (
            f"$p = {lnk}; $s = $shell.CreateShortcut($p); "
            f"$s.TargetPath = {_powershell_quote(target)}; "
            f"$s.Arguments = {_powershell_quote(arguments)}; "
            f"$s.WorkingDirectory = [Environment]::GetFolderPath('UserProfile'); "
            + (f"$s.IconLocation = {_powershell_quote(icon)}; " if icon else "")
            + f"$s.Description = {_powershell_quote('Graphical interface of opticalib')}; "
            "$s.Save(); Write-Output $p"
        )
    return (
        "$ErrorActionPreference = 'Stop'; $shell = New-Object -ComObject WScript.Shell; "
        f"foreach ($folder in {folders}) {{ {body} }}"
    )


def _run_powershell(script: str) -> List[str]:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "PowerShell failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def install_launcher(config_path: Optional[str] = None) -> List[str]:
    """
    Create the ``OptiCalib`` desktop launcher of the GUI.

    * Linux: a ``.desktop`` entry running ``calpy --gui`` in the
      applications menu and on the desktop;
    * Windows: Desktop and Start-menu shortcuts to ``OptiCalib.exe``.

    Parameters
    ----------
    config_path : str, optional
        Configuration file of the experiment the launcher opens (default:
        the opticalib default configuration).

    Returns
    -------
    list of str
        The files created.

    Raises
    ------
    RuntimeError
        On unsupported platforms, or when ``OptiCalib.exe`` is missing.
    """
    name, stem = _launcher_names(config_path)
    if sys.platform.startswith("linux"):
        command = _calpy_command() + ["--gui"]
        if config_path:
            command += ["-f", config_path]
        content = desktop_entry(command, name=name, icon=_icon_file("png"))
        created = []
        for file in _linux_launcher_files(stem):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(content)
            file.chmod(0o755)
            if shutil.which("gio"):  # GNOME: allow launching from the desktop
                subprocess.run(["gio", "set", str(file), "metadata::trusted", "true"],
                               capture_output=True, check=False)
            created.append(str(file))
        if shutil.which("update-desktop-database"):
            subprocess.run(["update-desktop-database", str(Path(created[0]).parent)],
                           capture_output=True, check=False)
        return created
    if sys.platform.startswith("win"):
        exe = _opticalib_exe()
        if exe is None:
            raise RuntimeError(
                f"{APP_NAME}.exe not found: reinstall opticalib (e.g. 'pip install -e .') to generate it."
            )
        arguments = f'-f "{config_path}"' if config_path else ""
        return _run_powershell(windows_shortcut_script(name, exe, arguments, _icon_file("ico")))
    raise RuntimeError(f"Desktop launchers are not supported on {sys.platform}; run 'calpy --gui'.")


def uninstall_launcher(config_path: Optional[str] = None) -> List[str]:
    """
    Remove the launcher created by :func:`install_launcher`.

    Parameters
    ----------
    config_path : str, optional
        The experiment the launcher was created for.

    Returns
    -------
    list of str
        The files removed.
    """
    name, stem = _launcher_names(config_path)
    if sys.platform.startswith("linux"):
        removed = []
        for file in _linux_launcher_files(stem):
            if file.exists():
                file.unlink()
                removed.append(str(file))
        return removed
    if sys.platform.startswith("win"):
        return _run_powershell(windows_shortcut_script(name, "", remove=True))
    raise RuntimeError(f"Desktop launchers are not supported on {sys.platform}.")


def main():
    """
    Main function to handle command-line arguments and launch IPython
    shell with optional configuration.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # --install-launcher / --uninstall-launcher [-f <path>]
    if args.install_launcher or args.uninstall_launcher:
        try:
            config = _existing_config(args.config_path) if args.config_path else None
            if args.install_launcher:
                paths = install_launcher(config)
                print(f"{APP_NAME} launcher created:")
            else:
                paths = uninstall_launcher(config)
                print(f"{APP_NAME} launcher removed:" if paths else f"No {APP_NAME} launcher found.")
        except (OSError, RuntimeError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        for path in paths:
            print(f"  {path}")
        return

    init_file = _resolve_init_file()
    if init_file is None:
        print(
            "Error: unable to locate 'initCalpy.py'. "
            "Reinstall opticalib or check package data installation."
        )
        sys.exit(1)
    # Check if IPython is installed in current interpreter
    if importlib.util.find_spec("IPython") is None:
        print("Error: IPython is not installed in this Python environment.")
        sys.exit(1)

    # -c <path> / --create <path>  (standalone: create config and exit)
    # Detected when 'create' holds a string path and no -f was given.
    if isinstance(args.create, str) and args.config_path is None:
        create_path = _resolve_config_path(args.create)
        from opticalib.core.root import create_configuration_file

        create_configuration_file(create_path, data_path=True)
        sys.exit(0)

    # -f <path> [--create] [--gui]
    if args.config_path is not None:
        config_path = _resolve_config_path(args.config_path)

        # --create (flag, no path) combined with -f: create config then continue
        if args.create is not None:
            from opticalib.core.root import create_configuration_file

            create_configuration_file(config_path, data_path=True)

        # Prefer SysConfig layout once the file exists (after create or for
        # existing experiments).  Set AOCONF only after the final path is known.
        config_path = _prefer_sysconfig(config_path)
        update_env_var(config_path)

        # --gui flag: open the graphical interface
        if args.gui:
            _launch_gui(config_path=config_path)
            return

        # Start an IPython session with the resolved config
        try:
            print("\n Initiating IPython Shell, importing Opticalib...\n")
            env = os.environ.copy()
            env["AOCONF"] = config_path
            # Launch IPython using the current interpreter for cross-platform compatibility
            ipython_cmd = [sys.executable, "-m", "IPython", "-i", init_file]
            subprocess.run(ipython_cmd, env=env, check=False)
        except OSError as ose:
            print(f"Error: {ose}")
            sys.exit(1)
        return

    # --gui with no other arguments → GUI with the default configuration
    if args.gui and args.config_path is None and args.create is None:
        _launch_gui(config_path=None)
        return

    # No arguments: plain IPython session with the default opticalib config
    subprocess.run(
        [sys.executable, "-m", "IPython", "-i", init_file], check=False
    )

if __name__ == "__main__":
    main()
