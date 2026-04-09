import os
import sys
import argparse
import subprocess
import importlib.util
from typing import Optional

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


def _launch_gui(config_path: Optional[str] = None) -> None:
    """
    Launch the CalpyGUI graphical interface.

    Parameters
    ----------
    config_path : str or None
        Absolute path to the ``configuration.yaml`` file to load, or
        *None* to use the opticalib default.
    """
    try:
        from opticalib.gui import launch_gui
    except ImportError as exc:
        print(
            f"Error: could not import the CalpyGUI module ({exc}).\n"
            "Make sure PyQt5 and qtconsole are installed:\n"
            "  pip install PyQt5 qtconsole"
        )
        sys.exit(1)
    launch_gui(config_path=config_path)


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
    return parser


def main():
    """
    Main function to handle command-line arguments and launch IPython
    shell with optional configuration.
    """
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

    parser = _build_parser()
    args = parser.parse_args()

    # --gui with no other arguments → GUI with the default configuration
    if args.gui and args.config_path is None and args.create is None:
        _launch_gui(config_path=None)
        return

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

        # --gui flag: open the graphical interface
        if args.gui:
            if not os.path.exists(config_path):
                # When a directory (not a .yaml file) was supplied, calpy
                # resolves the config to <dir>/SysConfig/configuration.yaml,
                # matching the folder tree created by `calpy -c <dir>`.
                config_path = os.path.join(
                    os.path.dirname(config_path),
                    "SysConfig",
                    "configuration.yaml",
                )
            _launch_gui(config_path=config_path)
            return

        # Start an IPython session with the resolved config
        try:
            if not os.path.exists(config_path):
                config_path = os.path.join(
                    os.path.dirname(config_path), "SysConfig", "configuration.yaml"
                )
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

    # No arguments: plain IPython session with the default opticalib config
    subprocess.run(
        [sys.executable, "-m", "IPython", "-i", init_file], check=False
    )


if __name__ == "__main__":
    main()
