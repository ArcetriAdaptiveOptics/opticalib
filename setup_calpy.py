import os
import sys
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

    # -h/--help is passed, show help message
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(docs)
        sys.exit(0)

    # --gui (no config path) → launch GUI with the default configuration
    elif len(sys.argv) == 2 and sys.argv[1] == "--gui":
        _launch_gui(config_path=None)

    # -f <path> [--gui | --create] is passed
    elif len(sys.argv) > 2 and sys.argv[1] == "-f" and sys.argv[2]:
        config_path = sys.argv[2]
        config_path = os.path.expanduser(config_path)
        # Use robust absolute path detection (works on Windows and Unix)
        if not os.path.isabs(config_path):
            current_path = os.getcwd()
            config_path = os.path.join(current_path, config_path)
        if not ".yaml" in config_path:
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)

        # --gui flag: open the graphical interface
        if "--gui" in sys.argv:
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

        if "--create" in sys.argv or "-c" in sys.argv:
            from opticalib.core.root import create_configuration_file

            create_configuration_file(config_path, data_path=True)
        try:
            if not os.path.exists(config_path):
                config_path = os.path.join(
                    os.path.dirname(config_path), "SysConfig", "configuration.yaml"
                )
            print("\n Initiating IPython Shell, importing Opticalib...\n")
            env = os.environ.copy()
            env["AOCONF"] = config_path
            # Launch IPython using the current interpreter for cross-platform compatibility
            args = [sys.executable, "-m", "IPython", "-i", init_file]
            subprocess.run(args, env=env, check=False)
        except OSError as ose:
            print(f"Error: {ose}")
            sys.exit(1)

    # -c <path> is passed
    elif (
        len(sys.argv) > 2
        and any(sys.argv[1] == "-c", sys.argv[1] == "--create")
        and sys.argv[2]
    ):
        config_path = sys.argv[2]
        config_path = os.path.expanduser(config_path)
        # Use robust absolute path detection (works on Windows and Unix)
        if not os.path.isabs(config_path):
            current_path = os.getcwd()
            config_path = os.path.join(current_path, config_path)
        if not ".yaml" in config_path:
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)
        from opticalib.core.root import create_configuration_file

        create_configuration_file(config_path, data_path=True)
        sys.exit(0)

    # no option is passed
    elif len(sys.argv) == 1:
        # Start plain IPython session with temp opticalib file loaded
        args = [sys.executable, "-m", "IPython", "-i", init_file]
        subprocess.run(args, check=False)

    # Handle invalid arguments
    else:
        print("Error: Invalid use. Use -h or --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
