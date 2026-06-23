"""
This module provides utilities for reading, writing, and updating YAML configuration files
used in the opticalib system. It supports configuration management for devices such as
deformable mirrors and interferometers, as well as acquisition and alignment settings.

Features
--------
- Load and dump YAML configuration files.
- Retrieve and update configuration blocks for IFF acquisition, DM devices, and interferometers.
- Copy configuration files for record keeping.
- Parse and convert configuration values, including numpy arrays.
- Access alignment and stitching settings as structured objects.

Author(s)
---------
- Pietro Ferraiuolo: written in 2025
- Runa Briguglio
"""

import os as _os
import re as _re
from ast import literal_eval as _literal_eval

import numpy as _np
import yaml

from .exceptions import DeviceNotFoundError
from .decorators import expand_list_arguments as _ela
from typing import Any as _Any

global _cfold
global _iffold
global _cfile


def _update_imports():
    global _cfold
    global _iffold
    global _cfile
    from .root import (
        CONFIGURATION_FOLDER,
        IFFUNCTIONS_ROOT_FOLDER,
        CONFIGURATION_FILE,
    )

    _cfold = CONFIGURATION_FOLDER
    _iffold = IFFUNCTIONS_ROOT_FOLDER
    _cfile = CONFIGURATION_FILE


_update_imports()

yaml_config_file = "configuration.yaml"
_iff_config_file = "iffConfig.yaml"

_TZERO = "trailing_zeros"
_MODEID = "modes_list"
_MODEAMP = "amplitude"
_TEMPLATE = "template"
_MODALBASE = "modal_base"
_SHUFFLE = "shuffle"
_NREP = "n_repetitions"

_items = [_TZERO, _MODEID, _MODEAMP, _TEMPLATE, _MODALBASE, _SHUFFLE, _NREP]
_IFSECTIONS = ["TRIGGER", "REGISTRATION", "IFFUNC"]


def _resolve_config_path(path: str | None = None) -> str:
    """
    Resolve the absolute path of a configuration file.

    Parameters
    ----------
    path : str | None, optional
        Input path that can be either a configuration file or a directory.
        If ``None``, the main configuration file is used.

    Returns
    -------
    str
        Absolute path of the configuration file to read or write.
    """
    if path is None:
        return _cfile

    if path == _cfold:
        return _os.path.join(_cfold, yaml_config_file)

    if _os.path.isdir(path):
        if _iffold in _os.path.abspath(path):
            return _os.path.join(path, _iff_config_file)
        return _os.path.join(path, yaml_config_file)

    if _iffold in _os.path.abspath(path) and not path.endswith(_iff_config_file):
        return _os.path.join(path, _iff_config_file)
    return path


def load(path: str | None = None) -> dict[str, _Any]:
    """
    Loads the YAML configuration file.

    Parameters
    ----------
    path : str, optional
        Base path of the file to read. Default points to the configuration root folder.

    Returns
    -------
    config : dict
        The configuration dictionary.
    """
    fname = _resolve_config_path(path)
    with open(fname, "r") as f:
        config = yaml.safe_load(f) or {}
    return config


def dump(config: dict[str, _Any], path: str | None = None) -> None:
    """
    Writes the configuration dictionary back to the YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to write.
    bpath : str, optional
        Base path of the file to write. Default points to the configuration root folder.
    """
    fname = _resolve_config_path(path)
    with open(fname, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_yaml_config(path: str | None = None) -> dict[str, _Any]:
    """
    Backward-compatible wrapper for :func:`load`.
    """
    return load(path)


def dump_yaml_config(config: dict[str, _Any], path: str | None = None) -> None:
    """
    Backward-compatible wrapper for :func:`dump`.
    """
    dump(config, path)


def _get_section_config(
    section: str,
    sub_section: str | None = None,
    path: str | None = None,
) -> dict[str, _Any]:
    """
    Read a section (and optional subsection) from a configuration file.

    Parameters
    ----------
    section : str
        Top-level section name in the configuration file.
    sub_section : str | None, optional
        Nested section name under *section*.
    path : str | None, optional
        Configuration file or folder path.

    Returns
    -------
    dict[str, Any]
        Requested section configuration.
    """
    config = load(path)
    if section not in config:
        raise KeyError(f"Configuration section `{section}` not found in the YAML file")
    if sub_section is None:
        return config[section]
    section_config = config[section]
    if sub_section not in section_config:
        raise KeyError(
            f"Configuration subsection `{sub_section}` not found in section "
            f"`{section}`."
        )
    return section_config[sub_section]


def get_section_config(section: str, sub_section: str | None = None) -> dict[str, _Any]:
    """
    Read a section (and optional subsection) from ``configuration.yaml``.

    Parameters
    ----------
    section : str
        Top-level section name in the main configuration file.
    sub_section : str | None, optional
        Nested section name under *section*.

    Returns
    -------
    dict[str, Any]
        Requested section configuration.
    """
    return _get_section_config(section=section, sub_section=sub_section, path=_cfile)


def get_device_config(device_type: str, device_name: str | None = None):
    """
    Retrieves the device configuration from the YAML configuration file.

    Parameters
    ----------
    device_type : str
        Type of the device (e.g., 'CAMERAS', 'DEFORMABLE.MIRRORS', 'WFS').
    device_name : str | None
        Name of the device. If None, returns the configuration for all devices of the specified type.

    Returns
    -------
    config : dict
        The device configuration dictionary.
    """
    try:
        config = get_section_config("DEVICES", device_type)
        if device_name is not None:
            config = config[device_name]
    except KeyError:
        raise DeviceNotFoundError(
            f"{device_type} '{device_name}' not found in configuration."
        )
    return config


def get_phasing_config():
    """
    Retrieves the phasing configuration from the YAML configuration file.

    Returns
    -------
    config : dict
        The phasing configuration dictionary.
    """
    try:
        return get_section_config("PHASING")
    except KeyError:
        raise KeyError("Phasing configuration not found in the YAML file.")


def get_iff_config(key: str|None, bpath: str = _cfold):
    """
    Reads the configuration from the YAML file for the IFF acquisition.
    The key passed is the block of information retrieved within the 
    INFLUENCE.FUNCTIONS section.
    
    If ``key=None``, the function returns the entire INFLUENCE.FUNCTIONS section.

    Parameters
    ----------
    key : str
        Key value of the block of information to read. Can be
            - 'TRIGGER'
            - 'REGISTRATION'
            - 'IFFUNC'
    bpath : str, optional
        Base path of the file to read. Default points to the configuration root folder.

    Returns
    -------
    info : dict
        A dictionary containing the configuration info:
            - zeros
            - modes
            - amplitude
            - template
            - modalBase
    """
    # The nested block is under INFLUENCE.FUNCTIONS in the
    # full configuration file
    # but under INFLUENCE.FUNCTIONS/IFFUNC in the IFF copied
    # config file
    try:
        cc = _get_section_config(
            section="INFLUENCE.FUNCTIONS",
            sub_section=key,
            path=bpath,
        )
    except KeyError:
        try:
            cc = _get_section_config(section='INFLUENCE.FUNCTIONS', path=bpath)
        except KeyError:
            # Assuming this loading is the `iffConfig.yaml` file copied during 
            # the IFF acquisition
            cc = load(bpath)

    for section in cc.keys():
        if key is None:
            if section in _IFSECTIONS:
                for k, vals in cc[section].items():
                    cc[section][k] = _parse_val(vals)
        else:
            cc[section] = _parse_val(cc[section])

    return cc


def copy_iff_config_file(tn: str, old_path: str = _cfold):
    """
    Copies the YAML configuration file to the new folder for record keeping of the
    configuration used on data acquisition.

    Parameters
    ----------
    tn : str
        Tracking number for the new data.
    old_path : str, optional
        Base path where the YAML configuration file resides.

    Returns
    -------
    res : str
        Path where the file was copied.
    """
    config = _get_section_config(section="INFLUENCE.FUNCTIONS", path=old_path)
    nfname = _os.path.join(_iffold, tn, "iffConfig.yaml")
    with open(nfname, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"IFF configuration copied to {nfname.rsplit('/' + yaml_config_file, 1)[0]}")
    return nfname

@_ela(['item', 'value'])
def update_iff_config(tn: str, item: str|list[str], value: _Any|list[_Any]):
    """
    Updates the YAML configuration file for the IFF acquisition.
    The item passed is within the INFLUENCE.FUNCTIONS/IFFUNC section.

    Parameters
    ----------
    tn : str
        Tracking number of the `iffConfig.yaml` copied from the original
        `configuration.yaml` file.
    item : str, list of str
        The configuration item(s) to update.
    value : any, list of any
        New value(s) to update.
    """
    key = "IFFUNC"
    file = _os.path.join(_iffold, tn, _iff_config_file)
    config = load(file)
    if isinstance(value, (_np.ndarray, list)):
        vmax = _np.max(value)
        vmin = _np.min(value)
        step = value[1] - value[0] if len(value) > 1 else 1
        if step == 0.0:
            config[key][item] = f"[{','.join(str(v) for v in [vmax]*len(value))}]"
        elif _np.array_equal(value, _np.arange(vmin, vmax + 1, step)):
            config[key][item] = f"np.arange({vmin}, {vmax + 1}, {step})"
        else:
            config[key][item] = f"[{','.join(str(v) for v in value)}]"
    else:
        config[key][item] = str(value)
    dump(config, file)


def update_config_file(key: str, item: str, value: _Any, bpath: str = _cfold):
    """
    Updates the YAML configuration file for the IFF acquisition.
    The key passed is within the INFLUENCE.FUNCTIONS section.

    Parameters
    ----------
    key : str
        Key of the block to update (e.g., 'TRIGGER', 'REGISTRATION', 'IFFUNC').
    item : str
        The configuration item to update.
    value : any
        New value to update.
    bpath : str, optional
        Base path of the configuration file.
    """
    import warnings

    warnings.warn(
        "update_config_file is deprecated. Use update_iff_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _iff_config_file not in bpath:
        fname = _os.path.join(bpath, _iff_config_file)
    else:
        fname = bpath
    config = load(bpath)
    if key not in config["INFLUENCE.FUNCTIONS"]:
        raise KeyError(f"Configuration section `{key}` not found in the YAML file")
    if item not in _items:
        raise KeyError(f"Item `{item}` not found in the configuration file")
    # Update the value (convert np.ndarray to list if needed)
    if isinstance(value, _np.ndarray):
        vmax = _np.max(value)
        vmin = _np.min(value)
        if _np.array_equal(value, _np.arange(vmin, vmax + 1)):
            config["INFLUENCE.FUNCTIONS"][key][
                item
            ] = f'"np.arange({vmin}, {vmax + 1})"'
        else:
            config["INFLUENCE.FUNCTIONS"][key][item] = str(value.tolist())
    else:
        config["INFLUENCE.FUNCTIONS"][key][item] = str(value)
    dump(config, bpath)


def get_n_acts(bpath: str = _cfold):
    """
    Retrieves the number of actuators from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    nacts : int
        Number of DM actuators.
    """
    dm_config = _get_section_config(
        section="INFLUENCE.FUNCTIONS",
        sub_section="DM",
        path=bpath,
    )
    nacts = int(dm_config["nacts"])
    return nacts


def get_timing(bpath: str = _cfold):
    """
    Retrieves timing information from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    timing : int
        Timing used for synchronization.
    """
    dm_config = _get_section_config(
        section="INFLUENCE.FUNCTIONS",
        sub_section="DM",
        path=bpath,
    )
    timing = int(dm_config["timing"])
    return timing


def get_cmd_delay(bpath: str = _cfold):
    """
    Retrieves the command delay from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    cmdDelay : float
        Command delay for the interferometer synchronization.
    """
    dm_config = _get_section_config(
        section="INFLUENCE.FUNCTIONS",
        sub_section="DM",
        path=bpath,
    )
    if "sequentialDelay" in dm_config:
        cmdDelay = float(dm_config["sequentialDelay"])
    elif "delay" in dm_config:
        cmdDelay = float(dm_config["delay"])
    elif "triggeredMode" in dm_config and "cmdDelay" in dm_config["triggeredMode"]:
        cmdDelay = float(dm_config["triggeredMode"]["cmdDelay"])
    else:
        raise KeyError("Command delay not found in INFLUENCE.FUNCTIONS/DM.")
    return cmdDelay


def _parse_val(val: _Any):
    """
    Parses a value from the YAML configuration file.

    Parameters
    ----------
    val : str
        Value to parse.

    Returns
    -------
    parsed_val : int or float
        Parsed value, either as an integer or a float.
    """
    if isinstance(val, list):
        return _np.array(val)
    if isinstance(val, str):
        if val.startswith("np.arange"):
            match = _re.fullmatch(
                r"np\.arange\(\s*([^,]+)\s*,\s*([^,]+)\s*(?:,\s*([^)]+)\s*)?\)",
                val,
            )
            if match is None:
                raise ValueError(f"Malformed np.arange expression: {val}")
            start = _literal_eval(match.group(1))
            stop = _literal_eval(match.group(2))
            step = _literal_eval(match.group(3)) if match.group(3) is not None else 1
            return _np.arange(start, stop, step)
        try:
            return _literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    else:
        if isinstance(val, float):
            val = float(val)
        elif isinstance(val, int):
            val = int(val)
        else:
            raise ValueError(f"Value type {type(val)} could not be recognized.")
    return val


def get_cameras_config(device_name: str = None):
    """
    Reads the cameras settings in the configuration file.

    Returns
    -------
    config : dict
        The defined cameras parameters.
    """
    config = get_section_config("DEVICES", "CAMERAS")
    if device_name is not None:
        try:
            config = config[device_name]
        except KeyError:
            raise DeviceNotFoundError(device_name)
    return config


def get_dm_config(device_name: str) -> dict[str, _Any]:
    """
    Retrieves the DM address from the YAML configuration file.

    Parameters
    ----------
    device_name : str
        Name of the DM device.

    Returns
    -------
    config : dict
        DM dictionary containing the defined requested device in the configuration file.
    """
    try:
        config = get_section_config("DEVICES", "DEFORMABLE.MIRRORS")[device_name]
    except KeyError:
        raise DeviceNotFoundError(device_name)
    return config


def get_interf_config(device_name: str):
    """
    Retrieves the wavefront sensor address from the YAML configuration file.

    Returns
    -------
    ip : str
        Wavefront sensor ip address.
    port : int
        Wavefront sensor port.
    """
    try:
        config = get_section_config("DEVICES", "INTERFEROMETERS")[device_name]
    except KeyError:
        raise DeviceNotFoundError(device_name)
    return config


def get_alignment_config():
    """
    Reads the alignment settings in the configuration file.

    Returns
    -------
    config : class
        The alignment configuration as a class, for backwards compatibility.
    """
    config = get_section_config("SYSTEM.ALIGNMENT")
    config["slices"] = [slice(item["start"], item["stop"]) for item in config["slices"]]

    class AlignmentConfig:
        def __init__(self, config):
            self._conf = config

        def __getattr__(self, name):
            if name in self._conf:
                return self._conf[name]
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    return AlignmentConfig(config)


def get_stitching_config():
    """
    Reads the stitching settings in the configuration file.

    Returns
    -------
    config : dict
        The defined stitching parameters.
    """
    config = get_section_config("STITCHING")
    return config
