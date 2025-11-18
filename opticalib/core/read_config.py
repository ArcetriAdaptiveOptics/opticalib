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

import yaml
import os as _os
import numpy as _np
from .exceptions import DeviceNotFoundError
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

_nzeroName = "numberofzeros"
_modeIdName = "modeid"
_modeAmpName = "modeamp"
_templateName = "template"
_modalBaseName = "modalbase"

_items = [_nzeroName, _modeIdName, _modeAmpName, _templateName, _modalBaseName]


def load_yaml_config(path: str = None):
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
    if path is None or path == _cfold:
        fname = _os.path.join(_cfold, yaml_config_file)
    else:
        if _iffold in path and not _iff_config_file in path:
            fname = _os.path.join(path, _iff_config_file)
        else:
            fname = path
    with open(fname, "r") as f:
        config = yaml.safe_load(f)
    return config


def dump_yaml_config(config: dict[str, _Any], path: str = None):
    """
    Writes the configuration dictionary back to the YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to write.
    bpath : str, optional
        Base path of the file to write. Default points to the configuration root folder.
    """
    if path is None or path == _cfold:
        fname = _os.path.join(_cfold, yaml_config_file)
    else:
        if _iff_config_file not in path:
            fname = _os.path.join(path, _iff_config_file)
        else:
            fname = path
    with open(fname, "w") as f:
        yaml.dump(config, f)


def getIffConfig(key: str, bpath: str = _cfold):
    """
    Reads the configuration from the YAML file for the IFF acquisition.
    The key passed is the block of information retrieved within the INFLUENCE.FUNCTIONS section.

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
    config = load_yaml_config(bpath)
    # The nested block is under INFLUENCE.FUNCTIONS in the
    # full configuration file
    # but under INFLUENCE.FUNCTIONS/IFFUNC in the IFF copied
    # config file
    try:
        cc = config["INFLUENCE.FUNCTIONS"][key]
    except KeyError:
        cc = config[key]
    nzeros = int(cc[_nzeroName])
    modeId = _parse_val(cc[_modeIdName])
    modeAmp = _parse_val(cc[_modeAmpName])
    modalBase = cc[_modalBaseName]
    template = _np.asarray(cc[_templateName])
    info = {
        "zeros": nzeros,
        "modes": modeId,
        "amplitude": modeAmp,
        "template": template,
        "modalBase": modalBase,
    }
    return info


def copyIffConfigFile(tn: str, old_path: str = _cfold):
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
    config = load_yaml_config(old_path)
    nfname = _os.path.join(_iffold, tn, "iffConfig.yaml")
    with open(nfname, "w") as f:
        yaml.dump(config["INFLUENCE.FUNCTIONS"], f)
    print(f"IFF configuration copied to {nfname.rsplit('/' + yaml_config_file, 1)[0]}")
    return nfname


def updateIffConfig(tn: str, item: str, value: _Any):
    """
    Updates the YAML configuration file for the IFF acquisition.
    The item passed is within the INFLUENCE.FUNCTIONS/IFFUNC section.

    Parameters
    ----------
    tn : str
        Tracking number of the `iffConfig.yaml` copied from the original
        `configuration.yaml` file.
    item : str
        The configuration item to update.
    value : any
        New value to update.
    """
    key = "IFFUNC"
    file = _os.path.join(_iffold, tn, _iff_config_file)
    config = load_yaml_config(file)
    if isinstance(value, _np.ndarray):
        vmax = _np.max(value)
        vmin = _np.min(value)
        if _np.array_equal(value, _np.arange(vmin, vmax + 1)):
            config[key][item] = f"np.arange({vmin}, {vmax + 1})"
        else:
            config[key][item] = str(value.tolist())
    else:
        config[key][item] = str(value)
    dump_yaml_config(config, file)


def updateConfigFile(key: str, item: str, value: _Any, bpath: str = _cfold):
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
        "updateConfigFile is deprecated. Use updateIffConfig instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _iff_config_file not in bpath:
        fname = _os.path.join(bpath, _iff_config_file)
    else:
        fname = bpath
    config = load_yaml_config(bpath)
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
    dump_yaml_config(config, bpath)


def getDmIffConfig(bpath: str = _cfold):
    """
    Retrieves the DM configuration from the YAML file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    config : dict
        The DM configuration dictionary.
    """
    config = load_yaml_config(bpath)
    try:
        return config["INFLUENCE.FUNCTIONS"]["DM"]
    except KeyError:
        return config["DM"]


def getNActs(bpath: str = _cfold):
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
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    nacts = int(dm_config["nacts"])
    return nacts


def getTiming(bpath: str = _cfold):
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
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    timing = int(dm_config["timing"])
    return timing


def getCmdDelay(bpath: str = _cfold):
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
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    cmdDelay = float(dm_config["sequentialDelay"])
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
            return eval(val, {"np": _np})
        else:
            try:
                return eval(val)
            except Exception:
                return val
    else:
        if isinstance(val, float):
            val = float(val)
        elif isinstance(val, int):
            val = int(val)
        else:
            raise ValueError(f"Value type {type(val)} could not be recognized.")
    return val


def getCamerasConfig(device_name: str = None):
    """
    Reads the cameras settings in the configuration file.

    Returns
    -------
    config : dict
        The defined cameras parameters.
    """
    config = (load_yaml_config(_cfile))["DEVICES"]["CAMERAS"]
    if device_name is not None:
        try:
            config = config[device_name]
        except KeyError:
            raise DeviceNotFoundError(device_name)
    return config


def getDmConfig(device_name: str):
    """
    Retrieves the DM address from the YAML configuration file.

    Parameters
    ----------
    device_name : str
        Name of the DM device.

    Returns
    -------
    ip : str
        DM ip address.
    port : int
        DM port.
    """
    try:
        config = (load_yaml_config(_cfile))["DEVICES"]["DEFORMABLE.MIRRORS"][
            device_name
        ]
    except KeyError:
        raise DeviceNotFoundError(device_name)
    return config


def getInterfConfig(device_name: str):
    """
    Retrieves the interferometer address from the YAML configuration file.

    Returns
    -------
    ip : str
        Interferometer ip address.
    port : int
        Interferometer port.
    """
    try:
        config = (load_yaml_config(_cfile))["DEVICES"]["INTERFEROMETER"][device_name]
    except KeyError:
        raise DeviceNotFoundError(device_name)
    return config


def getAlignmentConfig():
    """
    Reads the alignment settings in the configuration file.

    Returns
    -------
    config : class
        The alignment configuration as a class, for backwards compatibility.
    """
    config = (load_yaml_config(_cfile))["SYSTEM.ALIGNMENT"]
    config["slices"] = [slice(item["start"], item["stop"]) for item in config["slices"]]

    class alignmentConfig:
        def __init__(self, config):
            self._conf = config

        def __getattr__(self, name):
            if name in self._conf:
                return self._conf[name]
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    return alignmentConfig(config)


def getStitchingConfig():
    """
    Reads the stitching settings in the configuration file.

    Returns
    -------
    config : dict
        The defined stitching parameters.
    """
    config = (load_yaml_config(_cfile))["STITCHING"]
    return config
