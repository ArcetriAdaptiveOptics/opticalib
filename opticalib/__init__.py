"""
OPTICALIB: adaptive OPTIcs package for dm CALIBration
=====================================================

Author(s)
---------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it
- Marco Xompero: marco.xompero@inaf.it
- Matteo Menessini: matteo.menessini@inaf.it

Description
-----------
OptiCalib is a package for the control of laboratory instrumentations, like
Interferometers and Deformable Mirrors. It also provides tools for the
analysis of wavefronts and images.
"""

from __future__ import annotations

from .__version__ import __version__

from .ground.osutils import load_fits, save_fits, get_file_list, read_phasemap
from .core.root import (
    folders,
    create_configuration_file,
    set_configuration_file,
)
from .core import config
from .core import _types as typings
from .core.fitsarray import fits_array

from . import (
    analyzer,
    devices,
    ground,
    dmutils,
    simulator,
    visualization,
)
vis = visualization

from .devices.interferometer import _4DInterferometer

get_camera_settings = _4DInterferometer.get_camera_settings
get_frame_rate = _4DInterferometer.get_frame_rate
del _4DInterferometer

from .devices import *

# ---------------------------------------------------------------------------- #
# Checking configuration version compatibility with the current package version
# ---------------------------------------------------------------------------- #
_conf_version = config.get_configuration_version()
if _conf_version is not None:
    v = tuple(int(i) for i in __version__[:5].split('.'))
    cv = tuple(int(i) for i in _conf_version[:5].split('.'))
    if any(cvx < vx for vx, cvx in zip(cv, v)):
        import warnings
        
        warnings.warn(
            f"  WARNING!\nYour experiment's configuration version `{_conf_version}` "
            f"is older than the current package version `{__version__}`. "
            "Please, ensure that nothing has changed in the configuration format "
            "and, in case, update it manually.",
            UserWarning
        )
        
        del warnings

    del v, cv, _conf_version
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

__all__ = [
    "analyzer",
    "devices",
    "ground",
    "dmutils",
    "simulator",
    "visualization",
    "load_fits",
    "save_fits",
    "read_phasemap",
    "get_file_list",
    "folders",
    "create_configuration_file",
    "set_configuration_file",
    "config",
    "typings",
    "get_camera_settings",
    "get_frame_rate",
    "fits_array",
]
