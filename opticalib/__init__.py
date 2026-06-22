"""
OPTICALIB: adaptive OPTIcs package for dm CALIBration
=====================================================

Author(s)
---------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it

Description
-----------
`opticalib` is a package for the control of laboratory instrumentations, like
Interferometers and Deformable Mirrors. It also provides tools for the
analysis of wavefronts and images.

How to Use:
-----------
```python
> import opticalib
> interf = opticalib.PhaseCam('193.206.155.218', 8011)
> img = interf.acquire_map()
```
"""

from .__version__ import __version__

from .ground.osutils import load_fits, save_fits, get_file_list, read_phasemap
from .core.root import (
    folders,
    create_configuration_file,
    set_configuration_file,
)
from .core import read_config
from .core.fitsarray import fits_array
from .devices import *
from .devices.interferometer import _4DInterferometer

get_camera_settings = _4DInterferometer.get_camera_settings
get_frame_rate = _4DInterferometer.get_frame_rate

del _4DInterferometer

from . import (
    analyzer,
    devices,
    ground,
    dmutils,
    simulator,
    visualization,
)

vis = visualization

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
    "read_config",
    "get_camera_settings",
    "get_frame_rate",
    "fits_array",
]
