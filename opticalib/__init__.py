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
from . import analyzer
from .ground.osutils import load_fits, save_fits, getFileList, read_phasemap
from .core.root import (
    folders,
    create_configuration_file,
)
from .core import read_config
from .core.fitsarray import fits_array
from .devices import *
from .devices.interferometer import _4DInterferometer

getCameraSettings = _4DInterferometer.getCameraSettings
getFrameRate = _4DInterferometer.getFrameRate

del _4DInterferometer

__all__ = [
    "analyzer",
    "load_fits",
    "save_fits",
    "read_phasemap",
    "getFileList",
    "folders",
    "create_configuration_file",
    "read_config",
    "getCameraSettings",
    "getFrameRate",
    "fits_array",
]
