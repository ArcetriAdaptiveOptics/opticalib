"""
PROCEDURES module
=================
2026

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it

Description:
------------
This module gathers all the procedures implemented in the opticalib package, 
which are the high-level routines that can be used to perform specific tasks. 
These include:
- `iff.py`: high level module for managing the acquisition of IFFs.
- `alignment.py`: high level module for managing the alignment of a DM.
- `phasing.py`: high level module for managing the phasing of a segmented mirror.
- `measurements.py`: high level module for managing the time series measurements.
"""

from . import iff
from .alignment import Alignment
from .phasing import SPL #FIXME
from .measurements import Measurements

from .iff import iff_data_acquisition, piston_data_acquisition