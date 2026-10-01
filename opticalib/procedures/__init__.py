"""
Procedures — stateful, multi-step bench operations
==================================================

Procedures are **stateful, multi-step operations** that drive real (or
simulated) hardware over time.  They are classes you instantiate,
configure, and step through — unlike the mostly stateless helpers in
:mod:`opticalib.dmutils`.

Current procedures include beam alignment, influence-function and piston
acquisition, time-series measurements, segmented-mirror phasing (SPL),
and multi-subfield capture with offline stitching.

"""

from . import iff
from .alignment import Alignment
from .phasing import SPL
from .measurements import TimeSeries

from .iff import iff_data_acquisition, piston_data_acquisition
