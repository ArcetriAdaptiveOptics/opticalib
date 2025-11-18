"""
DEVICES module
==============
2025

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it

Description:
------------
This module contains the classes for interfacing the devices used in optical
benches, or in general optical devices.

Contents:
---------
- deformable_mirrors.py: Contains classes for different deformable mirrors.
  The definitions and low level interfaces to these devices are handled in the
  `_API` submodule.
- interferometer.py: Contains classes for different interferometers. The
  definitions and low level interfaces to these devices are handled in the
  `_API` submodule.
"""

from .interferometer import PhaseCam, AccuFiz
from .deformable_mirrors import SplattDm, AlpaoDm, AdOpticaDm, DP, M4AU

__all__ = ["AdOpticaDm", "PhaseCam", "AccuFiz", "SplattDm", "AlpaoDm", "DP", "M4AU"]
