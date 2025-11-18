"""
DMUTILS subpackage
==================
2024

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it

Description:
------------
This subpackage contains all the utility modules concerning a Deformable Mirror,
which are its calibration and flattening.

Contents:
---------
- `iff_acquisition_preparation.py`: Module for preparing the acquisition of the Influence Functions.
- `iff_processing.py`: Module for processing the Influence Functions.
- `iff_module.py`: high level module for managing the acquisition of IFFs.
- `flattening.py`: module containing the procedures for flattening a DM.

"""

from . import flattening, iff_module, iff_processing
from .flattening import Flattening
from .iff_acquisition_preparation import IFFCapturePreparation

__all__ = [
    "Flattening",
    "IFFCapturePreparation",
    "iff_module",
    "iff_processing",
    "flattening",
]
