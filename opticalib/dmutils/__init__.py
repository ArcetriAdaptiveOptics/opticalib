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
- `iff_preparation.py`: Module for preparing the acquisition of the Influence Functions.
- `iff_processing.py`: Module for processing the Influence Functions.
- `procedures/iff.py`: high level module for managing the acquisition of IFFs.
- `flattening.py`: module containing the procedures for flattening a DM.

"""

from . import flattening, iff_processing, slaving
from ..procedures import iff as iff_module, stitching
from .flattening import Flattening
from ..core.data_classes import FlatData, IffData
from .iff_preparation import IFFCapturePreparation

from ._misc import *

__all__ = [
    "Flattening",
    "FlatData",
    "IffData",
    "IFFCapturePreparation",
    "iff_module",
    "iff_processing",
    "flattening",
    "slaving",
    "stitching",
    "make_modal_base",
]
