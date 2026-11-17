"""
DM Utilities — deformable-mirror calibration helpers
====================================================

Stateless and semi-stateless helpers for deformable-mirror calibration:
planning an influence-function capture, reducing the acquired data,
running a flattening loop, and slaving one mirror to another.

For the *multi-step, stateful* orchestration of a full calibration run,
see :mod:`opticalib.procedures`.

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
