import os
from os.path import join
import xupy as xp
import numpy as np
import opticalib
from opticalib import dmutils
from opticalib import analyzer as az
from opticalib.ground import osutils
from opticalib.ground import modal_decomposer

optfp = fp = opticalib.folders

zern = modal_decomposer  # alias for backward compatibility
opt = opticalib
osu = osutils

ifp = dmutils.iff_processing
ifm = dmutils.iff_module

from matplotlib.pyplot import *
ion()

__all__ = [
    "np",
    "xp",
    "opticalib",
    "os",
    "join",
    "dmutils",
    "az",
    "osutils",
    "modal_decomposer",
    "zern",
    "opt",
    "osu",
]
