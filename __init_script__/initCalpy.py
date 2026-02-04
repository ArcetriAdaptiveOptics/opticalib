import opticalib
import numpy as np
import xupy as xp
from os.path import join
from opticalib import dmutils
from opticalib import analyzer as az
from opticalib.ground import osutils
from opticalib.ground import modal_decomposer

zern = modal_decomposer  # alias for backward compatibility
opt = opticalib
osu = osutils

from matplotlib.pyplot import *
#ion()

__all__ = [
    "np",
    "xp",
    "opticalib",
    "join",
    "dmutils",
    "az",
    "osutils",
    "modal_decomposer",
    "zern",
    "opt",
    "osu",
]
