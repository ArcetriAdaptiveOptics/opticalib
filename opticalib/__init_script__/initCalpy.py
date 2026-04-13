import os
import xupy as xp
import numpy as np
import opticalib
from opticalib import dmutils

opt = opticalib
join = os.path.join

folders = fp = opticalib.folders
modal_decomposer = zern = opticalib.ground.modal_decomposer
osutils = osu = opticalib.ground.osutils
analyzer = az = opticalib.analyzer
simulator = sim = opticalib.simulator

ifp = dmutils.iff_processing
ifm = dmutils.iff_module

from matplotlib.pyplot import *

ion()

__all__ = [
    "os",
    "xp",
    "np",
    "opticalib",
    "dmutils",
    "opt",
    "join",
    "folders",
    "fp",
    "modal_decomposer",
    "zern",
    "osutils",
    "osu",
    "analyzer",
    "az",
    "simulator",
    "sim",
    "ifp",
    "ifm",
]
