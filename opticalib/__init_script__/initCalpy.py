import os
import xupy as xp
import numpy as np
import opticalib
from opticalib import dmutils
from opticalib import procedures

opt = opticalib
join = os.path.join

folders = opaths = opticalib.folders
modal_decomposer = zern = opticalib.ground.modal_decomposer
osutils = osu = opticalib.ground.osutils
analyzer = az = opticalib.analyzer
simulator = sim = opticalib.simulator
oplt = opticalib.visualization
roi = opticalib.ground.roi

ifp = dmutils.iff_processing
ifm = procedures.iff

from matplotlib.pyplot import *

try:
    ion()
except Exception:
    pass

__all__ = [
    "os",
    "xp",
    "np",
    "opticalib",
    "dmutils",
    "opt",
    "join",
    "folders",
    "opaths",
    "modal_decomposer",
    "zern",
    "osutils",
    "osu",
    "analyzer",
    "az",
    "simulator",
    "sim",
    "roi",
    "ifp",
    "ifm",
]
