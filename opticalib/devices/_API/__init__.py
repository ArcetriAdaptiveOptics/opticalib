"""
API submodule
==============

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it

Description:
------------
This submodule contains the classes that interface with the hardware devices.

"""

from .splattAPI import SPLATTEngine
from .alpaoAPI import BaseAlpaoMirror
from .i4dAPI import I4D
from .base_devices import BaseWavefrontSensor, BaseDeformableMirror, BaseCamera
from .micAPI import BaseAdOpticaDm
from .piAPI import BasePetalMirror

__all__ = [
    "SPLATTEngine",
    "BaseAlpaoMirror",
    "I4D",
    "BaseWavefrontSensor",
    "BaseDeformableMirror",
    "BaseAdOpticaDm",
    "BasePetalMirror",
    "BaseCamera",
]
