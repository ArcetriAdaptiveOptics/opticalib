from .image_processing import *
from .signals import *
from .timeseries import *


def _get_safe_interf_fullframe():
    from ..devices.interferometer import _4DInterferometer

    func = _4DInterferometer.into_full_frame
    return func


into_4d_full_frame = _get_safe_interf_fullframe()
