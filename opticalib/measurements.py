import time as _t
from os.path import join as _join
from .ground.osutils import (
    create_data_folder as _cdf,
    newtn as _ts,
    save_fits as _save_fits,
)
from .core.root import OPD_SERIES_ROOT_FOLDER as _ops
from . import typings as _ot


class Measurements:  # TODO: Change name
    """
    Class to handle optical time-series measurements.

    This class handles the acquisition of temporized measurements, using a camera
    device, which can be a CCD or an interferometer.

    Parameters
    ----------
    camera : InterferometerDevice | CameraDevice
        An instance of an interferometer device or a camera device to be used
        for measurements.
    devices : GenericDevice | list of GenericDevice, optional
        A single device or a list of devices to be used in conjunction with the
        camera for measurements (for movements or telemetry data).
    """

    def __init__(
        self,
        camera: _ot.InterferometerDevice | _ot.CameraDevice,
        devices: _ot.Optional[_ot.GenericDevice | list[_ot.GenericDevice]] = None,
    ) -> None:
        """The constructor"""
        self._camera = camera
        self._devices = devices

        if _ot.isinstance_(self._camera, "InterferometerDevice"):
            self._acquire_func = self._camera.acquire_map
        elif _ot.isinstance_(self._camera, "CameraDevice"):
            self._acquire_func = self._camera.acquire_frames
        else:
            raise TypeError("Unsupported camera device type.")

    def acquire_time_series(self, nframes: int, delay: float = 0) -> None:
        """
        Acquires a time series of measurements from the camera device.

        Parameters
        ----------
        nframes : int
            The number of frames to acquire in the time series.
        delay : float, optional
            The delay in seconds between consecutive frame acquisitions.

            The default is 0.

        Returns
        -------
        tn: str
            The Tracking Number of the data in the OPDSeries folder.
        """
        path = _cdf(_ops)
        for _ in range(nframes):
            tn = _ts()
            img = self._acquire_func()
            _save_fits(_join(path, tn), img)
            if delay > 0:
                _t.sleep(delay)
        return path.split("/")[-1]
