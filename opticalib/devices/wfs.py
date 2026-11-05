"""
Wavefront sensor drivers
========================

Drivers for wavefront sensors used on adaptive-optics benches.
"""

import numpy as _np
from ..core import _types as _ot
from ..core import exceptions as _oe
from ..devices import cameras as _cam
from ..ground.logger import SystemLogger as _SL
from ._API.wfs.pupil_detector import PupilDetector
from ..core.config import get_device_config as _gdc
from ..analyzer.image_processing import mode_rebinner as _modeRebinner


class Ingot(PupilDetector):
    r"""
    Class for the Ingot Wavefront Sensor (IWS).

    The I-WFS is a new class of WFS customized to the 3D geometry of the Laser
    Guide Stars (LGS). It consists of a prismatic structure that splits the LGS
    beacon in regions, and depending on the prism can form between three to six
    pupils.

    ```
            A           ┌----> x
          /             |             (in the `origin='upper'` convention)
        B  -->  C     y v
    ```

    The slopes are computed as:

    .. math::
        Sx = \dfrac{B - C}{A + B + C}
        Sy = \dfrac{A}{A + B + C}

    Parameters
    ----------
    camera : str | CameraDevice
        The camera device to use for acquiring images. Can be a string
        representing the camera name defined in the experiment's configuration
        file or an instance of an object compatible with the ``opticalib.CameraDevice``.
        
        If None, the one defined in the experiment's configuration will be used.
    """

    def __init__(
        self,
        name: str = "Ingot",
        camera: _ot.Optional[str | _ot.CameraDevice] = None
    ):
        """
        The constructor initializes local network bindings and matches alignment
        transformation constraints.
        """
        self._name = name
        self._logger = _SL(the_class=__class__)
        try:
            config = _gdc("WFS", name)
        except _oe.DeviceNotFoundError:
            self._logger.error(f"Device {name} not found in configuration.")
            config = {}

        # 1. Initialize persistent camera connection via your wrapper
        if isinstance(camera, str):
            self._camera = _cam.GigaVision(camera)
        elif _ot.isinstance_(camera, "CameraDevice"):
            self._camera = camera
        elif "camera" in config.keys() and config["camera"] is not None:
            camera = config.get("camera").split(":")[-1]
            self._camera = _cam.GigaVision(camera)
        else:
            raise _oe.DeviceNotFoundError(
                "No valid camera configuration found for Ingot WFS."
            )
        # 2. initialize all the needed parameters
        super().__init__(config, n_pupils=3)
        # 3. Set initial exposure time for the camera
        self.set_exptime(getattr(self, "_camera_base_exptime", 1))

    def set_exptime(self, exposure_ms: int) -> None:
        """
        Sets the camera sensor exposure time.

        Parameters
        ----------
        exposure_ms : int
            The desired exposure time in milliseconds.
        """
        if not exposure_ms == self._exptime:
            self._exptime = exposure_ms
            self._camera.set_exptime(
                exposure_ms * 1000
            )  # Convert ms to microseconds for the camera API
            self._logger.info(
                f"Ingot camera integration profile modified to: {exposure_ms}ms"
            )

    def get_exptime(self) -> int:
        """
        Get the current camera exposure time in milliseconds.

        Returns
        -------
        exposure_ms : int
            The current exposure time in milliseconds.
        """
        return self._exptime

    def acquire_detector(self, nframes: int = 1) -> _ot.ImageData:
        """
        Acquires raw un-processed detector data directly from the camera sensor,
        leaving geometry masking and sub-pupil transformations bypassed.

        Parameters
        ----------
        nframes : int, optional
            The number of frames to acquire from the camera. Default is 1.

        Returns
        -------
        frame : ImageData
            The raw detector data as an image or cube, depending on the number
            of frames.
        """
        return self._camera.acquire_frames(nframes)

    def acquire_pupils(
        self,
        frames: _ot.ImageData | _ot.CubeData | list[_ot.ImageData] | int,
        detect_pupils: bool = False,
        pupil_order: str = "+x+y",
    ) -> _ot.CubeData:
        """
        Function for acquiring the Ingot WFS pupils.

        It acquires a frame from the camera and extract the pupils in it by using
        the defined pupils informations. If ``detect_pupils`` is set to True,
        it will first detect the pupils in the frame and then extract them.

        Parameters
        ----------
        frames : ImageData | CubeData | list[ImageData] | int
            The input frames to process. If an integer is provided, it will
            acquire that many frames from the camera.
        detect_pupils : bool, optional
            Whether to detect pupils in the frames before extracting them. Default is False.
        pupil_order : str, optional
            The order in which to sort the detected pupils.
            The sign indicates the direction of sorting ('+' for ascending, '-'
            for descending), while the order of the axes ('x' or 'y') indicates
            the primary and secondary sorting directions.

            Default is '+x+y', which sorts left-to-right, top-to-bottom.

        Returns
        -------
        pupil_images : CubeData
            The extracted pupil images in the cube format [n_pup, height, width].
        """
        if isinstance(frames, int):
            frames = self._camera.acquire_frames(frames)
            frames = (
                _modeRebinner(frames, self._camera_binning, "sum")
                if self._camera_binning > 1
                else frames
            )

        # Execute pupil detection step (if needed)
        if detect_pupils:
            self._detect_pupils(frames, pupil_order=pupil_order)

        pupil_images = self._extract_pupils(frames)
        return pupil_images

    def acquire_map(
        self,
        nframes: int = 1,
        output_type: str = "slopes",
        detect_pupils: bool = False,
        pupil_order: str = "+y+x",
    ) -> _ot.ImageData:
        """
        Acquires data from the Ingot WFS.

        It can either acquire the pupils, making it equal to the ``acquire_pupils``
        method, or compute the slopes from the pupils, depending on the
        ``output_type`` parameter.

        Parameters
        ----------
        nframes : int, optional
            The number of frames to acquire from the camera. Default is 1.
        output_type : str, optional
            The type of output to return. Can be either "pupils" (equivalent to
            ``acquire_pupils``) or "slopes".

            Default is "slopes".
        detect_pupils : bool, optional
            Whether to detect pupils in the frames before extracting them.
            Default is False.
        pupil_order : str, optional
            The order in which to sort the detected pupils.
            The sign indicates the direction of sorting ('+' for ascending, '-'
            for descending), while the order of the axes ('x' or 'y') indicates
            the primary and secondary sorting directions.

            Default is '+y+x', which sorts top-to-bottom, left-to-right (origin 
            lower-left).

        Returns
        -------
        pupils or slopes : ImageData
            The acquired data, either pupil images or slope maps depending on
            ``output_type``.
        """
        image = self._camera.acquire_frames(nframes)
        if self._camera_binning > 1:
            image = _modeRebinner(image, self._camera_binning, "sum")

        the_output = self.acquire_pupils(
            frames=image, detect_pupils=detect_pupils, pupil_order=pupil_order
        )

        if output_type == "slopes":
            Sx, Sy = self._compute_slopes_from_pupils(the_output)
            the_output = _np.ma.vstack([Sx, Sy])

        return the_output

    def _compute_slopes_from_pupils(
        self, pupil_images: _ot.CubeData
    ) -> tuple[_ot.ImageData, _ot.ImageData]:
        """
        Algorithm to compute the slopes from the pupil images.

        Given the the pupil images:
        ```
            A
          B   C
        ```
        The slopes are computed as:
        .. math::
            Sx = \frac{B - C}{A + B + C}
            Sy = \frac{A}{A + B + C}

        Parameters
        ----------
        pupil_images : ImageData
            The input pupil images from which to compute the slopes.

        Returns
        -------
        Sx, Sy : ImageData, ImageData
            The computed slopes Sx and Sy.
        """
        with _np.errstate(invalid="ignore"):
            A = pupil_images[0, :, :]
            B = pupil_images[1, :, :]
            C = pupil_images[2, :, :]
            I_tot = A + B + C
            Sx = (B - C) / I_tot
            Sy = A / I_tot
            mask_Sx = _np.isnan(Sx)
            mask_Sy = _np.isnan(Sy)
            Sx = _np.ma.masked_array(Sx, mask=mask_Sx)
            Sy = _np.ma.masked_array(Sy, mask=mask_Sy)
        return Sx, Sy

    def __repr__(self) -> str:
        """The string representation of the Ingot WFS object."""
        return (
            f"{__class__.__name__}(camera={self._camera.__repr__()}, n_pupils={self.n_pupils})"
        )


class Pyramid(PupilDetector):
    r"""
    Pyramid Wavefront Sensor (PWFS) with four circular pupils.

    Pupil geometry follows the ArcetriLAB ``PupilDesc`` approach 
    (detect / FITS save-load).

    The pupil order is assumed to be ``-x+y`` (A=TR, B=TL, C=BR, D=BL):

    ```
        B <- A      ┌----> x
          \         |           (in the `origin='upper'` convention)
        D <- C    y v
    ```

    And slopes are computed as:
    
    .. math::
        Sx = \dfrac{A + B - C - D}{A + B + C + D}
        Sy = \dfrac{B + C - A - D}{A + B + C + D}

    The order can be changed by modifying the pupil 

    Parameters
    ----------
    camera : str | CameraDevice
        The camera device to use for acquiring images. Can be a string
        representing the camera name defined in the experiment's configuration
        file or an instance of an object compatible with the ``opticalib.CameraDevice``.
        
        If None, the one defined in the experiment's configuration will be used.
    """

    def __init__(
        self,
        name: str = "Pyramid",
        camera: _ot.Optional[str | _ot.CameraDevice] = None,
    ):
        """
        The constructor initializes local network bindings and matches alignment
        transformation constraints.
        """
        self._name = name
        self._logger = _SL(the_class=__class__)
        try:
            config = _gdc("WFS", name)
        except _oe.DeviceNotFoundError:
            self._logger.error(f"Device {name} not found in configuration.")
            config = {}

        # 1. Initialize persistent camera connection via your wrapper
        if isinstance(camera, str):
            self._camera = _cam.GigaVision(camera)
        elif _ot.isinstance_(camera, "CameraDevice"):
            self._camera = camera
        elif "camera" in config.keys() and config["camera"] is not None:
            camera = config.get("camera").split(":")[-1]
            self._camera = _cam.GigaVision(camera)
        else:
            raise _oe.DeviceNotFoundError(
                "No valid camera configuration found for Pyramid WFS."
            )
        # 2. initialize all the needed parameters
        super().__init__(config, n_pupils=4)
        # 3. Set initial exposure time for the camera
        self.set_exptime(getattr(self, "_camera_base_exptime", 1))

    def set_exptime(self, exposure_ms: int | float) -> None:
        """
        Set the camera exposure time.

        Parameters
        ----------
        exposure_ms : int | float
            Exposure time in milliseconds.
        """
        if not exposure_ms == self._exptime:
            self._exptime = exposure_ms
            self._camera.set_exptime(float(exposure_ms) * 1000.0)
            self._logger.info(
                f"Pyramid camera integration profile modified to: {exposure_ms}ms"
            )

    def get_exptime(self) -> int | float:
        """Return the configured exposure time in milliseconds."""
        return self._exptime

    def set_fps(self, fps: int | float) -> None:
        """
        Set the camera frame rate.

        Parameters
        ----------
        fps : int | float
            Frame rate in Hz.
        """
        if not fps == self._camera_fps:
            self._camera_fps = float(fps)
            try:
                if hasattr(self._camera, "set_fps"):
                    self._camera.set_fps(self._camera_fps)
                else:
                    self._camera._cam.setParameter("fps", self._camera_fps)
            except Exception as e:
                self._logger.error(f"Failed to set Pyramid camera frame rate: {e}")
            self._logger.info(f"Pyramid camera frame rate set to: {self._camera_fps} Hz")

    def get_fps(self) -> float | None:
        """Return the configured camera frame rate in Hz."""
        return self._camera_fps or (1000.0/self._exptime)

    def acquire_detector(self, nframes: int = 1) -> _ot.ImageData:
        """
        Acquire raw detector frames (no pupil / slope processing).

        Parameters
        ----------
        nframes : int, optional
            Number of frames to average. Default is 1.
        """
        return self._camera.acquire_frames(nframes)

    def acquire_pupils(
        self,
        frames: _ot.ImageData | _ot.CubeData | list[_ot.ImageData] | int,
        detect_pupils: bool = False,
        pupil_order: str = "-x+y",
    ) -> _ot.CubeData:
        """
        Function for acquiring the Pyramid WFS pupils.

        It acquires a frame from the camera and extract the pupils in it by using
        the defined pupils informations. If ``detect_pupils`` is set to True,
        it will first detect the pupils in the frame and then extract them.

        Parameters
        ----------
        frames : ImageData | CubeData | list[ImageData] | int
            The input frames to process. If an integer is provided, it will
            acquire that many frames from the camera.
        detect_pupils : bool, optional
            Whether to detect pupils in the frames before extracting them. Default is False.
        pupil_order : str, optional
            The order in which to sort the detected pupils.
            The sign indicates the direction of sorting ('+' for ascending, '-'
            for descending), while the order of the axes ('x' or 'y') indicates
            the primary and secondary sorting directions.

            Default is '-x+y'.

        Returns
        -------
        pupil_images : CubeData
            The extracted pupil images in the cube format [n_pup, height, width].
        """
        if isinstance(frames, int):
            frames = self._camera.acquire_frames(frames)
            frames = (
                _modeRebinner(frames, self._camera_binning, "sum")
                if self._camera_binning > 1
                else frames
            )

        # Execute pupil detection step (if needed)
        if detect_pupils:
            self._detect_pupils(frames, pupil_order=pupil_order)

        pupil_images = self._extract_pupils(frames)
        return pupil_images

    def acquire_map(
        self,
        nframes: int = 1,
        output_type: str = "slopes",
        detect_pupils: bool = False,
        pupil_order: str = "-x+y",
    ) -> _ot.ImageData:
        """
        Acquires data from the Pyramid WFS.

        It can either acquire the pupils, making it equal to the ``acquire_pupils``
        method, or compute the slopes from the pupils, depending on the
        ``output_type`` parameter.

        Parameters
        ----------
        nframes : int, optional
            The number of frames to acquire from the camera. Default is 1.
        output_type : str, optional
            The type of output to return. Can be either "pupils" (equivalent to
            ``acquire_pupils``) or "slopes".

            Default is "slopes".
        detect_pupils : bool, optional
            Whether to detect pupils in the frames before extracting them.
            Default is False.
        pupil_order : str, optional
            The order in which to sort the detected pupils.
            The sign indicates the direction of sorting ('+' for ascending, '-'
            for descending), while the order of the axes ('x' or 'y') indicates
            the primary and secondary sorting directions.

            Default is '-x+y', which sorts right-to-left, bottom-to-top.

        Returns
        -------
        pupils or slopes : ImageData
            The acquired data, either pupil images or slope maps depending on
            ``output_type``.
        """
        image = self._camera.acquire_frames(nframes)
        if self._camera_binning > 1:
            image = _modeRebinner(image, self._camera_binning, "sum")

        the_output = self.acquire_pupils(
            frames=image, detect_pupils=detect_pupils, pupil_order=pupil_order
        )

        if output_type == "slopes":
            Sx, Sy = self._compute_slopes_from_pupils(the_output)
            the_output = _np.ma.vstack([Sx, Sy])

        return the_output     

    def _compute_slopes_from_pupils(
        self, 
        pupil_images: _ot.CubeData,
    ) -> _np.ndarray:
        """
        Compute slopes from pupil images using the specified pupil geometry.
        """
        with _np.errstate(invalid="ignore"):
            A = pupil_images[0]
            B = pupil_images[1]
            C = pupil_images[2]
            D = pupil_images[3]

            I_tot = A + B + C + D
            Sx = (A + B - C - D) / I_tot
            Sy = (B + D - A - C) / I_tot

            mask_Sx = _np.isnan(Sx)
            mask_Sy = _np.isnan(Sy)
            Sx = _np.ma.masked_array(Sx, mask=mask_Sx)
            Sy = _np.ma.masked_array(Sy, mask=mask_Sy)

        return Sx, Sy

    def __repr__(self):
        return (
            f"{__class__.__name__}(camera={self._camera.__repr__()}, n_pupils={self.n_pupils})"
        )


class ShackHartmann(): ...
