"""
WaveFront Sensor (WFS) devices module
=====================================

Author(s):
----------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it
- Tania Sofia Gomes Machado : tania.gomesmachado@inaf.it
"""

import numpy as _np
from ..core import root as _fn
from ..core import _types as _ot
from ..ground import geometry as _geo
from ..devices import cameras as _cam
from skimage import transform as _transform
from ..ground.logger import SystemLogger as _SL
from ._API.base_devices import BaseWavefrontSensor
from ..core.config import get_device_config as _gdc
from ..analyzer.image_processing import mode_rebinner as _modeRebinner


class Ingot(BaseWavefrontSensor):
    """
    Class for the Ingot Wavefront Sensor (IWS).

    The I-WFS is a new class of WFS customized to the 3D geometry of the Laser
    Guide Stars (LGS). It consists of a prismatic structure that splits the LGS
    beacon in regions, and depending on the prism can form between three to six
    pupils.

    Parameters
    ----------
    camera : str | CameraDevice
        The camera device to use for acquiring images. Can be a string
        representing the camera name defined in the experiment's configuration
        file or an instance of an object compatible with the ``opticalib.CameraDevice``.

    Methods
    -------
    acquire_detector(nframes) -> ImageData
        Acquires raw un-processed detector data directly from the camera sensor.
    acquire_pupils(frames, detect_pupils) -> CubeData
        Acquires the Ingot WFS pupils from the camera frames.
    acquire_map(nframes, output_type, detect_pupils) -> ImageData
        Acquires data from the Ingot WFS, either pupils or slopes.
    set_exptime(exposure_ms)
        Sets the camera sensor exposure time in milliseconds.
    get_exptime()
        Gets the current camera exposure time in milliseconds.
    """

    def __init__(self, camera: str | _ot.CameraDevice):
        """
        The constructor initializes local network bindings and matches alignment
        transformation constraints.
        """
        self._name = "Ingot"
        self._logger = _SL(the_class=__class__)

        # 1. Initialize persistent camera connection via your wrapper
        if isinstance(camera, str):
            self._camera = _cam.GigaVision(camera)
            self._config = _gdc("WFS", "INGOT")
        elif _ot.isinstance_(camera, "CameraDevice"):
            self._camera = camera
            self._config = {}

        # Sane defaults for pupil parsing parameters matching your class variables
        self.n_pup = self._config.get("n_pupils", 3)
        self.rmin = 250
        self.rmax = 260
        self.sigma_threshold = 6

        # Spatial Calibration Configuration Switches
        self._pupil_radius = self._config.get("subapertures", None)
        self._pupil_centers = self._config.get("pupils_centers", None)
        self._pupil_info = (
            _np.asarray(
                [
                    (x, y, r)
                    for (x, y), r in zip(
                        self._pupil_centers, [self._pupil_radius] * self.n_pup
                    )
                ],
                dtype=[("xc", float), ("yc", float), ("radius", float)],
            )
            if self._pupil_centers and self._pupil_radius
            else None
        )
        self._camera_binning = self._config.get("camera_binning", 1)

        # Internal configuration storage
        self.pupdata = None
        self.exposure_time = None
        self.set_exptime(
            self._config.get("camera_base_exptime", 2000)
        )  # Default exposure time from camera

    @property
    def pupil_info(self):
        return self._pupil_info.copy()

    def set_exptime(self, exposure_ms: int) -> None:
        """
        Sets the camera sensor exposure time.

        Parameters
        ----------
        exposure_ms : int
            The desired exposure time in milliseconds.
        """
        if not exposure_ms == self.exposure_time:
            self.exposure_time = exposure_ms
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
        return self.exposure_time

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

        Returns
        -------
        _ot.CubeData
            The extracted pupil images.
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
            self._detect_pupils(frames)

        pupil_images = self._extract_pupils(frames)
        return pupil_images

    def acquire_map(
        self, nframes: int = 1, output_type: str = "slopes", detect_pupils: bool = False
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
            The type of output to return. Can be either "pupils" or "slopes".
            Default is "slopes".
        detect_pupils : bool, optional
            Whether to detect pupils in the frames before extracting them.
            Default is False.

        Returns
        -------
        pupils or slopes : ImageData
            The acquired data, either pupil images or slope maps depending on
            ``output_type``.
        """
        image = self._camera.acquire_frames(nframes)
        if self._camera_binning > 1:
            image = _modeRebinner(image, self._camera_binning, "sum")

        the_output = self.acquire_pupils(frames=image, detect_pupils=detect_pupils)

        if output_type == "slopes":
            # Map gradients across normalized spatial matrices
            Sx, Sy = self._compute_slopes_from_pupils(the_output)
            the_output = _np.vstack([Sx, Sy])

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
        A = pupil_images[0, :, :]
        B = pupil_images[1, :, :]
        C = pupil_images[2, :, :]
        # Suppress/hide the warning
        _np.seterr(invalid="ignore")
        Sx = (B - C) / (A + B + C)
        Sy = A / (A + B + C)
        mask_Sx = _np.isnan(Sx)
        mask_Sy = _np.isnan(Sy)
        Sx = _np.ma.masked_array(Sx, mask=mask_Sx)
        Sy = _np.ma.masked_array(Sy, mask=mask_Sy)
        return Sx, Sy

    def _detect_pupils(self, image: _ot.ImageData):
        """
        Pupil detection algorithm using the Hough Transform to find circular
        features in the image.

        Parameters
        ----------
        image : ImageData
            The input image in which to detect pupils.

        Raises
        ------
        RuntimeError
            If not enough pupils are detected in the image.
        """
        import warnings
        from skimage import feature

        dtype = [("xc", float), ("yc", float), ("radius", float)]
        pupdata = _np.array([], dtype=dtype)

        edges = feature.canny(
            image.astype(_np.float32), sigma=self.sigma_threshold
        )  # find edges with canny algorithm
        # finding the best fitting radius
        hough_radii = _np.arange(self.rmin, self.rmax + 1, 1)
        hough_res = _transform.hough_circle(edges, hough_radii)
        # print(hough_res)
        _, cx, cy, hough_radii = _transform.hough_circle_peaks(
            hough_res,
            hough_radii,
            total_num_peaks=1,
            min_xdistance=self.rmin,
            min_ydistance=self.rmin,
        )
        # print(hough_radii)
        if hough_radii == self.rmin or hough_radii == self.rmax:
            warnings.warn(
                "Best fitting radius has reached the limit of the input search range. Consider extending the search range"
            )
        # find best matching pupils
        hough_res = _transform.hough_circle(edges, hough_radii)
        # plt.imshow(hough_res[0])
        # plt.show()
        _, cx, cy, radii = _transform.hough_circle_peaks(
            hough_res,
            hough_radii,
            total_num_peaks=self.n_pup,
            min_xdistance=self.rmin,
            min_ydistance=int(_np.sqrt(3) * self.rmin),
        )
        # print(accums, cy, cx, radii)

        if len(cx) < self.n_pup:
            raise RuntimeError(
                f"Not enough pupils detected. Found {len(cx)} pupils, expected {self.n_pup}."
            )

        for center_y, center_x, radius in zip(cy, cx, radii):
            pupdata = _np.append(
                pupdata, _np.array((center_x, center_y, radius), dtype=dtype)
            )

        temp = _np.sort(pupdata, order=["yc", "xc"])  # first pupil on the top
        temp2 = _np.sort(
            temp[1:], order=["xc"]
        )  # second pupil on the left, third on the right
        pupdata = _np.append(temp[0], temp2)
        self._pupil_info = pupdata

    def _extract_pupils(self, image: _ot.ImageData):
        """
        Pupil extraction algorithm that extracts the pupils from the input image
        based on the detected pupil information and normalizes them to the total
        pupils flux.

        Parameters
        ----------
        image : ImageData
            The input image from which to extract pupils.

        Returns
        -------
        pupils : CubeData
            The extracted pupil images as a cube.
        """

        r = int(self._pupil_info["radius"][0])
        shape = tuple([2 * r + 1] * 2)
        pupil = _geo.draw_circular_pupil(shape, r)
        pupils = _np.ma.zeros((self.n_pup, shape[0], shape[1]))
        for pp in range(self.n_pup):
            xc = int(self._pupil_info["xc"][pp])
            yc = int(self._pupil_info["yc"][pp])
            pupils[pp, :, :] = image[(yc - r) : (yc + r + 1), (xc - r) : (xc + r + 1)]
            pupils[pp, pupil] = _np.nan
            pupils[pp, :, :].mask = pupil

        for pup, angle in zip([1, 2], [60, -60]):
            pupils[pup] = _transform.rotate(pupils[pup], angle)
            pupils[pup] = _np.flipud(pupils[pup])

        pupils /= _np.sum(pupils, axis=0)
        return pupils

    def __repr__(self) -> str:
        """The string representation of the Ingot WFS object."""
        return f"{self._name}(camera={self._camera._name}, n_pupils={self.n_pup}, exposure_time={self.exposure_time}ms)"


class ShackHartmann(BaseWavefrontSensor): ...


class Pyramid(BaseWavefrontSensor): ...


class BiOEdge(BaseWavefrontSensor): ...
