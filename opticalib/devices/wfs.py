"""
WaveFront Sensor (WFS) devices module
=====================================

Author(s):
----------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it
- Tania Sofia Gomes Machado : tania.gomesmachado@inaf.it
"""

import os as _os
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


class Pyramid(BaseWavefrontSensor):
    """
    Pyramid Wavefront Sensor (PWFS) with four circular pupils.

    Uses a C-Blue (pysilico) camera by default. Pupil geometry follows the
    ArcetriLAB ``PupilDesc`` approach (detect / FITS save-load).

    Slopes are Specula 1D vectors ``concatenate([sx, sy])`` of length
    ``2 * n_subap`` from the four ``indpup`` intensities::

        Sx = (A + B - C - D) / (A + B + C + D)
        Sy = (B + C - A - D) / (A + B + C + D)

    with Specula pupil order ``A=TR, B=TL, C=BL, D=BR``.
    ``acquire_map(..., output_type="slopes2d")`` remaps that vector into a
    side-by-side 2D masked map via ``slopes2d`` (inverse: ``slopes1d``).

    Parameters
    ----------
    camera : str | CameraDevice
        Camera name from the experiment configuration, or an existing
        ``CameraDevice`` instance.
    """

    # ArcetriLAB ncoords order is BL, BR, TR, TL → Specula A,B,C,D = TR,TL,BL,BR
    _SPECULA_NCOORD_ORDER = (2, 3, 0, 1)

    def __init__(self, camera: str | _ot.CameraDevice):
        from ._pyramid_pupils import PyramidPupilData

        self._name = "Pyramid"
        self._logger = _SL(the_class=__class__)
        self._PyramidPupilData = PyramidPupilData

        if isinstance(camera, str):
            self._camera = _cam.CBlue(camera)
            self._config = _gdc("WFS", "PYRAMID")
        elif _ot.isinstance_(camera, "CameraDevice"):
            self._camera = camera
            self._config = {}
        else:
            raise TypeError(
                "camera must be a config name (str) or a CameraDevice instance"
            )

        self.n_pup = int(self._config.get("n_pupils", 4))
        self.thr_value = float(self._config.get("thr_value", 0))
        self.detect_threshold = float(self._config.get("detect_threshold", 1000))
        # Default shlike = per-subap denom (Specula 1D slopes)
        self.norm = str(self._config.get("norm", "shlike")).lower()
        norm_factor = self._config.get("norm_factor", None)
        self.norm_factor = None if norm_factor in (None, "null", "") else float(norm_factor)

        self._pupil_data = None
        self.pupdata = None
        # Exposure / fps come from CAMERAS:CBlue (applied when CBlue connects)
        self.exposure_time = None
        self.fps = None
        if hasattr(self._camera, "get_exptime") and hasattr(self._camera, "get_fps"):
            try:
                # CBlue get_exptime is in microseconds
                self.exposure_time = float(self._camera.get_exptime()) / 1000.0
            except Exception:
                self.exposure_time = None
            try:
                self.fps = self._camera.get_fps()
            except Exception:
                self.fps = None

        pupils_file = self._config.get("pupils_file", "") or ""
        if pupils_file and _os.path.isfile(pupils_file):
            self.load_pupils(pupils_file)

    @property
    def pupil_info(self):
        """Return a copy of the loaded / detected ``PyramidPupilData``, or None."""
        return self._pupil_data

    @property
    def indpup(self):
        """ArcetriLAB-ordered pupil index lists, or None if pupils are unset."""
        if self._pupil_data is None:
            return None
        return self._pupil_data.indpup

    def set_exptime(self, exposure_ms: int | float) -> None:
        """
        Set the camera exposure time.

        Parameters
        ----------
        exposure_ms : int | float
            Exposure time in milliseconds.
        """
        if not exposure_ms == self.exposure_time:
            self.exposure_time = exposure_ms
            self._camera.set_exptime(float(exposure_ms) * 1000.0)
            self._logger.info(
                f"Pyramid camera integration profile modified to: {exposure_ms}ms"
            )

    def get_exptime(self) -> int | float:
        """Return the configured exposure time in milliseconds."""
        return self.exposure_time

    def set_fps(self, fps: int | float) -> None:
        """
        Set the camera frame rate.

        Parameters
        ----------
        fps : int | float
            Frame rate in Hz.
        """
        if not fps == self.fps:
            self.fps = float(fps)
            if hasattr(self._camera, "set_fps"):
                self._camera.set_fps(self.fps)
            else:
                self._camera._cam.setParameter("fps", self.fps)
            self._logger.info(f"Pyramid camera frame rate set to: {self.fps} Hz")

    def get_fps(self) -> float | None:
        """Return the configured camera frame rate in Hz."""
        return self.fps

    def acquire_detector(self, nframes: int = 1) -> _ot.ImageData:
        """
        Acquire raw detector frames (no pupil / slope processing).

        Parameters
        ----------
        nframes : int, optional
            Number of frames to average. Default is 1.
        """
        return self._camera.acquire_frames(nframes)

    def save_pupils(self, path: str, overwrite: bool = True) -> None:
        """
        Persist current pupil geometry to an ArcetriLAB-compatible FITS file.

        Parameters
        ----------
        path : str
            Destination FITS path.
        overwrite : bool, optional
            Overwrite an existing file. Default is True.
        """
        if self._pupil_data is None:
            raise RuntimeError("No pupil data to save; run detect_pupils first.")
        self._pupil_data.save(path, overwrite=overwrite)
        self._logger.info(f"Saved Pyramid pupil data to {path}")

    def load_pupils(self, path: str) -> None:
        """
        Load pupil geometry from an ArcetriLAB-compatible FITS file.

        Parameters
        ----------
        path : str
            Source FITS path.
        """
        self._pupil_data = self._PyramidPupilData.load(path)
        self.pupdata = self._pupil_data
        self._logger.info(
            f"Loaded Pyramid pupil data from {path} "
            f"(n_subap={self._pupil_data.n_subap})"
        )

    def _ensure_pupils(self, image: _ot.ImageData | None = None, detect: bool = False):
        if detect:
            if image is None:
                raise ValueError("An image is required when detect_pupils=True")
            self._detect_pupils(image)
        if self._pupil_data is None:
            raise RuntimeError(
                "Pyramid pupil geometry is not set. Call acquire_map/acquire_pupils "
                "with detect_pupils=True, or load_pupils(path)."
            )

    def _detect_pupils(self, image: _ot.ImageData):
        """Detect four pupils and store geometry / indpup."""
        frame = self._as_2d_frame(image)
        self._pupil_data = self._PyramidPupilData.from_img(
            frame, threshold=self.detect_threshold, verbose=False
        )
        self.pupdata = self._pupil_data
        self._logger.info(
            f"Detected {self.n_pup} pupils "
            f"(n_subap={self._pupil_data.n_subap}, avgD={self._pupil_data.avg_d:.2f})"
        )

    @staticmethod
    def _as_2d_frame(image) -> _np.ndarray:
        """Reduce a single frame or small cube to a 2-D float array."""
        arr = _np.asarray(image, dtype=_np.float64)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # Prefer averaging along the shortest axis (frame stack)
            axis = int(_np.argmin(arr.shape))
            return arr.mean(axis=axis)
        raise ValueError(f"Expected 2-D or 3-D image, got shape {arr.shape}")

    def _extract_pupils(self, image: _ot.ImageData) -> _ot.CubeData:
        """
        Crop four circular pupil images (Ingot-style), Specula order A,B,C,D.

        Returns
        -------
        CubeData
            Masked array of shape ``(4, 2*r+1, 2*r+1)``.
        """
        if self._pupil_data is None:
            raise RuntimeError("Pupil geometry is required to extract pupils.")

        frame = _np.asarray(image, dtype=_np.float64)
        r = max(1, int(round(self._pupil_data.avg_d / 2.0)))
        shape = (2 * r + 1, 2 * r + 1)
        circ_mask = _geo.draw_circular_pupil(shape, r)
        pupils = _np.ma.zeros((self.n_pup, shape[0], shape[1]), dtype=_np.float64)

        # ncoords entries are [x, y]; Specula A,B,C,D from ArcetriLAB order
        for pp, nidx in enumerate(self._SPECULA_NCOORD_ORDER):
            xc = int(round(float(self._pupil_data.ncoords[nidx][0])))
            yc = int(round(float(self._pupil_data.ncoords[nidx][1])))
            patch = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
            if patch.shape != shape:
                raise RuntimeError(
                    f"Pupil crop out of bounds at ({xc},{yc}) r={r} "
                    f"on frame {frame.shape}; got patch {patch.shape}."
                )
            pupils[pp, :, :] = patch
            pupils[pp, circ_mask] = _np.nan
            pupils[pp].mask = circ_mask

        if self.thr_value:
            pupils = _np.ma.masked_array(
                _np.where(pupils.data > self.thr_value, pupils.data - self.thr_value, 0.0),
                mask=pupils.mask,
            )
        return pupils

    def acquire_pupils(
        self,
        frames: _ot.ImageData | _ot.CubeData | list[_ot.ImageData] | int,
        detect_pupils: bool = False,
    ) -> _ot.CubeData:
        """
        Extract the four circular pupil images from camera frames.

        Parameters
        ----------
        frames : ImageData | CubeData | list | int
            Input frames, or an integer number of frames to acquire.
        detect_pupils : bool, optional
            If True, re-detect pupils on the acquired / provided frame.

        Returns
        -------
        CubeData
            Masked array of shape ``(4, 2*r+1, 2*r+1)`` in Specula order
            A, B, C, D (TR, TL, BL, BR).
        """
        if isinstance(frames, int):
            frames = self._camera.acquire_frames(frames)

        image = self._as_2d_frame(frames)
        self._ensure_pupils(image=image, detect=detect_pupils)
        return self._extract_pupils(image)

    def _require_pupils(self):
        if self._pupil_data is None:
            raise RuntimeError(
                "Pyramid pupil geometry is not set. Call acquire_map/acquire_pupils "
                "with detect_pupils=True, or load_pupils(path)."
            )
        return self._pupil_data

    def slopes2d(self, slopes, useNaN: bool = True):
        """Remap Specula 1D slopes to a 2D side-by-side masked map."""
        return self._require_pupils().slopes2d(slopes, useNaN=useNaN)

    def slopes1d(self, frame):
        """Inverse of ``slopes2d``: 2D map(s) back to Specula 1D slopes."""
        return self._require_pupils().slopes1d(frame)

    def _compute_slopes_from_frame(self, image: _ot.ImageData) -> _np.ndarray:
        """
        Specula-style 1D slopes via ``indpup`` (length ``2 * n_subap``).
        """
        if self._pupil_data is None:
            raise RuntimeError("Pupil geometry is required to compute slopes.")

        flat = _np.asarray(image, dtype=_np.float64).ravel().copy()
        flat -= self.thr_value
        flat[flat < 0] = 0.0

        ind = self._pupil_data.ind_pup_specula  # (n_subap, 4) A,B,C,D
        A = flat[ind[:, 0]]
        B = flat[ind[:, 1]]
        C = flat[ind[:, 2]]
        D = flat[ind[:, 3]]

        flux_per_subap = A + B + C + D
        n_subap = int(flux_per_subap.size)
        total_intensity = float(_np.sum(flux_per_subap))

        if self.norm == "fixed":
            if self.norm_factor is None or self.norm_factor == 0:
                raise ValueError("norm='fixed' requires a non-zero norm_factor")
            factor = 1.0 / float(self.norm_factor)
            sx = (A + B - C - D) * factor
            sy = (B + C - A - D) * factor
        elif self.norm == "shlike":
            sx = A + B - C - D
            sy = B + C - A - D
            flux_clamped = _np.where(flux_per_subap > 0, flux_per_subap, 1.0)
            sx = sx / flux_clamped
            sy = sy / flux_clamped
        else:
            if total_intensity <= 0 or n_subap == 0:
                sx = _np.zeros(n_subap, dtype=_np.float64)
                sy = _np.zeros(n_subap, dtype=_np.float64)
            else:
                factor = n_subap / total_intensity
                sx = (A + B - C - D) * factor
                sy = (B + C - A - D) * factor

        return _np.concatenate([sx, sy])

    def acquire_map(
        self,
        nframes: int = 1,
        output_type: str = "slopes2d",
        detect_pupils: bool = False,
    ) -> _ot.ImageData:
        """
        Acquire Pyramid WFS data (slopes or pupil maps).

        Parameters
        ----------
        nframes : int, optional
            Number of frames to acquire / average. Default is 1.
        output_type : str, optional
            ``'slopes2d'`` (default) remaps Specula 1D slopes to a 2D
            side-by-side ``(h, 2w)`` masked map.
            ``'pupils'`` returns the cropped ``(4, ny, nx)`` pupil cube.
            ``'slopes'`` / ``'slopes_vector'`` return Specula 1D
            ``concatenate([sx, sy])``.
        detect_pupils : bool, optional
            Re-detect pupils before processing.

        Returns
        -------
        ImageData
            Slope maps, pupil cube, or 1D slopes depending on ``output_type``.
        """
        image = self._as_2d_frame(self._camera.acquire_frames(nframes))
        self._ensure_pupils(image=image, detect=detect_pupils)

        if output_type == "pupils":
            return self._extract_pupils(image)
        if output_type in ("slopes", "slopes_vector"):
            return self._compute_slopes_from_frame(image)
        if output_type == "slopes2d":
            slopes = self._compute_slopes_from_frame(image)
            return self._pupil_data.slopes2d(slopes, useNaN=True)
        raise ValueError(
            "output_type must be 'slopes2d', 'pupils', 'slopes', or 'slopes_vector'"
        )

    def __repr__(self) -> str:
        n_sub = self._pupil_data.n_subap if self._pupil_data is not None else None
        cam_name = getattr(self._camera, "_name", type(self._camera).__name__)
        return (
            f"{self._name}(camera={cam_name}, n_pupils={self.n_pup}, "
            f"n_subap={n_sub}, exposure_time={self.exposure_time}ms, "
            f"fps={self.fps}Hz, norm={self.norm})"
        )


class BiOEdge(BaseWavefrontSensor): ...
