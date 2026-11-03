import numpy as _np
from opticalib.core import _types as _ot
from ..base_devices import BaseWavefrontSensor
from skimage import transform as _transform, feature as _sfeat
from opticalib.ground.geometry import draw_circular_pupil as _dcp


class PupilDetector(BaseWavefrontSensor):
    """
    Base class for pupil detection and extraction algorithms.

    This class provides the basic framework for detecting and extracting pupils
    from images for Wavefront Sensors Devices, which can be extended by specific
    pupil detection algorithms.

    Methods
    -------
    _detect_pupils(image: ImageData)
        Detects pupils in the given image using the Hough Transform.
    _extract_pupils(image: ImageData) -> CubeData
        Extracts the pupil images from the given image based on detected pupil
        positions.

    Usage
    -----
    This class is intended to be used as a base class for basic pupil detection
    algorithms. the user should always implement the ``_compute_slopes_from_pupils``
    method in the subclass to compute the slopes from the detected pupils.
    """

    def __init__(self, config_dict: dict[str, _ot.Any]):
        """The constructor"""
        _ATTRMAP = {
            "n_pupils": "n_pupils",
            "_dr": "delta_r",
            "sigma_threshold": "sigma_threshold",
            "_pupil_diameter": "subapertures",
            "_pupil_centers": "pupils_centers",
            "_camera_binning": "camera_binning",
            "_camera_base_exptime": "camera_base_exptime",
            "_camera_fps": "camera_fps",
            "exposure_time": -1,  # placeholder to set None
            "pupil_data": -1,  # placeholder to set None
        }
        for attr, key in _ATTRMAP.items():
            setattr(self, attr, config_dict.get(key, None))

        self._pupil_radius = self._pupil_diameter / 2
        self._rmin = self._pupil_radius - self._dr
        self._rmax = self._pupil_radius + self._dr

        self._pup_dtype = [("xc", float), ("yc", float), ("radius", float)]

        self._pupil_info = (
            _np.asarray(
                [
                    (x, y, r)
                    for (x, y), r in zip(
                        self._pupil_centers, [self._pupil_radius] * self.n_pupils
                    )
                ],
                dtype=self._pup_dtype,
            )
            if self._pupil_centers and self._pupil_radius
            else None
        )

    @property
    def pupil_info(self):
        return self._pupil_info

    def _detect_pupils(self, image: _ot.ImageData, pupil_order: str = "+y+x"):
        """
        Detect pupils in the given image using the Hough Transform.

        This method applies the Canny edge detection followed by the Hough
        Transform to identify circular features corresponding to pupils in
        the input image.

        Parameters
        ----------
        image : ImageData
            The input image in which to detect pupils.
        pupil_order : str, optional
            The order in which to sort the detected pupils.
            The sign indicates the direction of sorting ('+' for ascending, '-'
            for descending), while the order of the axes ('x' or 'y') indicates
            the primary and secondary sorting directions.

            Default is '+y+x', which sorts in the increasing y direction 
            (bottom-to-top), and then in the increasing x direction 
            (left-to-right).

        Raises
        ------
        RuntimeError
            If not enough pupils are detected in the image.
        """
        import warnings

        pupdata = _np.array([], dtype=self._pup_dtype)

        # find edges with canny algorithm
        edges = _sfeat.canny(image.astype(_np.float32), sigma=self.sigma_threshold)

        # finding the best fitting radius
        hough_radii = _np.arange(self._rmin, self._rmax + 1, 1)
        hough_res = _transform.hough_circle(edges, hough_radii)

        _, cx, cy, hough_radii = _transform.hough_circle_peaks(
            hough_res,
            hough_radii,
            total_num_peaks=1,
            min_xdistance=self._rmin,
            min_ydistance=self._rmin,
        )

        if any(hough_radii == x for x in [self._rmin, self._rmax]):
            warnings.warn(
                "Best fitting radius has reached the limit of the input search range. "
                "Consider extending the search range"
            )

        # find best matching pupils
        hough_res = _transform.hough_circle(edges, hough_radii)
        _, cx, cy, radii = _transform.hough_circle_peaks(
            hough_res,
            hough_radii,
            total_num_peaks=self.n_pupils,
            min_xdistance=self._rmin,
            min_ydistance=int(_np.sqrt(3) * self._rmin),
        )

        if len(cx) < self.n_pupils:
            raise RuntimeError(
                f"Not enough pupils detected. "
                f"Found {len(cx)} pupils, expected {self.n_pupils}."
            )

        for center_y, center_x, radius in zip(cy, cx, radii):
            pupdata = _np.append(
                pupdata, _np.array((center_x, center_y, radius), dtype=self._pup_dtype)
            )
        if pupil_order not in {"+y+x", "+x+y", "-y+x", "-x+y", "-y-x", "-x-y"}:
            raise ValueError(f"Unsupported pupil_order: {pupil_order}")
        secondary = pupil_order[:2]
        primary = pupil_order[2:]
        primary_values = pupdata[primary[1] + "c"] * (1 if primary[0] == "+" else -1)
        secondary_values = pupdata[secondary[1] + "c"] * (
            1 if secondary[0] == "+" else -1
        )
        pupdata_sorted = pupdata[_np.lexsort((secondary_values, primary_values))]
        self._pupil_info = pupdata_sorted

    def _extract_pupils(self, image: _ot.ImageData):
        """
        Pupil extraction algorithm that extracts the pupils from the input image
        based on the detected pupil information.

        This returns the raw pupils in the frame, without applying any normalization
        nor transformation.

        Parameters
        ----------
        image : ImageData
            The input image from which to extract pupils.

        Returns
        -------
        pupils : CubeData
            The extracted raw pupil images as a cube of shape [n_pup, height, width].
        """
        r = int(self._pupil_info["radius"][0])
        shape = tuple([2 * r + 1] * 2)
        pupil = _dcp(shape, r)
        pupils = _np.ma.zeros((self.n_pupils, shape[0], shape[1]))

        for pp in range(self.n_pupils):
            xc = int(self._pupil_info["xc"][pp])
            yc = int(self._pupil_info["yc"][pp])
            pupils[pp, :, :] = image[(yc - r) : (yc + r + 1), (xc - r) : (xc + r + 1)]
            pupils[pp, pupil] = _np.nan
            pupils[pp, :, :].mask = pupil

        return pupils
