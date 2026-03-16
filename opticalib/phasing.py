import os as _os
import numpy as _np
from . import typings as _ot
from .ground import osutils as _osu
from .ground.logger import SystemLogger as _SL
from .devices.cameras import AVTCamera as _cam
from .core.fitsarray import fits_array as _fits_array
from .core.read_config import (
    getDeviceConfig as _gdc, getPhasingConfig as _gpc
)
from .analyzer import frame
from .core.root import folders as _fn
from scipy import ndimage as _ndi

_splconf = _gpc()

def _get_tunable_filter():
    """
    initiate the tunable filter with standard parameters
    """
    from plico_motor import motor  # type: ignore

    devtype, device = _splconf["filter"].split(":")
    ip, port = _gdc(devtype, device).values()

    return motor(ip, port, axis=0)


_FILTER_BANDWIDTH_MODE = {"narrow": 8, "medium": 4, "wide": 2, "black": 1}


class SPL:
    """
    Sensor for Phase Lag
    ====================

    The Sensor for Phase Lag (SPL) is a device composed of a laser and a camera,
    which measures the phase lag between the incoming light and a reference signal, by
    acquiring images at different wavelengths and analyzing the resulting fringes.

    Parameters
    ----------
    camera: AVTCamera
        The camera used to acquire images in the SPL system
    tunable_filter: object
        The tunable filter client to regulate the wavelength of the incoming light.
    tnfringes: str | None
        The tracking number of the simulated fringes measurements, for template
        comparison during analysis.
    """

    def __init__(
        self,
        camera: str | _cam | None = None,
        tunable_filter: object | None = None,
        tnfringes: str | None = None,
    ):
        if isinstance(camera, str):
            if camera.lower() == "none":
                camera = None
            else:
                camera = _cam(name=camera)
        elif camera is None:
            try:
                _, device = _splconf["camera"].split(":")
                camera = _cam(name=device)
            except Exception:
                camera = None

        if tunable_filter is None:
            try:
                tunable_filter = _get_tunable_filter()
            except Exception:
                tunable_filter = None

        self._camera = camera
        self._filter = tunable_filter

        self.set_tn_fringes(tnfringes)

        self._darkFrame1sec = None
        self._curr_exptime = None
        self._last_measure_tn = None
        self._logger = _SL(__class__)


    def set_tn_fringes(self, tnfringes: str | None):
        """
        Set the tracking number of the simulated fringes measurements, for
        template comparison during analysis.

        Parameters
        ----------
        tnfringes : str | None
            The tracking number of the simulated fringes measurements,
            for template comparison during analysis.
        """
        self._tnfringes = tnfringes
        if tnfringes is not None:
            self._fringes_fold = _os.path.join(_fn.SPL_FRINGES_ROOT_FOLDER, tnfringes)
        else:
            self._fringes_fold = None

    def set_filter_mode(self, mode: str):
        """
        Set the tunable filter bandwidth mode.

        Parameters
        ----------
        mode : str
            The bandwidth mode to set. Must be one of "narrow", "medium", "wide", "black".

        Raises
        ------
        ValueError
            If the mode is not one of the allowed values.
        """
        if mode not in _FILTER_BANDWIDTH_MODE:
            self._logger.error(
                f"Invalid filter mode: {mode}. Must be one of {list(_FILTER_BANDWIDTH_MODE.keys())}"
            )
            raise ValueError(
                f"Invalid filter mode: {mode}. Must be one of {list(_FILTER_BANDWIDTH_MODE.keys())}"
            )

        self._filter.set_bandwidth_mode(_FILTER_BANDWIDTH_MODE[mode])
        self._logger.info(f"Set filter bandwidth mode to: {mode}")

    def set_exptime(self, exptime: float):
        """
        Set the SPL camera exposure time, if the target value is different from the current one

        Parameters
        ----------
        exptime : float
            the exposure time in [s]

        """
        if self._curr_exptime != exptime:
            self._camera.set_exptime(exptime * 1e6)
            self._curr_exptime = exptime
        else:
            self._logger.warning(
                "The requested exposure time for the camera is equal to the current one. Skipping"
            )
            pass

    def acquire_dark_frame(self, exptime: float, nframes: int = 1) -> _ot.ImageData:
        """
        Acquire a dark frame for the camera

        Parameters
        ----------
        exptime : float
            the exposure time in [s]
        nframes : int
            the number of frames to acquire

        Returns
        -------
        dark_frame : ImageData
            The dark frame acquired, scaled to 1 second exposure time
        """
        self.set_exptime(exptime)
        self.set_filter_mode("black")
        self._darkFrame1sec = self._camera.acquire_frames(nframes) / exptime

        if not hasattr(self._darkFrame1sec, "mask"):
            mask = _np.zeros(self._darkFrame1sec.shape)
        else:
            mask = self._darkFrame1sec.mask

        self._darkFrame1sec = _fits_array(data=self._darkFrame1sec, mask=mask)
        self._darkFrame1sec.header["EXPTIME"] = exptime
        self._darkFrame1sec.header["NFRAMES"] = nframes
        self._darkFrame1sec.header["ISDARK"] = True
        self._darkFrame1sec.header["ISRAW"] = True

        return self._darkFrame1sec.copy()

    def preview_detection(
        self,
        img: _ot.ImageData | None = None,
        exptime: _ot.Optional[float] = None,
        filter_mode: _ot.Optional[str] = None,
        wavelength: _ot.Optional[float] = None,
        **kwargs: dict[str, _ot.Any],
    ):
        """
        Acquire an image with the provided settings (or the current ones) and
        preview the detected PSF centroids and crop boxes.

        Parameters
        ----------
        exptime : float, optional
            The exposure time to set for the camera before acquiring the image.

            If None, the current exposure time is used.
        filter_mode : str, optional
            The filter bandwidth mode to set for the tunable filter before
            acquiring the image. Must be one of "narrow", "medium", "wide".

            If None, the current filter mode is used.
        wavelength : float, optional
            The wavelength to set for the tunable filter before acquiring the image.

            If None, the current wavelength is used.
        kwargs : dict
            Additional keyword arguments to pass to the `detect_psf_centroids`
            method, such as `n_psf`, `nsigma`, `min_pixels`, etc.

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if img is None:
            if exptime is not None:
                self.set_exptime(exptime)

            if filter_mode is not None:
                self.set_filter_mode(filter_mode)

            if wavelength is not None and self._filter is not None:
                self._filter.move_to(wavelength)

            img = self._camera.acquire_frames(1)

        centroids = self.detect_psf_centroids(img, **kwargs)
        _, boxes = self.crop_around_centroids(img, centroids)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(img, cmap="gray", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        i = 0
        for (xc, yc), (x0, y0, w, h) in zip(centroids, boxes):
            ax.plot(xc, yc, "r+", markersize=12, markeredgewidth=2)
            ax.add_patch(
                Rectangle((x0, y0), w, h, fill=False, edgecolor="cyan", linewidth=1.5)
            )
            ax.text(x0, y0 - h, f"{i}", fontdict={"fontsize": 20, "color": "white"})
            i += 1

        ax.set_title("Detected PSF centroids and crop boxes")
        plt.tight_layout()
        plt.show()

    def acquire(
        self,
        exptime: float,
        filter_mode: str | None = None,
        lambda_vector: _ot.ArrayLike | None = None,
        nframes: int = 1,
        mask: _ot.MaskData | None = None,
    ):
        """
        Acquire SPL measurements at different wavelengths, by moving the
        "tunable filter".

        Parameters
        ----------
        exptime: float
            Base exposure time of the camera in seconds
        lambda_vector : ArrayLike, optional
            Wavelenghts vector, of wavelengths between 400 and 700 nm. If None,
            a default vector is used:
            - from 400 to 700 with 20 nm step

            By default None.
        nframes : int, optional
            number of frames to average for each wavelength, by default 1
        mask: MaskData | None, optional
            Mask to apply to the measurements. By default, an ampty mask
            is applied.

        Returns
        -------
        tn: string
            Tracking number of measurements, in the `.../SPL` folder.
        """
        if lambda_vector is not None:
            lambda_vector = _np.asarray(lambda_vector)
            if _np.min(lambda_vector) < 400 or _np.max(lambda_vector) > 700:
                self._logger.error(
                    f"AcquisitionError: Wavelengths must be between 400 and 700 nm"
                )
                raise ValueError("Wavelengths must be between 400 and 700 nm")
        else:
            lambda_vector = _np.arange(440, 721, 20)

        if filter_mode is not None:
            self.set_filter_mode(filter_mode)

        datapath = _osu.create_data_folder(basepath=_fn.SPL_DATA_ROOT_FOLDER)
        tn = datapath.split("/")[-1]
        print(tn)

        _osu.save_fits(
            _os.path.join(datapath, "lambda_vector.fits"),
            lambda_vector,
        )

        if self._darkFrame1sec is not None:
            _osu.save_fits(
                _os.path.join(datapath, "darkFrame.fits"), self._darkFrame1sec
            )
            self._logger.info("Saved current dark frame")
        else:
            self.acquire_dark_frame(exptime=1.0, nframes=1)
            _osu.save_fits(
                _os.path.join(datapath, "darkFrame.fits"), self._darkFrame1sec
            )
            self._logger.info("Acquired and saved new dark frame")

        self._logger.info(f"Starting SPL acquisition with tracking number: {tn}")

        # Create the Gain Vector
        expgain = _np.ones(lambda_vector.shape[0]) * 0.5
        expgain[_np.where(lambda_vector < 550)] = 1  # 8
        expgain[_np.where(lambda_vector < 530)] = 2  # 8
        expgain[_np.where(lambda_vector > 650)] = 1  # 3
        expgain[_np.where(lambda_vector > 700)] = 1.5  # 8
        self._logger.info(f"Acquisition of frames")

        for wl, t_int in zip(lambda_vector, (expgain * exptime)):
            self._filter.move_to(wl)
            self.set_exptime(t_int)
            self._logger.info(
                f"Acquiring image at {wl:.1f} [nm] with exposure time of {self._curr_exptime:.3f} [s]"
            )
            print(f"Moving to lambda: {wl}")
            img = self._camera.acquire_frames(nframes)
            if mask is None:
                mask = _np.zeros(img.shape)

            image = _fits_array(
                data=img,
                mask=mask,
                header={
                    "EXPTIME": self._curr_exptime,
                    "WAVELEN": wl,
                    "EXPGAIN": t_int / self._curr_exptime,
                },
            )
            image.writeto(
                _os.path.join(datapath, f"rawframe_{wl}nm.fits"), overwrite=True
            )

        self._filter.move_to(600)
        self._logger.info(f"Saved tracking number: {tn}")

        self._last_measure_tn = tn

        return tn

    def analysis(
        self,
        tn: str | None = None,
        n_psfs: _ot.Optional[int] = None,
        **process_kwargs: dict[str, _ot.Any]
    ) -> list[float]:
        """
        Analyze the measurements acquired with the `acquire` method, by detecting
        the PSF centroids and cropping the frames around them.

        Parameters
        ----------
        tn : str
            Tracking number of the measurement to be processed. If None, the last
            acquired measurement is processed.
        n_psfs : int
            The number of PSFs expected to analyze. By default, 1.
        """
        self.process_psfs(tn=tn, n_psfs=n_psfs, **process_kwargs)

        n_psfs = n_psfs or _splconf.get("expected_psfs", 1)

        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        lambda_vector = _osu.load_fits(_os.path.join(datapath, "lambda_vector.fits"))
        self._last_lambdavec = lambda_vector.copy()

        pists = []
        for i in range(n_psfs):
            cube = _osu.load_fits(_os.path.join(datapath, f"psf{i}_cube.fits"))

            matrix, matrix_smooth = self._compute_matrix(lambda_vector, cube)
            pist, pist_smooth = self._template_comparison(
                matrix, matrix_smooth, lambda_vector
            )

            matrix.header["DPIST"] = (pist, "measured differential piston in nm")
            matrix.header["DPISTSM"] = (
                pist_smooth,
                "smoothed differential piston in nm",
            )
            matrix.header["TEMPIDX"] = (self._idp, "index of the best template match")

            # self._save_piston_results(datapath, pist, pist_smooth)
            _osu.save_fits(
                _os.path.join(datapath, f"psf{i}_fringes_result.fits"), matrix
            )

            pists.append(pist)

        return pists

    def process_psfs(
        self,
        tn: str | None = None,
        n_psfs: _ot.Optional[int] = None,
        min_pixels: _ot.Optional[int] = None,
        nsigma: _ot.Optional[float] = None,
        angles: float | list[float] | None = 0.0,
        initial_half_size: _ot.Optional[int] = None,
        final_half_size: _ot.Optional[int] = None,
        centroid_min_distance: _ot.Optional[int] = None,
        method: str = "lsf_peaks",
        remove_dark: bool = False,
        remove_median: bool = False,
        rotation_order: int = 3,
        rotation_cval: float = 0.0,
    ):
        """
        Process the raw frames acquired with the `acquire` method, by detecting
        the PSF centroids and cropping the frames around them.

        Parameters
        ----------
        tn : str
            Tracking number of the measurement to be processed.
        n_psfs : int
            The number of PSFs to detect.

            If None, it is read from the configuration file, and if absent,
            defaults to 1.
        min_pixels : int
            The minimum number of pixels required for a detection to be
            considered a PSF.

            If None, it is read from the configuration file, and if absent,
            defaults to 30.
        nsigma : float
            The number of sigma above the background to use as threshold for the
            detection.

            If None, it is read from the configuration file, and if absent,
            defaults to 2.0.
        angles : list of float | None
            List of angles in degrees to rotate each PSF crop. If None, no rotation is
            applied.

            By default, 0.
        initial_half_size : int
            The half size of the initial cropped images, in pixels.

            By default, 70, which corresponds to a 300x300 pixels crop.
        final_half_size : int
            The half size of the final cropped images, in pixels.

            By default, 40, which corresponds to an 80x80 pixels crop.
        centroid_min_distance : int
            The minimum distance in pixels between centroids to consider them as
            different PSFs. This is used to filter out multiple detections of the
            same PSF, since each PSF has more than one lobe.

            By default, 50 pixels.
        method: str
            The method to use for computing the photometric centroid in the cropped PSF images.
            Can be:
            - 'lsf_peaks': find the peaks of the line spread function along `x` and `y` axis
            - "com": center of mass (`photutils.centroids.centroid_com`)
            - '2dg': 2D Gaussian fit (`photutils.centroids.centroid_2dg`)   
        remove_dark : bool
            If True, removes the dark frame from each frame before processing.

            By default, False.
        remove_median : bool
            If True, removes the median value from each cropped PSF image.

            By default, False.
        rotation_order : int
            The order of the spline interpolation used for the rotation of the 
            PSF crops.
            
            By default, 3 (cubic).
        rotation_cval : float
            The constant value to fill the area outside the input image after 
            rotation of the PSF crops.
            
            By default, 0.0.
        """
        n_psfs = n_psfs or _splconf.get("expected_psfs", 1)
        nsigma = nsigma or _splconf.get("sigma_threshold", 2.0)
        min_pixels = min_pixels or _splconf.get("min_px_threshold", 30)
        angles = angles or _splconf.get("psfs_angles", [0.0] * n_psfs)
        centroid_min_distance = centroid_min_distance or _splconf.get("centroid_min_dist", 50)

        tn = tn or self._last_measure_tn
        if tn is None:
            self._logger.error(
                "No tracking number provided and no previous measurement found."
            )
            raise ValueError(
                "No tracking number provided and no previous measurement found."
            )

        if not isinstance(angles, list):
            angles = [angles] * n_psfs

        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        filelist = _osu.getFileList(fold=datapath, key="rawframe")
        rawlist = [self._heal_bad_pixels( _osu.load_fits(x) ) for x in filelist]

        ## FIRST CENTROID DETECTION, made on the sum of all the images
        # Needed for the fist generous image crop.

        sumimg = _np.ma.sum(_np.ma.dstack(rawlist), axis=2)

        # Removing the saved dark frame
        if remove_dark:
            dark = self._get_dark_frame(tn)
            print("Removing Dark")
            sumimg = sumimg - dark * sumimg.header["EXPTIME"] * len(rawlist)
            sumimg.header["RDARK"] = True

        centroids = self.detect_psf_centroids(
            sumimg,
            n_psfs=n_psfs,
            min_pixels=min_pixels,
            nsigma=nsigma,
            centroid_min_distance=centroid_min_distance
        )

        ichalf_size = initial_half_size or _splconf.get(
            "initial_crop_half_size", 150
        )

        for img, filename in zip(rawlist, filelist):
            med = _np.median(img)

            # This is the 1st, BIG crop around spatial centroids (NOT photometric)
            crops, _ = self.crop_around_centroids(img, centroids, half_size=ichalf_size)

            for i, crop in enumerate(crops):

                if remove_median:
                    crop = _fits_array(
                        _np.clip(crop - med, 0, None), header=crop.header.copy()
                    )
                    crop.header["RMEDIAN"] = True
                    crop.header["MEDVAL"] = med

                header = img.header.copy()
                header["PSFNUM"] = i

                header["CENTX"] = (
                    centroids[i][0],
                    "PSF_x centroid in the original frame",
                )
                header["CENTY"] = (
                    centroids[i][1],
                    "PSF_y centroid in the original frame",
                )

                crop.header.update(header)
                # This save for now for debugging
                crop.writeto(filename.replace("frame", f"psf{i}"), overwrite=True)

        # Now we find the photometric centroid, shift the crops to put it 
        # at the center of the image, rotate them, and then cut again, 
        # using a smalled window.

        self._shift_and_rotate_psf(
            tn,
            n_psfs,
            method,
            angles,
            nsigma,
            order=rotation_order,
            cval=rotation_cval
        )

        self._create_psf_cubes_and_crop_again(tn, n_psfs, final_half_size)

    def rotate_psf(
        self,
        cropped_img: _ot.ImageData,
        angle: float,
        order: int = 3,
        cval: float = 0.0,
    ) -> _ot.ImageData:
        """
        Rotate the cropped PSF image by a given angle around its centroid, which
        is assumed to be at the center of the cropped image.

        Parameters
        ----------
        cropped_img : ImageData
            The cropped PSF image to be rotated.
        angle : float
            The angle in degrees by which to rotate the image. Positive values
            correspond to counter-clockwise rotation.
        order : int
            The order of the spline interpolation used for the rotation.

            By default, 3 (cubic).
        cval : float
            The constant value to fill the area outside the input image after
            rotation.

            By default, 0.0.

        Returns
        -------
        rotated_img : ImageData
            The rotated PSF image.
        """
        header = cropped_img.header.copy()

        rotated_img = _ndi.rotate(
            cropped_img, angle, reshape=False, order=order, mode="constant", cval=cval
        )

        return _fits_array(data=rotated_img, mask=False, header=header)

    def detect_psf_centroids(
        self,
        raw_frame: _ot.ImageData,
        n_psfs: _ot.Optional[int] = None,
        nsigma: _ot.Optional[float] = None,
        min_pixels: _ot.Optional[int] = None,
        centroid_min_distance: int = 50,
    ) -> list[tuple[int, int]]:
        """
        Detect the centroids of the PSFs in the raw frame acquired with the
        `acquire` method.

        This centroid detection is the first in the analysis chain, as it serves
        to do an initial crop around each found PSF, which is then used to find
        the photometric centroid and do a second crop for the final analysis.
        For this reason, this detection is more "generous" in the number of
        pixels required for a detection, and in the threshold, to be sure to
        find all the PSFs even in the noisiest frames.

        Parameters
        ----------
        raw_frame : ImageData
            The raw frame acquired with the `acquire` method, from which the PSF
            centroids are to be detected.
        n_psfs : int
            The number of PSFs to detect. If None, it is read from the
            configuration file, ad if not found there, it defaults to 1.
        nsigma : float
            The number of sigma above the background to use as threshold for the
            detection.
        min_pixels : int
            The minimum number of pixels required for a detection to be
            considered a PSF. By default, 30.
        centroid_min_distance : int
            The minimum distance in pixels between centroids to consider them as
            different PSFs.

        Returns
        -------
        centroids_xy : list of tuples
            The list of centroids of the detected PSFs, in (x, y) format.
        """
        nsigma = nsigma or _splconf.get("sigma_threshold", 2.0)
        n_psfs = n_psfs or _splconf.get("expected_psfs", 1)
        min_pixels = min_pixels or _splconf.get("min_px_threshold", 30)
        centroid_min_distance = centroid_min_distance or _splconf.get(
            "centroid_min_dist", 50
        )
        desired_order = _splconf.get("expected_psf_pos", None)

        # 1) Robust background/noise estimate
        bkg = _np.median(raw_frame)
        mad = _np.median(_np.abs(raw_frame - bkg))
        sigma = 1.4826 * mad if mad > 0 else _np.std(raw_frame)

        # 2) Threshold + connected components
        mask = raw_frame > (bkg + nsigma * sigma)
        mask = _ndi.binary_opening(mask, structure=_np.ones((3, 3)))
        labels, nlab = _ndi.label(mask)

        centroids = []
        for lab in range(1, nlab + 1):

            # Taking the pixels of the current label
            yy, xx = _np.where(labels == lab)

            # first check
            if yy.size < min_pixels:
                continue

            I = raw_frame[yy, xx].astype(float)

            # backup check
            Itot = I.sum()
            if Itot <= 0:
                continue

            xc = _np.sum(xx * I) / Itot
            yc = _np.sum(yy * I) / Itot
            centroids.append((xc, yc, yy.size))

        def custom_sort_key(c):
            if desired_order is None:
                # Fallback to the old sorting method
                return (c[1], c[0])
            # Find the index of the closest point in `desired_order` to the centroid `c`
            return min(
                range(len(desired_order)),
                key=lambda i: (
                    (c[0] - desired_order[i][0]) ** 2
                    + (c[1] - desired_order[i][1]) ** 2
                ),
            )

        # First, filtering to have only one centroid per detected PSF
        # (since each PSF has more than 1 lobe), then keep the largest `n_psfs`
        for c in centroids:
            other_cs = [oc for oc in centroids if oc != c]
            for oc in other_cs:
                if all([abs(c[i] - oc[i]) < centroid_min_distance for i in range(2)]):
                    if oc in centroids:
                        centroids.remove(oc)
        centroids = sorted(centroids, key=lambda t: t[2], reverse=True)[:n_psfs]
        centroids_xy = [(int(c[0].round()), int(c[1].round())) for c in centroids]

        # Sorting
        centroids_xy = sorted(centroids_xy, key=custom_sort_key)

        return centroids_xy

    def detect_photometric_centroid(
        self,
        cropped_psf: _ot.ImageData,
        nsigma: _ot.Optional[float] = None,
        method: str = "2dg",
    ) -> list[tuple[int, int]]:
        """
        Detect the photometric centroid of the PSF in the cropped image.

        Parameters
        ----------
        cropped_psf : ImageData
            The cropped PSF image, from which the photometric centroid is to be detected.
        method : str
            The method to use for the centroid detection. 
            Can be:
            - 'lsf_peaks': find the peaks of the line spread function along 
            `x` and `y` axis, and take their intersection as centroid.
            - "com": center of mass (`photutils.centroids.centroid_com`)
            - '2dg': 2D Gaussian fit (`photutils.centroids.centroid_2dg`)
            - any custom callable method passed: the method should take as input 
            a 2D array (the cropped PSF) and return a tuple of (x, y) coordinates
            of the centroid.

        Returns
        -------
        centroid_xy : list of tuple of int
            The (x, y) coordinates of the photometric centroid in the cropped image.

            Returned as a list for code compatibility
        """
        nsigma = nsigma or _splconf.get("sigma_threshold", 5.0)
        match method:
            case "2dg":
                from photutils.centroids import centroid_2dg as method
            case "com":
                from photutils.centroids import centroid_com as method
            case 'lsf_peaks':
                am = _np.argmax
                sum = _np.sum
                method = lambda img: (am(sum(img, axis=0)), am(sum(img, axis=1)))
            case _ if callable(method):
                pass
            case _:
                self._logger.error(
                    f"Invalid method for photometric centroid detection: {method}"
                )
                raise ValueError(
                    f"Invalid method for photometric centroid detection: {method}"
                )

        # Mask threshold computation
        counts, bin_edges = _np.histogram(cropped_psf, bins=100)
        bin_edges = bin_edges[1:]
        thr = nsigma * bin_edges[_np.where(counts == max(counts))]
        mask = cropped_psf < thr
        img = _np.ma.masked_array(cropped_psf, mask=mask)
        peaks = method(img)
        return (int(_np.round(peaks[0])), int(_np.round(peaks[1])))

    def crop_around_centroids(
        self,
        frame: _ot.ImageData,
        centroids: list[tuple[int, int]],
        half_size: _ot.Optional[list[int] | int] = None,
    ) -> tuple[list[_ot.ImageData], list[tuple[int, int, int, int]]]:
        """
        Crop the input frame around the detected centroids, to obtain the 6 PSF
        images.

        Parameters
        ----------
        frame : ImageData
            The input frame to crop, typically the raw frame acquired with the
            `acquire` method.
        centroids : list of tuples
            The list of centroids of the detected PSFs, in (x, y) format, as
            obtained from `detect_psf_centroids` method.
        half_size : list of int | int, optional
            The half size of the cropped images, in pixels. If None, it is read
            from the configuration file, and if not found there, it defaults to
            75, which corresponds to a 150x150 pixels crop. Individual values for
            x and y axis can be provided as a list of two integers.
        """
        if half_size is None:
            half_size = _splconf.get("initial_crop_half_size", 75)

        if not isinstance(half_size, list):
            half_size = [half_size] * 2

        crops = []
        boxes = []
        h, w = frame.shape
        for xc, yc in centroids:
            x0 = max(0, xc - half_size[0])
            x1 = min(w, xc + half_size[0])
            y0 = max(0, yc - half_size[1])
            y1 = min(h, yc + half_size[1])
            crops.append(frame[y0:y1, x0:x1].copy())

            # Useful for plotting
            boxes.append((x0, y0, x1 - x0, y1 - y0))

        return crops, boxes

    def plot_comparison(
        self, tn: str | None = None, psf_n: int | str = "all"
    ) -> None:
        """
        Plot measured fringes and best-match templates.

        Parameters
        ----------
        tn : str | None
            Tracking number. If None, use last measurement.
        psf_n : int | str
            PSF index, or "all" to plot all available PSFs.
        """
        import math
        import matplotlib.pyplot as plt

        tn = tn or self._last_measure_tn
        if tn is None:
            msg = "No tracking number provided and no previous measurement found."
            self._logger.error(msg)
            raise ValueError(msg)

        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        wl = _osu.load_fits(_os.path.join(datapath, "lambda_vector.fits"))

        if psf_n == "all":
            n_expected = int(_splconf.get("expected_psfs", 6))
            psf_indices = [
                i
                for i in range(n_expected)
                if _os.path.exists(
                    _os.path.join(datapath, f"psf{i}_fringes_result.fits")
                )
            ]
            if not psf_indices:
                raise FileNotFoundError(
                    f"No psf*_fringes_result.fits found in {datapath}."
                )
        elif isinstance(psf_n, int):
            psf_indices = [psf_n]
        else:
            raise TypeError("psf_n must be an integer or 'all'.")

        # Outer layout: 2 columns of PSF-panels, multiple rows
        outer_cols = 2
        outer_rows = math.ceil(len(psf_indices) / outer_cols)

        fig = plt.figure(figsize=(14, 3.5 * outer_rows), constrained_layout=False)
        outer = fig.add_gridspec(
            outer_rows,
            outer_cols,
            wspace=0.3,
            hspace=0.3)

        for k, i_psf in enumerate(psf_indices):
            r = k // outer_cols
            c = k % outer_cols

            # Inner layout in each PSF-panel: measured | template
            inner = outer[r, c].subgridspec(1, 2, wspace=0.15)
            ax_meas = fig.add_subplot(inner[0, 0])
            ax_temp = fig.add_subplot(inner[0, 1])

            mat = _osu.load_fits(
                _os.path.join(datapath, f"psf{i_psf}_fringes_result.fits")
            )
            temp_idx = int(mat.header["TEMPIDX"])
            dpist = mat.header.get("DPIST", "N/A")
            extent = [wl[0], wl[-1], 0, mat.shape[0]]

            ax_meas.imshow(mat, aspect="auto", extent=extent)
            ax_meas.set_title(f"PSF {i_psf} - Measured")
            ax_meas.set_xlabel(r"$\lambda$ [nm]")
            ax_meas.set_ylabel("")
            ax_meas.set_xlabel("")

            ax_temp.imshow(self._Qt[:, :, temp_idx], aspect="auto", extent=extent)
            ax_temp.set_title(f"Template - DPIST={dpist} nm")
            ax_temp.set_xlabel(r"$\lambda$ [nm]")
            ax_temp.set_ylabel("")
            ax_temp.set_xlabel("")
            ax_temp.set_yticks([])
            ax_temp.set_yticklabels([])

        fig.supxlabel(r"$\lambda$ [nm]", fontsize=14)
        fig.supylabel("Pixels", fontsize=14)
        fig.suptitle(f"Fringes comparison - {tn}", fontsize=16, fontweight="semibold")
        plt.show()
    
    def _heal_bad_pixels(
        self,
        img: _ot.ImageData,
        r: int = 2, method: str = 'median',
        sigma_thr: float = 5.5
    ) -> _ot.ImageData:
        """
        Function which finds and heals bad pixels in the input image.

        Parameters
        ----------
        img : ImageData
            The input image with potential bad pixels to heal.
        r : int
            The radius of the neighborhood to consider for healing. By default, 2.
        method : str
            The method to use for healing the bad pixels. Can be:
            - 'median'
            - 'mean'
            - 'gaussian'
            - a user defined custom callable
            
            By default, 'median'.
        sigma_thr : float
            The sigma threshold to identify bad pixels based on the gradient. 
            
            By default, 5.5.

        Returns
        -------
        healed_img : ImageData
            The output image with bad pixels healed.
        """
        from scipy.signal import convolve2d

        # Finding the Bad Pixels
        gradX,gradY = _np.gradient(img)
        grad = _np.sqrt(gradX**2+gradY**2)
        ker = _np.array([[0,1,0],
                        [1,0,1],
                        [0,1,0]])/4
        filt_grad = convolve2d(grad,ker,mode='same',boundary='symm')

        N_hot_pixels = len(grad[grad>sigma_thr*_np.std(grad)+_np.mean(grad)])//4
        hot_pix_ids = _np.argsort(filt_grad.flatten())[-N_hot_pixels:]

        healed_img = img.copy().flatten()
        rows = _np.repeat(_np.arange(img.shape[0]),img.shape[1])
        cols = _np.tile(_np.arange(img.shape[1]),img.shape[0])

        # Healing the bad pixels
        for pix_id in hot_pix_ids:
            row = rows[pix_id]
            col = cols[pix_id]
            row_start = _np.max((0,int(_np.floor(row-(r-1)/2))))
            row_end = _np.min((img.shape[0],int(row+_np.ceil((r-1)/2))))
            col_start = _np.max((0,int(col-_np.floor((r-1)/2))))
            col_end = _np.min((img.shape[1],int(col+_np.ceil((r-1)/2))))

            match method:
                case 'median':
                    healed_img[pix_id] = _np.median(img[row_start:row_end+1,col_start:col_end+1])
                case 'mean':
                    healed_img[pix_id] = _np.mean(img[row_start:row_end+1,col_start:col_end+1])
                case 'gaussian':
                    from scipy.ndimage import gaussian_filter
                    
                    healed_img[pix_id] = gaussian_filter(
                        img[row_start:row_end+1,col_start:col_end+1],
                        sigma=1
                    )[(row-row_start),(col-col_start)]
                case _ if callable(method):
                    healed_img[pix_id] = method(img[row_start:row_end+1,col_start:col_end+1])
                case _:
                    self._logger.error(f"Invalid method for bad pixel healing: {method}")
                    raise ValueError(f"Invalid method for bad pixel healing: {method}")

        return healed_img.reshape(img.shape)


    def _shift_and_rotate_psf(
        self,
        tn: str,
        n_psfs: int,
        method: str,
        angles: list[float],
        nsigma: float,
        order: int = 3,
        cval: float = 0.0,
    ):
        """"""
        for p in range(n_psfs):

            fl = _osu.getFileList(tn, fold=_fn.SPL_DATA_ROOT_FOLDER, key=f"rawpsf{p}")
            rpsfl = [self._heal_bad_pixels(_osu.load_fits(x)) for x in fl]

            sumimg = _np.ma.sum(_np.ma.dstack(rpsfl), axis=2)
            phot_centroid = self.detect_photometric_centroid(sumimg, method=method, nsigma=nsigma)

            s = sumimg.shape
            shift = _np.array(
                (phot_centroid[1] - s[1] // 2, phot_centroid[0] - s[0] // 2)
            )
            spsf = [_np.roll(img, shift=-shift, axis=(0,1)) for img in rpsfl]

            for crop, fn in zip(spsf, fl):
                header = crop.header.copy()
                rotated = False
                if not angles[p] == 0.0:
                    rotated = True
                    crop = self.rotate_psf(
                        crop,
                        angle=angles[p],
                        order=order,
                        cval=cval
                    )

                header["ROTATED"] = (rotated, "was de-rotated")
                header["ROTANG"] = (
                    angles[p],
                    "rotation angle in degrees",
                )

                header['PHOTCENX'] = (
                    phot_centroid[1],
                    "photometric X-centroid",
                )
                header['PHOTCENY'] = (
                    phot_centroid[0],
                    "photometric Y-centroid",
                )

                crop.header = header
                crop.writeto(
                    fn.replace("rawpsf", f"rot_psf"), overwrite=True
                )

    def _create_psf_cubes_and_crop_again(
        self,
        tn: str,
        n_psfs: int,
        final_half_size: _ot.Optional[int | list[int]] = None,
    ):
        """ """
        final_half_size = final_half_size or _splconf.get("crop_half_size", 40)

        if not isinstance(final_half_size, list):
            final_half_size = [final_half_size] * 2

        for i in range(n_psfs):
            cube = _osu.loadCubeFromFilelist(
                tn, fold=_fn.SPL_DATA_ROOT_FOLDER, key=f"rot_psf{i}"
            )

            header = cube.header.copy()

            cropped_cube = _fits_array(
                self._crop_cube(cube, centroid=None, half_size=final_half_size), header=header
            )

            cropped_cube.writeto(
                _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn, f"psf{i}_cube.fits"),
                overwrite=True,
            )

    def _crop_cube(
        self,
        cube: _ot.CubeData,
        centroid: tuple[int, int] = None,
        half_size: _ot.Optional[int | list[int]] = None,
    ) -> list[_ot.ImageData]:
        """
        Simple wrapper to crop the cube inplace, instead of looping over the frames.

        Parameters
        ----------
        cube : CubeData
            The cube of images to be cropped, with shape [height, width, n_frames].
        centroid : tuple of int
            The (y, x) coordinates of the photometric centroid in the cube, around which to crop.

        Returns
        -------
        cropped_cube : list of ImageData
            The cropped cube of images, with shape [cropped_height, cropped_width, n_frames].
        """
        h, w, _ = cube.shape

        if centroid is None:
            centroid = (h // 2, w // 2)

        x0 = max(0, centroid[1] - half_size[1])
        x1 = min(w, centroid[1] + half_size[1])
        y0 = max(0, centroid[0] - half_size[0])
        y1 = min(h, centroid[0] + half_size[0])

        cropped_cube = cube[y0:y1, x0:x1, :].copy()
        return cropped_cube

    def _get_dark_frame(self, tn: str) -> _ot.ImageData:
        """
        Get the dark frame for the camera from the last measurement tracking number folder.
        If the dark frame is not found, it returns the dark frame acquired with the `acquire_dark_frame`
        method, and saves it in the tracking number folder.

        Parameters
        ----------
        tn : str
            Tracking number of the measurement to get the dark frame from.

        Returns
        -------
        dark : ImageData
            The dark frame for the camera.

        """
        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        try:
            dark = _osu.load_fits(_os.path.join(datapath, "darkFrame.fits"))
        except FileNotFoundError:
            if self._darkFrame1sec is not None:
                dark = self._darkFrame1sec.copy()
                _osu.save_fits(_os.path.join(datapath, "darkFrame.fits"), dark)
                self._logger.info(f"Saved current dark frame for tracking number: {tn}")
            else:
                self._logger.error(f"Dark frame not found for tracking number: {tn}")
                raise FileNotFoundError("Dark frame not found")
        return dark.copy()

    def _compute_matrix(
        self, lambda_vector: _ot.ArrayLike, cube: _ot.CubeData
    ) -> tuple[_ot.MatrixLike, _ot.MatrixLike]:
        """
        Calculate the matrix of fringes from the acquired cube of images.

        Parameters
        ----------
        lambda_vector: ArrayLike
            Vector of wavelengths (between 400/700 nm)
        cube: CubeData
            Cube of images [pixels, pixels, n_frames=lambda]

        Returns
        -------
        matrix: MatrixLike
            Matrix of fringes
        matrix_smooth: MatrixLike
            Smoothed matrix of fringes
        """
        sx = cube.shape[0]
        matrix = _np.zeros((sx, lambda_vector.shape[0]))
        matrix_smooth = _np.zeros((sx, lambda_vector.shape[0]))

        for i in range(lambda_vector.shape[0]):
            img = frame(i, cube)

            y = _np.ma.sum(img, axis=1)
            area = _np.ma.sum(y[:])
            y_norm = y / area
            matrix[:, i] = y_norm

            w = self._applySmoothing(y_norm, 4)
            w = w[:sx]
            matrix_smooth[:, i] = w

        matrix[_np.where(matrix == _np.nan)] = 0
        self._matrix = matrix
        self._matrixSmooth = matrix_smooth
        return _fits_array(matrix), _fits_array(matrix_smooth)

    def _template_comparison(
        self,
        matrix: _ot.MatrixLike,
        matrix_smooth: _ot.MatrixLike,
        lambda_vector: _ot.ArrayLike,
    ) -> tuple[int, int]:
        """
        Compare the matrix obtained from the measurements with
        the one recreated with the synthetic data in tn_fringes.

        Parameters
        ----------
        matrix: MatrixLike
            Measured matrix, [pixels, lambda]
        matrix_smooth: MatrixLike
            Measured smoothed matrix, [pixels, lambda]
        lambda_vector: ArrayLike
            Vector of wavelengths
        Returns
        -------
        piston: int
                piston value
        """
        from tqdm import trange

        self._logger.debug(f"Template Comparison with data in {self._tnfringes}")
        delta, lambda_synth = self._readDeltaAndLambdaFromFringesFolder()
        idx = _np.isin(lambda_synth, lambda_vector)

        Qm = matrix - _np.mean(matrix)
        Qm_smooth = matrix_smooth - _np.mean(matrix_smooth)
        self._Qm = Qm
        self._QmSmooth = Qm_smooth

        F = []
        for i in range(1, delta.shape[0]):
            file_name = _os.path.join(self._fringes_fold, "Fringe_%05d.fits" % i)
            fringe = _osu.load_fits(file_name)
            fringe_selected = fringe[:, idx]
            F.append(fringe_selected)
        F = _np.dstack(F)
        Qt = F - _np.mean(F)
        self._Qt = Qt

        R = _np.zeros(delta.shape[0] - 1)
        R_smooth = _np.zeros(delta.shape[0] - 1)
        for i in trange(delta.shape[0] - 1, desc=f"Comparing with synthetic data"):

            R[i] = _np.sum(Qm[:, :] * Qt[:, :, i]) / (
                _np.sum(Qm[:, :] ** 2) ** 0.5 * _np.sum(Qt[:, :, i] ** 2) ** 0.5
            )
            R_smooth[i] = _np.sum(Qm_smooth[:, :] * Qt[:, :, i]) / (
                _np.sum(Qm_smooth[:, :] ** 2) ** 0.5 * _np.sum(Qt[:, :, i] ** 2) ** 0.5
            )

        idp = _np.nanargmax(R)  # _np.where(R == max(R))
        idp_smooth = _np.nanargmax(R_smooth)
        self._idp = idp
        self._idp_smooth = idp_smooth
        piston = delta[idp]
        piston_smooth = delta[idp_smooth]
        return piston, piston_smooth

    def _readDeltaAndLambdaFromFringesFolder(
        self,
    ) -> tuple[_ot.ArrayLike, _ot.ArrayLike]:
        """
        Reads the delta piston and synthetic wavelength data from the fringes folder.

        Returns
        -------
        delta: ArrayLike
            Delta piston values
        lambda_synth_from_data: ArrayLike
            Synthetic wavelength values
        """
        delta = _osu.load_fits(
            _os.path.join(self._fringes_fold, "Differential_piston.fits")
        )
        lambda_synth_from_data = _osu.load_fits(
            _os.path.join(self._fringes_fold, "Lambda.fits")
        )
        lambda_synth_from_data = (_np.round(lambda_synth_from_data / 5) * 5).astype(int)

        return delta, lambda_synth_from_data

    # TODO: TO BE REMOVED
    def _applySmoothing(self, a: _ot.ArrayLike, WSZ: int):
        """'

        Parameters
        ----------
        a: NumPy 1-D array
            containing the data to be smoothed
        WSZ: int
            smoothing window size needs, which must be odd number,
            as in the original MATLAB implementation

        Returns
        -------
        smooth: numpy array
                smoothd data
        """
        out0 = _np.ma.convolve(a, _np.ones(WSZ, dtype=int), "valid") / WSZ
        r = _np.arange(1, WSZ - 1, 2)
        start = _np.ma.cumsum(a[: WSZ - 1])[::2] / r
        stop = (_np.ma.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
        return _np.ma.concatenate((start, out0, stop))
