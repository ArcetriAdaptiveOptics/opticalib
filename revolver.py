import os as _os
import numpy as _np
from opticalib import typings as _ot
from opticalib.ground import osutils as _osu
from opticalib.ground.logger import SystemLogger as _SL
from opticalib.devices.camera import AVTCamera as _cam
from opticalib.core.fitsarray import fits_array as _fits_array
from opticalib import folders as _fn
from scipy import ndimage as _ndi
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from photutils.centroids import centroid_com


def _get_tunable_filter():
    """
    initiate the tunable filter with standard parameters
    """
    from plico_motor import motor  # type: ignore

    return motor("192.168.29.4", 7100, axis=0)


_FILTER_BANDWIDTH_MODE = {"narrow": 8, "medium": 4, "wide": 2, "black": 1}


class ThorRevolver:
    """
    Class defining the Petalometer bench's "Thor's Revolver".

    This instrument is composed of:
    - A tunable filter, which operates in the range 500-750 nm
    - A system composed of injected light from a lamp, which goes through a 6 lens
    system which gets beam-splitted to the PetalMirror and to the camera, forming the
    6 spot pattern for piston phasing.
    - An AVT Camera.

    Parameters
    ----------
    camera: AVTCamera
        The camera used to acquire images in the TR system
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
        if camera is None:
            camera = "SplCam0"
        elif isinstance(camera, str):
            camera = _cam(name=camera)
        if tunable_filter is None:
            tunable_filter = _get_tunable_filter()
        self._camera = camera
        self._filter = tunable_filter
        self._darkFrame1sec = None
        self._curr_exptime = None
        self._tnfringes = tnfringes
        self._logger = _SL(__class__)

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

        Parameters:
        -----------
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

        Parameters:
        -----------
        exptime : float
            the exposure time in [s]
        nframes : int
            the number of frames to acquire

        Returns:
        --------
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

    def acquire(
        self,
        exptime: float,
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

    def analysis(self, tn: str | None = None, process_psfs: bool = True):
        """
        Analyze the measurements acquired with the `acquire` method, by detecting
        the PSF centroids and cropping the frames around them.

        Parameters:
        -----------
        tn : str
            Tracking number of the measurement to be processed. If None, the last
            acquired measurement is processed.
        """
        if process_psfs:
            self.process_psfs(tn=tn)

        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        lambda_vector = _osu.load_fits(_os.path.join(datapath, "lambda_vector.fits"))

        cubelist = _osu.getFileList(fold=datapath, key="cube")




    def process_psfs(
        self,
        tn: str | None = None,
        n_psfs: int = 6,
        min_pixels: int = 30,
        half_size: int = 70,
        remove_dark: bool = True,
        remove_median: bool = False,
        angles: list[float] | None = 0.0,
    ):
        """
        Process the raw frames acquired with the `acquire` method, by detecting
        the PSF centroids and cropping the frames around them.

        Parameters:
        -----------
        tn : str
            Tracking number of the measurement to be processed.
        n_psfs : int
            The number of PSFs to detect. 
            
            By default, 6.
        min_pixels : int
            The minimum number of pixels required for a detection to be
            considered a PSF.
            
            By default, 30.
        half_size : int
            The half size of the cropped images, in pixels. 
            
            By default, 70, which corresponds to a 140x140 pixels crop.
        remove_dark : bool
            If True, removes the dark frame from each frame before processing.
            
            By default, True.
        remove_median : bool
            If True, removes the median value from each frame before processing.
            
            By default, False.
        angles : list of float | None
            List of angles in degrees to rotate each PSF crop. If None, no rotation is
            applied.

            By default, 0.
        """
        tn = tn or self._last_measure_tn
        if tn is None:
            self._logger.error(
                "No tracking number provided and no previous measurement found."
            )
            raise ValueError(
                "No tracking number provided and no previous measurement found."
            )

        datapath = _os.path.join(_fn.SPL_DATA_ROOT_FOLDER, tn)
        filelist = _osu.getFileList(fold=datapath, key="rawframe")
        rawlist = [_osu.load_fits(x) for x in filelist]

        # We prepare the PSF cubes list, which will be filled with the rotated
        # crops for each PSF, at each wavelength
        psf_cubes = []
        for i in range(n_psfs):
            psf_cubes.append([])

        for img, filename in zip(rawlist, filelist):

            # Removing the saved dark frame
            if remove_dark:
                dark = self._get_dark_frame(tn)
                print("Removing Dark")
                img = img - dark * img.header["EXPTIME"]
                img.header["RDARK"] = True
            
            if remove_median:
                ... # ?? 

            # detect the centroids positions in the raw frame
            centroids = self.detect_psf_centroids(
                img, n_psf=n_psfs, min_pixels=min_pixels
            )

            crops, _ = self.crop_around_centroids(img, centroids, half_size=half_size)

            for i, crop in enumerate(crops):
                header = img.header.copy()
                header["PSFNUM"] = i

                # This save for now for debugging
                crop.writeto(filename.replace("rawframe", f"psf{i}"), overwrite=True)

                crop = self.rotate_psf(crop, centroid=centroids[i], angle=angles[i])
                header["ROTATED"] = (True, "was de-rotated")
                header["ROTANG"] = (
                    angles[i],
                    "psf rotation angle wrt the vertical in degrees",
                )
                header["CENTX"] = (
                    centroids[i][0],
                    "x coordinate of the PSF centroid in the original frame",
                )
                header["CENTY"] = (
                    centroids[i][1],
                    "y coordinate of the PSF centroid in the original frame",
                )

                ncentr = centroid_com(crop.data)
                header["NCENTX"] = (
                    ncentr[0],
                    "x coordinate of the PSF centroid in the cropped frame",
                )
                header["NCENTY"] = (
                    ncentr[1],
                    "y coordinate of the PSF centroid in the cropped frame",
                )
                crop.header = header

                psf_cubes[i].append(crop)

        # Save the PSF cubes
        for i, cube in enumerate(psf_cubes):
            cube = _fits_array(data=_np.ma.dstack(cube), header=cube[0].header)
            cube.writeto(_os.path.join(datapath, f"psf{i}_cube.fits"), overwrite=True)


    def rotate_psf(
        self,
        cropped_img: _ot.ImageData,
        centroid: tuple[int, int],
        angle: float,
        order: int = 3,
        cval: float = 0.0,
    ) -> _ot.ImageData:
        """
        Rotate the cropped PSF image by a given angle around its centroid.

        Parameters:
        -----------
        cropped_img : ImageData
            The cropped PSF image to be rotated.
        centroid : tuple of int
            The (x, y) coordinates of the centroid around which to rotate the
            image.
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

        Returns:
        --------
        rotated_img : ImageData
            The rotated PSF image.
        """
        raise NotImplementedError("Rotation of PSF crops is not implemented yet.")

    def detect_psf_centroids(
        self,
        raw_frame: _ot.ImageData,
        n_psf: int = 6,
        nsigma: float = 5.0,
        min_pixels: int = 30,
    ) -> list[tuple[int, int]]:
        """
        Detect the centroids of the PSFs in the raw frame acquired with the `acquire` method.

        Being this class specifically designed for the SPL system, it is expected
        that 6 PSFs are found, arranged in a hexagonal pattern.
        The detection is based on abackground estimation, followed by a
        thresholding.

        Parameters:
        -----------
        raw_frame : ImageData
            The raw frame acquired with the `acquire` method, from which the PSF
            centroids are to be detected.
        n_psf : int
            The number of PSFs to detect. By default, 6.
        nsigma : float
            The number of sigma above the background to use as threshold for the
            detection. By default, 5.0.
        min_pixels : int
            The minimum number of pixels required for a detection to be
            considered a PSF. By default, 30.

        Returns:
        --------
        centroids_xy : list of tuples
            The list of centroids of the detected PSFs, in (x, y) format.
        """
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

        # keep the 6 largest detections
        centroids = sorted(centroids, key=lambda t: t[2], reverse=True)[:n_psf]
        centroids_xy = [(int(c[0].round()), int(c[1].round())) for c in centroids]

        # Sorting Top-Bottom, Left-Right
        centroids_xy = sorted(centroids_xy, key=lambda p: (p[1], p[0]))

        return centroids_xy

    def crop_around_centroids(
        self,
        frame: _ot.ImageData,
        centroids: list[tuple[int, int]],
        half_size: int = 80,
    ) -> tuple[list[_ot.ImageData], list[tuple[int, int, int, int]]]:
        """
        Crop the input frame around the detected centroids, to obtain the 6 PSF
        images.

        Parameters:
        -----------
        frame : ImageData
            The input frame to crop, typically the raw frame acquired with the
            `acquire` method.
        centroids : list of tuples
            The list of centroids of the detected PSFs, in (x, y) format, as
            obtained from `detect_psf_centroids` method.
        half_size : int
            The half size of the cropped images, in pixels. By default, 80,
            which corresponds to a 160x160 pixels crop.
        """
        crops = []
        boxes = []
        h, w = frame.shape
        for xc, yc in centroids:
            x0 = max(0, xc - half_size)
            x1 = min(w, xc + half_size + 1)
            y0 = max(0, yc - half_size)
            y1 = min(h, yc + half_size + 1)
            crops.append(frame[y0:y1, x0:x1].copy())

            # Useful for plotting
            boxes.append((x0, y0, x1 - x0, y1 - y0))

        return crops, boxes

    def _mediam_mad_background_subtraction(
        self, frame: _ot.ImageData
    ) -> tuple[float, float]:
        """
        Estimate the background and noise of the input frame using the median and MAD, and return the background-subtracted frame.

        Parameters:
        -----------
        frame : ImageData
            The input frame from which to estimate the background and noise.

        Returns:
        --------
        bkg : float
            The estimated background level.
        sigma : float
            The estimated noise level (standard deviation).
        """
        bkg = _np.median(frame)
        mad = _np.median(_np.abs(frame - bkg))
        sigma = 1.4826 * mad if mad > 0 else _np.std(frame)
        return frame - mad, sigma

    def _create_mask_from_threshold(
        self, frame: _ot.ImageData, bkg: float, nsigma: float = 5.0
    ) -> _ot.MaskData:
        """
        Create a binary mask from the input frame by applying a threshold based on the estimated background and noise.

        Parameters:
        -----------
        frame : ImageData
            The input frame from which to create the mask.
        bkg : float
            The estimated background level.
        sigma : float
            The estimated noise level (standard deviation).
        nsigma : float
            The number of sigma above the background to use as threshold for the detection. By default, 5.0.

        Returns:
        --------
        mask : MaskData
            The binary mask created from the input frame.
        """
        frame, sigma = self._mediam_mad_background_subtraction(frame)
        threshold = bkg + nsigma * sigma
        mask = frame > threshold
        return mask.astype(bool)

    def _get_dark_frame(self, tn: str) -> _ot.ImageData:
        """
        Get the dark frame for the camera from the last measurement tracking number folder.
        If the dark frame is not found, it returns the dark frame acquired with the `acquire_dark_frame`
        method, and saves it in the tracking number folder.

        Parameters:
        -----------
        tn : str
            Tracking number of the measurement to get the dark frame from.

        Returns:
        --------
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
