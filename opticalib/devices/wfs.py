from ._API.base_devices import BaseWavefrontSensor
"""
This module contains the high-level classes for the use of Ingot Wavefront Sensor devices.
Integrates the Ingot Adaptive Optics Testbench directly into the opticalib framework.
"""

import os as _os
import numpy as _np
import time as _time
import shutil as _sh
import subprocess as _sb
# Import your exact native testbench components

from ingot2 import INGOT
from ..core import _types as _ot
from ..core import root as _fn
from ..devices import cameras as _cam
from ..ground import osutils as _osu
from ..ground import geometry as _geo
from ..ground.logger import SystemLogger as _SL
from ..analyzer.image_processing import mode_rebinner as _modeRebinner
from ..core.config import get_device_config as _gdc
from skimage.transform import rotate as _rot

global _folds
_folds = _fn.folders
_confReader = _fn.ConfSettingReader4D
_OPDIMG = _folds.OPD_IMAGES_ROOT_FOLDER


class Ingot(BaseWavefrontSensor):
    """
    Class for the Ingot Wavefront Sensor (IWS).
    Acts as a direct, plug-and-play alternative to the Interferometer.
    """

    def __init__(self, camera: str|_ot.CameraDevice):
        """
        The constructor initializes local network bindings, the calculation kernel,
        and matches alignment transformation constraints.
        """
        self._name = "Ingot"
        self._logger = _SL(the_class=__class__)

        # 1. Initialize persistent camera connection via your wrapper
        if isinstance(camera, str):
            self._camera = _cam.GigaVision(camera)
            self._config = _gdc('WFS', 'INGOT')
        elif _ot.isinstance_(camera, 'CameraDevice'):
            self._camera = camera
            self._config = {}

        # 2. Initialize your computation kernel with its exact class name
        self._kernel = INGOT()

        # Sane defaults for pupil parsing parameters matching your class variables
        self.n_pup = 3
        self.rmin = 250
        self.rmax = 260
        self.sigma_threshold = 6

        # Spatial Calibration Configuration Switches
        self._pupil_radius = self._config.get('pupil_radius', None)
        self._pupil_centers = self._config.get('pupils_centers', None)
        self._pupil_info = _np.asarray([(x, y, r) for (x, y), r in zip(self._pupil_centers,[self._pupil_radius]*3)], dtype=[('xc', float), ('yc', float), ('radius', float)]) if self._pupil_centers and self._pupil_radius else None
        self._camera_binning = self._config.get('camera_binning', 1)

        # Internal configuration storage
        self.pupdata = None
        self.exposure_time = None
        self.set_exptime(self._config.get('camera_base_exptime', 2000))  # Default exposure time from camera

    @property
    def pupil_info(self):
        return self._pupil_info.copy()

    def acquire_pupils(
        self,
        frames: _ot.ImageData|_ot.CubeData|list[_ot.ImageData]|int,
        detect_pupils: bool = False
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        In-memory translation of your native bench tracking logic.
        Detects, sorts, optionally patches, extracts, and rotates pupil footprints 
        across the physical sensing plane without intermediate file I/O operations.
        """
        if isinstance(frames, int):
            frames = self._camera.acquire_frames(frames)
            frames = _modeRebinner(frames, self._camera_binning, 'sum') if self._camera_binning > 1 else frames

        # 1. Execute pupil detection step (if needed)
        if detect_pupils:
            self._detect_pupils(frames)

        pupil_images = self._extract_pupils(frames)
        return pupil_images

    def acquire_map(
        self, 
        output_type: str = 'slopes',
        **kwargs
    ) -> _ot.ImageData:
        """
        Acquires physical camera frames (averaging if nframes > 1), processes them 
        through the full geometry pipeline, and directly returns the raw gradient slopes (Sx, Sy).
        """
        image = self._camera.acquire_frames(kwargs.get('nframes', 1))
        if self._camera_binning > 1:
            image = _modeRebinner(image, self._camera_binning, 'sum')

        the_output = self.acquire_pupils(frames=image, **kwargs)

        if output_type == 'slopes':
            # Map gradients across normalized spatial matrices
            Sx, Sy = self._compute_slopes_from_pupils(the_output)
            the_output = _np.vstack([Sx, Sy])

        return the_output
        
    def _compute_slopes_from_pupils(self, pupil_images: _ot.ImageData):
        """Il nuovo kernel.GetSignals()"""
        A = pupil_images[0,:,:]
        B = pupil_images[1,:,:]
        C = pupil_images[2,:,:]
        # Suppress/hide the warning
        _np.seterr(invalid='ignore')
        Sx = (B-C)/(A+B+C)
        Sy = A/(A+B+C)
        mask_Sx = _np.isnan(Sx)
        mask_Sy = _np.isnan(Sy)
        Sx = _np.ma.masked_array(Sx, mask=mask_Sx)
        Sy = _np.ma.masked_array(Sy, mask=mask_Sy)
        return Sx, Sy
    
    def _detect_pupils(self, image: _ot.ImageData):
        import warnings
        from skimage import feature
        from skimage.transform import hough_circle, hough_circle_peaks

        dtype = [('xc',float),('yc',float),('radius',float)]
        pupdata = _np.array([], dtype=dtype)

        edges = feature.canny(image.astype(_np.float32), sigma=self.sigma_threshold)             #find edges with canny algorithm
        #finding the best fitting radius
        hough_radii = _np.arange(self.rmin, self.rmax+1, 1)
        hough_res = hough_circle(edges, hough_radii)
        # print(hough_res)
        _, cx, cy, hough_radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1, min_xdistance=self.rmin, min_ydistance=self.rmin)
        # print(hough_radii)
        if hough_radii==self.rmin or hough_radii==self.rmax:
            warnings.warn('Best fitting radius has reached the limit of the input search range. Consider extending the search range')
        #find best matching pupils
        hough_res = hough_circle(edges, hough_radii)
        # plt.imshow(hough_res[0])
        # plt.show()
        _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.n_pup, min_xdistance=self.rmin, min_ydistance=int(_np.sqrt(3)*self.rmin))
        # print(accums, cy, cx, radii)

        if len(cx) < self.n_pup:
            raise RuntimeError(f"Not enough pupils detected. Found {len(cx)} pupils, expected {self.n_pup}.")

        for center_y, center_x, radius in zip(cy, cx, radii):
            pupdata = _np.append(pupdata, _np.array((center_x, center_y, radius), dtype=dtype))
        
        temp = _np.sort(pupdata, order=['yc','xc'])  #first pupil on the top
        temp2 = _np.sort(temp[1:], order=['xc'])     #second pupil on the left, third on the right        
        pupdata = _np.append(temp[0],temp2)
        self._pupil_info = pupdata


    def _extract_pupils(self, image: _ot.ImageData):
        r = int(self._pupil_info['radius'][0])
        shape = tuple([2*r+1]*2)
        pupil = _geo.draw_circular_pupil(shape, r)
        # r = int(pupdata['radius'][0])
        # nrows = 2*r+1
        # ncols = 2*r+1
        # xv, yv = np.meshgrid(np.linspace(-1,1,nrows), np.linspace(-1,1,ncols))
        # outer_disk_mask = np.sqrt(xv**2+yv**2) > 1
        # collect data and store in pup_stack
        pup_stack = _np.ma.zeros((self.n_pup, shape[0], shape[1]))
        for pp in range(self.n_pup):
            xc = int(self._pupil_info['xc'][pp])
            yc = int(self._pupil_info['yc'][pp])
            pup_stack[pp,:,:] = image[(yc-r):(yc+r+1),(xc-r):(xc+r+1)]
            pup_stack[pp, pupil] = _np.nan
            pup_stack[pp,:,:].mask = pupil
        
        for pup, angle in zip([1,2], [60, -60]):
            pup_stack[pup] = _rot(pup_stack[pup], angle)
            pup_stack[pup] = _np.flipud(pup_stack[pup])

        pup_stack /= _np.sum(pup_stack, axis=0)

        return pup_stack

    def acquire_detector(self, nframes: int = 1) -> _ot.ImageData:
        """
        Acquires raw un-processed detector data directly from the camera sensor,
        leaving geometry masking and sub-pupil transformations bypassed.
        """
        return self._camera.acquire_frames(nframes)


    def set_exptime(self, exposure_ms: int) -> None:
        """Sets internal active target integrated exposure times for sequential loops."""
        if not exposure_ms == self.exposure_time:
            self.exposure_time = exposure_ms
            self._camera.set_exptime(exposure_ms*1000)  # Convert ms to seconds for the camera API
            self._logger.info(f"Ingot camera integration profile modified to: {exposure_ms}ms")


    def get_exptime(self) -> int:
        """Returns the current exposure time in milliseconds."""
        return self.exposure_time


    def __repr__(self) -> str:
        return f"{self._name}(camera={self._camera._name}, n_pupils={self.n_pup}, exposure_time={self.exposure_time}ms)"


class ShackHartmann(BaseWavefrontSensor): ...


class Pyramid(BaseWavefrontSensor): ...


class BiOEdge(BaseWavefrontSensor): ...
