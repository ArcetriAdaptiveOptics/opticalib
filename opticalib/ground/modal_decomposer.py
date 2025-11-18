"""
Modal Decomposer Library
========================
This module provides functions and utilities for generating Modal Surfaces.

Author(s)
---------
- Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) : Original Author,  Created in 2011-10-12
- Pietro Ferraiuolo (pietro.ferraiuolo@inaf.it) : Adapted in 2024 / Modified in 2025
- Matteo Menessini  (matteo.menessini@inaf.it) : Enhancement in 2025

Example
-------
Example usage of the ZernikeFitter class:

```python
# Create a sample wavefront image (e.g., 256x256 pixels)
size = 256
y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
radius = size / 2

# Create a circular pupil mask
pupil_mask = (x**2 + y**2) <= radius**2

# Generate a simulated wavefront with some aberrations
# Adding defocus (Z4) and astigmatism (Z5, Z6)
wavefront = np.random.normal(0, 0.1, (size, size))
wavefront = np.ma.masked_array(wavefront, mask=~pupil_mask)

# Initialize the Zernike fitter with a circular pupil
fitter = ZernikeFitter(fit_mask=pupil_mask)

# Fit Zernike modes 1-10 to the wavefront
modes_to_fit = list(range(1, 11))
coefficients, fitting_matrix = fitter.fit(wavefront, modes_to_fit)

print(f"Fitted Zernike coefficients: {coefficients}")

# Remove tip-tilt (modes 2 and 3) from the wavefront
corrected_wavefront = fitter.removeZernike(wavefront, zernike_index_vector=[2, 3])

# Generate a pure Zernike surface (e.g., coma, mode 7)
coma_surface = fitter.makeSurface(modes=[7])

# Fit modes on multiple ROIs and get global average
roi_coefficients = fitter.fitOnRoi(wavefront, modes2fit=[1, 2, 3], mode='global')
print(f"ROI-averaged coefficients: {roi_coefficients}")
```
"""

import numpy as _np
from . import roi as _roi
from abc import abstractmethod, ABC
from opticalib import typings as _t
from contextlib import contextmanager as _contextmanager
from arte.utils.zernike_generator import ZernikeGenerator as _ZernikeGenerator
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix as _gfkcm
from arte.utils.karhunen_loeve_generator import KarhunenLoeveGenerator as _KLGenerator
from arte.utils.rbf_generator import RBFGenerator as _RBFGenerator
from arte.types.mask import CircularMask as _CircularMask
from functools import lru_cache as _lru_cache


class _ModeFitter(ABC):
    """
    Class for fitting Zernike polynomials to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(
        self,
        fit_mask: _t.Optional[_t.ImageData | _CircularMask | _t.MaskData] = None,
        method: str = "COG",
    ):
        """
        Class for fitting Zernike polynomials to an image.

        Parameters
        ----------
        fit_mask : ImageData | CircularMask | MaskData, optional
            Mask to be used for fitting. Can be:
            - ImageData : A masked array from which a CircularMask is estimated.
            - CircularMask : A pre-defined CircularMask object.
            - MaskData : A boolean mask array.
        method : str, optional
            Method used by the `CircularMask.fromMaskedArray` function. Default is 'COG'
        """
        if fit_mask is not None:
            self.setFitMask(fit_mask=fit_mask, method=method)
        else:
            self._fit_mask = None
            self.auxmask = None
            self._mgen = None

    def _create_fitting_matrix(
        self, modes: list[int], mask: _t.MaskData
    ) -> _t.MatrixLike:
        """
        Create the fitting matrix for the given modes.

        Parameters
        ----------
        modes : list[int]
            List of modal indices.
        mask : MaskData
            Boolean mask defining the fitting area.

        Returns
        -------
        mat : MatrixLike
            Fitting matrix for the specified modes.
        """
        return _np.vstack(
            [self._get_mode_from_generator(zmode)[mask] for zmode in modes]
        )

    @abstractmethod
    def _create_modes_generator(self, mask: _t.MaskData) -> object:
        """
        Create the modes generator.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_create_modes_generator` method."
        )

    @abstractmethod
    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_get_mode_from_generator` method."
        )

    @property
    def fitMask(self) -> _t.ImageData:
        """
        Get the current fitting mask.

        Returns
        -------
        fit_mask : ImageData
            Current fitting mask.
        """
        return self.auxmask

    def setFitMask(
        self, fit_mask: _t.ImageData | _CircularMask | _t.MaskData, method: str = "COG"
    ) -> None:
        """
        Set the fitting mask.

        Parameters
        ----------
        fit_mask : ImageData | CircularMask | MaskData, optional
            Mask to be used for fitting. Can be:
            - ImageData : A masked array from which a CircularMask is estimated.
            - CircularMask : A pre-defined CircularMask object.
            - MaskData : A boolean mask array.
        method : str, optional
            Method used by the `CircularMask.fromMaskedArray` function. Default is 'COG'.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if isinstance(fit_mask, _CircularMask):
                self._fit_mask = fit_mask
            elif isinstance(fit_mask, _np.ma.masked_array):
                self._fit_mask = _CircularMask.fromMaskedArray(
                    _np.ma.masked_array(fit_mask, mask=fit_mask.mask.astype(bool)),
                    method=method,
                )
            elif _t.isinstance_(fit_mask, "MaskData"):
                cmask = _CircularMask.fromMaskedArray(
                    _np.ma.masked_array(
                        _np.zeros_like(fit_mask), mask=fit_mask.astype(bool)
                    ),
                    method="COG",
                )
                cmask._mask = fit_mask.astype(bool)
                self._fit_mask = cmask
            else:
                self._fit_mask = _CircularMask.fromMaskedArray(fit_mask, method=method)
        self.auxmask = self._fit_mask.mask()
        self._mgen = self._create_modes_generator(self._fit_mask)

    def fit(
        self, image: _t.ImageData, mode_index_vector: list[int]
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Fit Zernike modes to an image.

        Parameters
        ----------
        image : ImageData
            Image for modal fit.
        mode_index_vector : list[int]
            List containing the index of modes to be fitted.
            If they are Zernike modes, the first index is 1.

        Returns
        -------
        coeff : numpy array
            Vector of modal coefficients.
        mat : numpy array
            Modes matrix.
        """
        image = self._make_sure_on_cpu(image)

        with self._temporary_mgen_from_image(image) as (pimage, _):
            mask = pimage.mask == 0
            mat = self._create_fitting_matrix(mode_index_vector, mask)
            A = mat.T
            B = _np.transpose(pimage.compressed())
            coeffs = _np.linalg.lstsq(A, B, rcond=None)[0]
            return coeffs, A

    def fitOnRoi(
        self,
        image: _t.ImageData,
        modes2fit: _t.Optional[list[int]] = None,
        mode: str = "global",
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Computes modal coefficients over a segmented fitting area, i.e. a pupil
        mask divided into Regions Of Interest (ROI). The computation is based on
        the fitting of modes independently on each ROI.

        Parameters
        ----------
        image : ImageData
            Image for modal fit.
        modes2fit : list[int], optional
            List containing the index of modes to be fitted.
            If they are Zernike modes, the first index is 1.
        mode : str, optional
            Mode of fitting.
            - `global` will return the mean of the fitted coefficient of each ROI
            - `local` will return the vector of fitted coefficient for each ROI
            Default is 'global'.

        Returns
        -------
        coeff : numpy array
            Vector of modal coefficients.
        mat : numpy array
            Matrix of modal polynomials.
        """
        if mode not in ["global", "local"]:
            raise ValueError("mode must be 'global' or 'local'")
        roiimg = _roi.roiGenerator(image)
        nroi = len(roiimg)
        print("Found " + str(nroi) + " ROI")
        coeff = _np.zeros([nroi, len(modes2fit)])
        for i in range(nroi):
            img2fit = _np.ma.masked_array(image.data, mask=roiimg[i])
            cc, _ = self.fit(img2fit, modes2fit)
            coeff[i, :] = cc
        if mode == "global":
            coeff = coeff.mean(axis=0)
        return coeff

    def makeSurface(self, modes: list[int], image: _t.ImageData = None) -> _t.ImageData:
        """
        Generate modal surface from image.

        Parameters
        ----------
        image : ImageData, optional
            Image for fit. If no image is provided, it will be generated a surface,
            defined on a circular mask with amplitude 1.
        modes : list[int], optional
            List of modes indices. Defaults to [1].

        Returns
        -------
        surface : ImageData
            Generated modal surface.
        """
        if image is None and self._mgen is None:
            raise ValueError(
                "Either an image must be provided or a fitting mask must be set."
            )
        elif image is not None:
            image = self._make_sure_on_cpu(image)
            mm = _np.where(image.mask == 0)
            surface = _np.zeros(image.shape)
            coeff, mat = self.fit(image, modes)
            surface[mm] = _np.dot(mat, coeff)
            surface = _np.ma.masked_array(surface, mask=image.mask)
        elif image is None and self._mgen is not None:
            if isinstance(modes, int):
                modes = [modes]
            surface = self._get_mode_from_generator(modes[0])
            if len(modes) > 1:
                for mode in modes[1:]:
                    surface += self._get_mode_from_generator(mode)
            surface[self.auxmask == 1] = 0.0
        return surface

    def filterModes(
        self, image: _t.ImageData, mode_index_vector: list[int]
    ) -> _t.ImageData:
        """
        Remove modes from the image using the current fit mask.

        Parameters
        ----------
        image : ImageData
            Image from which to remove modes.
        zernike_index_vector : list[int], optional
            List of mode indices to be removed.

        Returns
        -------
        new_ima : ImageData
            Filtered image.
        """
        # # try this:
        # if all([self._mgen is not None, self._fit_mask is not None]):
        #     surf = self.makeSurface(mode_index_vector, None)
        # else:
        image = self._make_sure_on_cpu(image)
        surf = self.makeSurface(mode_index_vector, image)
        return _np.ma.masked_array((image - surf).data, mask=image.mask)

    @_contextmanager
    def no_mask(self):
        """
        Context manager to temporarily clear the fitting mask and Zernike generator.

        Usage
        -----
        with zfitter.no_mask():
            coeffs, mat = zfitter.fit(image, modes)

        Within the context, ``self._fit_mask``, ``self._zgen`` and ``self.auxmask``
        are set to ``None`` so that ``fit`` will lazily create a temporary mask
        from the provided image. On exit, the previous values are restored.
        """
        prev_fit_mask = self._fit_mask
        prev_mgen = self._mgen
        prev_auxmask = self.auxmask.copy()
        try:
            self._fit_mask = None
            self._mgen = None
            self.auxmask = None
            yield self
        finally:
            self._fit_mask = prev_fit_mask
            self._mgen = prev_mgen
            self.auxmask = prev_auxmask

    @_contextmanager
    def _temporary_mgen_from_image(self, image: _t.ImageData):
        """
        Context manager to temporarily create a ModalGenerator from an image
        when self._mgen is None, and restore the original state afterwards.

        Parameters
        ----------
        image : ImageData
            Image from which to create a temporary ModalGenerator

        Yields
        ------
        tuple
            (modified_image, was_temporary) where was_temporary indicates if a temp generator was created
        """
        prev_mgen = self._mgen
        was_temporary = False

        try:
            if self._mgen is None:
                self._mgen = self._create_fit_mask_from_img(image)
                was_temporary = True
            image = _np.ma.masked_array(
                image.copy().data, mask=self._mgen._boolean_mask.copy()
            )
            yield image, was_temporary
        finally:
            if was_temporary:
                self._mgen = prev_mgen

    def _create_fit_mask_from_img(self, image: _t.ImageData) -> _CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        fit_mask : CircularMask
            Default fitting mask.
        """
        if not isinstance(image, _np.ma.masked_array):
            try:
                image = _np.ma.masked_array(image, mask=image == 0)
            except Exception as e:
                raise ValueError(
                    "Input image must be a numpy masked array or convertible to one."
                ) from e
        cmask = _CircularMask(image.shape)
        cmask._mask = image.mask
        mgen = self._create_modes_generator(cmask)
        return mgen

    def _make_sure_on_cpu(self, img: _t.ImageData) -> _t.ImageData:
        """
        Ensure the image is on CPU.

        Parameters
        ----------
        img : ImageData
            Input image.

        Returns
        -------
        img_cpu : ImageData
            Image on CPU.
        """
        if isinstance(img, _np.ma.MaskedArray):
            return img
        else:
            import xupy as xp

            if isinstance(img, xp.ma.MaskedArray):
                img = img.asmarray()
            elif isinstance(img, xp.ndarray):
                img = img.get()
        return img


class ZernikeFitter(_ModeFitter):
    """
    Class for fitting Zernike polynomials to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(self, fit_mask: _t.Optional[_t.ImageData] = None, method: str = "COG"):
        """The Initiator."""
        super().__init__(fit_mask)

    def removeZernike(
        self, image: _t.ImageData, zernike_index_vector: list[int] = None
    ) -> _t.ImageData:
        """
        Remove Zernike modes from the image using the current fit mask.

        Parameters
        ----------
        image : ImageData
            Image from which to remove Zernike modes.
        zernike_index_vector : list[int], optional
            List of Zernike mode indices to be removed. Default is [1, 2, 3].

        Returns
        -------
        new_ima : ImageData
            Image with Zernike modes removed.
        """
        if zernike_index_vector is None:
            zernike_index_vector = [1, 2, 3]
        return self.filterModes(image=image, mode_index_vector=zernike_index_vector)

    def _create_modes_generator(self, mask: _CircularMask) -> _CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        zgen : ZernikeGenerator
            The Zernike Generator defined on the created Circular Mask.
        """
        return _ZernikeGenerator(mask)

    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.

        Parameters
        ----------
        mode_index : int
            Index of the Zernike mode to retrieve.

        Returns
        -------
        mode_image : ImageData
            The Zernike mode image corresponding to the given index.
        """
        return self._mgen.getZernike(mode_index).copy()


class KLFitter(_ModeFitter):
    """
    Class for fitting Karhunen-Loeve modes to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(
        self,
        nKLModes: int,
        fit_mask: _t.Optional[_t.ImageData] = None,
        method: str = "COG",
    ):
        """The Initiator"""
        self.nModes = nKLModes
        super().__init__(fit_mask, method)

    def _create_modes_generator(self, mask: _CircularMask) -> _CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        klgen : KarhunenLoeveGenerator
            The Karhunen-Loeve Generator defined on the created Circular Mask.
        """
        zz = _ZernikeGenerator(mask)
        zbase = _np.rollaxis(
            _np.ma.masked_array([zz.getZernike(n) for n in range(2, self.nModes + 2)]),
            0,
            3,
        )
        kl = _KLGenerator(mask, _gfkcm(self.nModes))
        kl.generateFromBase(zbase)
        return kl

    @_lru_cache
    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.

        Parameters
        ----------
        mode_index : int
            Index of the mode to retrieve.

        Returns
        -------
        mode_image : ImageData
            The mode image corresponding to the given index.
        """
        return self._mgen.getKL(mode_index)


class RBFitter(_ModeFitter):
    """
    Class for fitting radial-basis functions to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(
        self,
        coords: _t.ArrayLike = None,
        rbfFunction: str = "TPS_RBF",
        eps: float = 1.0,
        fit_mask: _t.Optional[_t.ImageData] = None,
        method: str = "COG",
    ):
        """The Initiator"""
        self.rbfFunction = rbfFunction
        self._coordinates = coords
        self._eps = eps
        super().__init__(fit_mask, method)

    def _create_modes_generator(self, mask: _CircularMask) -> _CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        zgen : ZernikeGenerator
            The Zernike Generator defined on the created Circular Mask.
        """
        if self._coordinates is None:
            npmask = mask.mask()
            ny, nx = npmask.shape
            x = _np.arange(nx)
            y = _np.arange(ny)
            X, Y = _np.meshgrid(x, y)
            self._coordinates = _np.vstack((X[~npmask].ravel(), Y[~npmask].ravel())).T
        rbf = _RBFGenerator(
            mask, self._coordinates, rbfFunction=self.rbfFunction, eps=self._eps
        )
        rbf.generate()
        return rbf

    @_lru_cache
    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.

        Parameters
        ----------
        mode_index : int
            Index of the mode to retrieve.

        Returns
        -------
        mode_image : ImageData
            The mode image corresponding to the given index.
        """
        return self._mgen.getRBF(mode_index)
