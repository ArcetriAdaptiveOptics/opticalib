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
from .logger import SystemLogger as _SL
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
        self._logger.info('Getting fitting matrix for modes: ' + str(modes))
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
        self._logger.info('Creating Fitting Mask')
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
        self._logger.info('Fitting image with modal decomposition')

        # FIXME: now handles the case of mgen is available, but
        # need to rethink how it works when no mask is available
        with self._temporary_mgen_from_image(image) as (_, _):
            mask = image.mask == 0
            mat = self._create_fitting_matrix(mode_index_vector, mask)
            self._logger.info('Solving least squares for fitting coefficients')
            A = mat.T
            B = _np.transpose(image.compressed())
            coeffs = _np.linalg.lstsq(A, B, rcond=None)[0]
            return coeffs, A

    def fitOnRoi(
        self,
        image: _t.ImageData,
        modes2fit: _t.Optional[list[int]] = None,
        mode: str = "global",
    ) -> _t.ArrayLike:
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
        self._logger.info(f'Fitting modes {modes2fit} on image\'s ROIs with mode {mode}')
        print("Found " + str(nroi) + " ROI")
        self._logger.info(f'Found {nroi} ROIs to fit')
        coeff = _np.zeros([nroi, len(modes2fit)])
        for i in range(nroi):
            img2fit = _np.ma.masked_array(image.data, mask=roiimg[i])
            cc, _ = self.fit(img2fit, modes2fit)
            coeff[i, :] = cc
        if mode == "global":
            coeff = coeff.mean(axis=0)
        return coeff

    def makeSurface(
        self,
        modes_indices: list[int],
        image: _t.ImageData = None,
        mode: str = "full-aperture",
        **kwargs: dict[str, _t.Any]
    ) -> _t.ImageData:
        """
        Generate modal surface from image.

        Parameters
        ----------
        modes_indices : list[int], optional
            List of modes indices. Defaults to [1].
        image : ImageData, optional
            Image to fit to retrieve the modal coefficients needed for the surface to compute.
            If no image is provided, a surface defined in the `fitter` pupil normalized at 1
            is generated.

            If the additional argument `coeffs` is provided, they will be used
            instead of computing them from the image, and the latter is interpreted as
            the mask where the surface is defined.

            If also the `mat` argument is provided, it will be used as fitting matrix
            instead of computing it from the image, leaving the `image` argument not
            needed.
        mode : str, optional
            If more than one ROI is detected, it's the mode of ROI fitting. Options are:
            - `full-aperture` : generate the surface on the full aperture pupil (as if no ROIs were present)
            - `global` : will be created a surface from the mean of the modal coefficients of each fitted ROI
            - `local` : will return a surface in which heach roi has it's own modal surface reconstructed inside

            Default is 'full-aperture'.

        **kwargs : dict, optional
            Additional arguments.
            - coeffs : ArrayLike
                Pre-computed modal coefficients to generate the surface.
            - mat : MatrixLike
                Pre-computed fitting matrix.
            - rois : list[MaskData]
                List of ROIs to generate the surface on, following the `mode`
                argument.

        Returns
        -------
        surface : ImageData
            Generated modal surface.
        """
        coeffs = kwargs.get("coeffs", None)
        mat = kwargs.get("mat", None)
        k_rois = kwargs.get("rois", None)
        
        self._logger.info(f'Generating modal surface for modes {modes_indices} with mode {mode}')

        if image is None and self._mgen is None:
            self._logger.error('No image or fitting mask available to generate surface')
            raise ValueError(
                "Either an image must be provided or a fitting mask must be set."
            )

        # An image has been passed
        elif image is not None:

            image = self._make_sure_on_cpu(image)

            # Handle the ROIs case
            roiimg = _roi.roiGenerator(image)
            nroi = len(roiimg)

            # FIXME: handle the case of passed ROIs
            # in particular, new image should only have those passed rois
            # or th left-over rois should be zeroed (idk, it's a modal surface...)
            if k_rois is not None:
                import warnings

                warnings.warn(
                    "Overriding automatically detected ROIs ("
                    + str(nroi)
                    + ") with provided ones ("
                    + str(len(k_rois))
                    + ").",
                    UserWarning,
                )
                roiimg = k_rois
                nroi = len(roiimg)

            # Got more than one ROI branch
            if nroi > 1 and mode != "full-aperture":

                # Here we don't try to overwrite coeffs/mat, as it would not make sense

                # LOCAL
                if mode == "local":
                    print("Found " + str(nroi) + " ROI")
                    surfs = []
                    for r in roiimg:
                        img2fit = _np.ma.masked_array(image.data, mask=r)

                        # Considering to use the `no_mask` context manager
                        # Does it make sense to have a local fitting on a global mask?
                        # TODO: evaluate this point
                        # UPDATE: it indeed fitted the same coefficients on all ROIs
                        # so, using `no_mask`
                        # with self.no_mask():
                        #     # Here `makeSurface` goes always to the Single ROI branch
                        #     # and with no mask
                        surf = self.makeSurface(modes_indices, img2fit)

                        surfs.append(surf)
                    surface = _np.ma.empty_like(image)
                    surface.mask = image.mask.copy()
                    for i in range(nroi):
                        surface.data[roiimg[i] == 0] = surfs[i].data[roiimg[i] == 0]

                # GLOBAL
                elif mode == "global":
                    if coeffs is None:
                        coeffs = self.fitOnRoi(
                            image, modes2fit=modes_indices, mode="global"
                        )
                    surface = _np.ma.zeros_like(image)
                    for r in roiimg:
                        with self.temporary_fit_mask(r):
                            mat = self._create_fitting_matrix(modes_indices, r)
                        surface.data[r] = _np.dot(mat.T, coeffs)

                else:
                    raise ValueError("mode for ROI fitting must be 'global' or 'local'")

            # Single ROI branch
            else:

                # Got an image with no ROIs, or asked for full-aperture surface

                # Extract the correct mask indices to use
                # We handle the case in which we have a fitting mask (auxmask), so we need to
                # recreate the surface on that fitting mask, and then remask the result
                with self._temporary_mgen_from_image(image) as (pimage, _):
                    fm = pimage.mask == 0
                    fmidx_ = _np.where(pimage.mask == 0)

                # We did not get coeffs/mat, so we need to fit
                if coeffs is None and mat is None:
                    coeffs, mat = self.fit(image, modes_indices)

                # we interpret the passed image argument as the mask for the Matrix computation
                elif mat is None:
                    # Extra safety in case we don't have a fitting mask initialized
                    mat = self._create_fitting_matrix(modes_indices, fm)

                # TODO: consider the case mat is passed but not coeffs? Not a lot of sense...

                # Check matrix orientation, for extra safety
                if not mat.shape[1] < mat.shape[0]:
                    mat = mat.T

                surface = _np.ma.zeros_like(image)
                surface[fmidx_] = _np.dot(mat, coeffs)

                # Remasking
                surface = _np.ma.masked_array(surface, mask=image.mask)

        # No image, but a fitting mask is available
        elif self._mgen is not None:

            if isinstance(modes_indices, int):
                modes_indices = [modes_indices]

            surface = self._get_mode_from_generator(modes_indices[0])

            if len(modes_indices) > 1:
                for mode in modes_indices[1:]:
                    surface += self._get_mode_from_generator(mode)

            surface[self.auxmask == 1] = 0.0

        return surface

    def filterModes(
        self, image: _t.ImageData, mode_index_vector: list[int], mode: str = "global"
    ) -> _t.ImageData:
        """
        Remove modes from the image using the current fit mask.

        Parameters
        ----------
        image : ImageData
            Image from which to remove modes.
        zernike_index_vector : list[int], optional
            List of mode indices to be removed.
        mode : str
            If more than one ROI is found in the fitting mask, this parameter
            controls how the modes are computed:
            - `global` will compute the mean of the fitted coefficient of each ROI
            - `local` will compute the fitted coefficient for each ROI

            Defaults to 'global'.

        Returns
        -------
        new_ima : ImageData
            Filtered image.
        """
        self._logger.info(f'Removing modes {mode_index_vector} from image')
        image = self._make_sure_on_cpu(image)
        surf = self.makeSurface(mode_index_vector, image, mode=mode)
        self._logger.info('Subtraction...')
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
        self._logger.warning('Entering the `no mask` context...')
        prev_fit_mask = self._fit_mask
        prev_mgen = self._mgen
        prev_auxmask = self.auxmask.copy()
        try:
            self._logger.info('Temporarily removing fitting mask and modal generator')
            self._fit_mask = None
            self._mgen = None
            self.auxmask = None
            yield self
        finally:
            self._logger.info('Restoring previous fitting mask and modal generator')
            self._fit_mask = prev_fit_mask
            self._mgen = prev_mgen
            self.auxmask = prev_auxmask

    @_contextmanager
    def temporary_fit_mask(self, fit_mask: _t.MaskData):
        """
        Context manager to temporarily set a fitting mask.

        Parameters
        ----------
        fit_mask : ImageData
            Mask to be used for fitting.

        Yields
        ------
        None
        """
        self._logger.warning('Entering the `temporary fit mask` context')
        prev_fit_mask = self._fit_mask
        prev_mgen = self._mgen
        prev_auxmask = self.auxmask.copy() if not self.auxmask is None else None
        try:
            self._logger.info('Temporarily setting a new fitting mask')
            if prev_fit_mask is None:
                self.setFitMask(fit_mask)
            yield
        finally:
            self._logger.info('Restoring previous fitting mask')
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
            (modified_image, was_temporary), where was_temporary indicates if a temp generator was created
        """
        prev_mgen = self._mgen
        was_temporary = False
        self._logger.warning('Entering the `temporary modal generator from image` context')

        try:
            self._logger.info('Creating temporary modal generator from image if needed')
            if self._mgen is None:
                self._mgen = self._create_fit_mask_from_img(image)
                was_temporary = True
            image = _np.ma.masked_array(
                image.copy().data, mask=self._mgen._boolean_mask.copy()
            )
            yield image, was_temporary
        finally:
            self._logger.info('Restoring previous modal generator if it was temporary')
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
        self._logger.info('Creating fitting mask from image')
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
        self._logger = _SL(__class__)
        super().__init__(fit_mask)

    def removeZernike(
        self,
        image: _t.ImageData,
        zernike_index_vector: list[int] = None,
        mode: str = "global",
    ) -> _t.ImageData:
        """
        Remove Zernike modes from the image using the current fit mask.

        Parameters
        ----------
        image : ImageData
            Image from which to remove Zernike modes.
        zernike_index_vector : list[int], optional
            List of Zernike mode indices to be removed. Default is [1, 2, 3].
        mode : str
            If more than one ROI is found in the fitting mask, this parameter
            controls how the modes are computed:
            - `global` will compute the mean of the fitted coefficient of each ROI
            - `local` will compute the fitted coefficient for each ROI

            Defaults to 'global'.

        Returns
        -------
        new_ima : ImageData
            Image with Zernike modes removed.
        """
        if zernike_index_vector is None:
            zernike_index_vector = [1, 2, 3]
        return self.filterModes(
            image=image, mode_index_vector=zernike_index_vector, mode=mode
        )

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
        self._logger.info(f'Getting mode {mode_index} from generator')
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
        self._logger = _SL(__class__)
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
        self._logger.info(f'Getting mode {mode_index} from generator')
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
        self._logger = _SL(__class__)
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
        self._logger.info(f'Getting mode {mode_index} from generator')
        return self._mgen.getRBF(mode_index)
