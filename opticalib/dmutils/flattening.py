"""
Module containing the class which computes the flattening command for a deformable
mirror, given an imput shape and a (filtered) interaction cube.

Author(s)
---------
- Pietro Ferraiuolo : written in 2024

Description
-----------
From the loaded tracking number (tn) the interaction cube will be loaded (and
filtered, if it's not already) from which the interaction matrix will be computed.
If an image to shape is provided on class instance, then the reconstructor will
be automatically computed, while if not, the load_img2shape methos is available
to upload a shape from which compute the reconstructor.

How to Use it
=============
Instancing the class only with the tn of the interaction cube

```python
from opticalib.dmutils import flattening as flt
tn = '20240906_110000' # example tn
f = flt.Flattening(tn)
# say we have acquired an image
img = interf.acquire_map()
f.load_image2shape(img)
f.computeRecMat()
'Computing reconstruction matrix...'
```

all is ready to compute the flat command, by simply running the method

```python
flatCmd = f.computeFlatCmd()
```

Update : all the steps above have been wrapped into the `applyFlatCommand` method,
which will also save the flat command and the images used for the computation in a
dedicated folder in the flat root folder.

"""

import os as _os
import numpy as _np
from opticalib import typings as _ot
from opticalib.ground import osutils as _osu
from opticalib.core.root import folders as _fn
from opticalib.ground import computerec as _crec
from . import iff_processing as _ifp, utils as _ut
from ..ground.logger import SystemLogger as _SL

_ts = _osu.newtn

class Flattening:
    """
    Class for computing and applying flattening commands to deformable mirrors.

    Overview
    --------
    This class manages the process of flattening a deformable mirror using an interaction cube
    and a reference shape (typically acquired from an interferometer). It supports loading and filtering
    interaction cubes, aligning and processing images, computing reconstruction matrices, and generating
    the appropriate command to flatten the mirror surface.

    Key Features
    ------------
    - Loads and filters interaction cubes based on Zernike modes.
    - Aligns input images to the interaction cube mask for accurate command computation.
    - Computes the reconstruction matrix using SVD, with options to discard modes or set thresholds.
    - Calculates the flattening command for a given shape and applies it to the deformable mirror.
    - Saves all relevant data (commands, images, metadata) for traceability and reproducibility.

    Public Methods
    --------------
    - applyFlatCommand(dm, interf, modes2flat, nframes=5, modes2discard=None):
        Acquires images, computes and applies the flattening command, and saves results.
    - computeFlatCmd(n_modes):
        Computes the flattening command for the loaded shape and selected modes.
    - loadImage2Shape(img, compute=None):
        Loads a new image to flatten and optionally computes the reconstruction matrix.
    - computeRecMat(threshold=None):
        Computes the reconstruction matrix for the loaded image.
    - filterIntCube(zernModes=None):
        Filters the interaction cube by removing specified Zernike modes.
    - loadNewTn(tn):
        Loads a new tracking number and updates internal data.

    Usage Example
    -------------
        >>> f = Flattening('20240906_110000')
        >>> img = interf.acquire_map()
        >>> f.loadImage2Shape(img)
        >>> f.computeRecMat()
        >>> flatCmd = f.computeFlatCmd(10)
        >>> f.applyFlatCommand(dm, interf, modes2flat=10)
    """

    def __init__(self, tn: str):
        """The Constructor"""
        self.tn = tn
        self.shape2flat = None
        self.flatCmd = None
        self.rebin = None
        self.filtered = False
        self.filteredModes = None
        self._path = _os.path.join(_ifp._intMatFold, self.tn)
        self._oldtn = tn
        self._intCube = self._loadIntCube()
        self._cmdMat = self._loadCmdMat()
        self._rec = self._loadReconstructor()
        self._recMat = None
        self._frameCenter = None
        self._flatOffset = None
        self._cavityOffset = None
        # self._synthFlat = None
        # self._flatResidue = None
        # self._flatteningModes = None
        self._logger = _SL(__class__)
        

    @property
    def RM(self) -> _ot.MatrixLike:
        """
        Reconstruction matrix property.
        """
        return self._recMat

    @property
    def CM(self) -> _ot.MatrixLike:
        """
        Command matrix property.
        """
        return self._cmdMat

    @property
    def IM(self) -> _ot.MatrixLike:
        """
        Interaction cube property.
        """
        return self._rec._intMat

    @property
    def analysisMask(self) -> _ot.MaskData:
        """
        Analysis mask property.
        """
        return self._rec._analysisMask

    def closedLoopFlattening(self, iterations: int | None = None, **kwargs: dict[str,_ot.Any]) -> None:
        """
        Computes, applies and saves the computed flat command to the DM in 
        closed loop, until an input to stop is provided.

        The parameters are the same of 

        Parameters
        ----------
        iterations : int, optional
            It is the number of flattening iterations to perform. If not provided,
            the loop will stop at the user's input.
        kwargs: dict
            The arguments for the `applyFlatCommand` function:
            - dm : DeformableMirrorDevice
                Deformable mirror object.
            - interf : InterferometerDevice
                Interferometer object to acquire phasemaps.
            - modes2flat : int | ArrayLike
                Modes to flatten.
            - modes2discard : int, optional
                Number of modes to discard when computing the reconstruction matrix. Default is 3.
            - nframes : int, optional
                Number of frames to average for phasemap acquisition. Default is 5.
        """
        if iterations is not None:
            self._logger.info(f"Starting closed-loop flattening for {iterations} iterations.")
            for _ in range(iterations):
                self.applyFlatCommand(**kwargs)
        else:
            raise NotImplementedError('e che ce vo')


    def applyFlatCommand(
        self,
        dm: _ot.DeformableMirrorDevice,
        interf: _ot.InterferometerDevice,
        modes2flat: int | _ot.ArrayLike,
        modes2discard: _ot.Optional[int] = None,
        nframes: int = 5,
        **setshape_kwargs: dict[str, _ot.Any],
    ) -> None:
        f"""
        Computes, applies and saves the computed flat command to the DM, given
        the {self.tn} calibration.

        Parameters
        ----------
        dm : DeformableMirrorDevice
            Deformable mirror object.
        interf : InterferometerDevice
            Interferometer object to acquire phasemaps.
        modes2flat : int | ArrayLike
            Modes to flatten.
        modes2discard : int, optional
            Number of modes to discard when computing the reconstruction matrix. Default is 3.
        nframes : int, optional
            Number of frames to average for phasemap acquisition. Default is 5.
        """
        new_tn = _ts()
        imgstart = interf.acquire_map(nframes, rebin=self.rebin)
        self.loadImage2Shape(imgstart)
        self.computeRecMat(modes2discard)
        deltacmd = self.computeFlatCmd(modes2flat)
        cmd = dm.get_shape()  # TODO: check if this is correct for DP

        # handle diverse DM set_shape args
        _ = setshape_kwargs.pop("differential", None)
        self._logger.info(f'Applying flat command to the {dm._name}')
        dm.set_shape(deltacmd, differential=True, **setshape_kwargs)

        imgflat = interf.acquire_map(nframes, rebin=self.rebin)
        files = [
            "flatCommand.fits",
            "flatDeltaCommand.fits",
            "imgstart.fits",
            "imgflat.fits",
        ]
        data = [cmd, deltacmd, imgstart, imgflat]
        fold = _os.path.join(_fn.FLAT_ROOT_FOLDER, new_tn)
        header = {}
        header["CALDATA"] = (self.tn, "calibration data used")
        header["MODFLAT"] = (str(modes2flat), "modes used for flattening")
        header["MDISCAR"] = (modes2discard, "modes discarded in reconstructor")
        header["DMNAME"] = (dm._name, "deformable mirror name")
        header["INTERF"] = (interf._name, "interferometer used")
        if not _os.path.exists(fold):
            _os.mkdir(fold)
        for f, d in zip(files, data):
            path = _os.path.join(fold, f)
            _osu.save_fits(path, d, header=header)
        print(f"Flat command saved in .../{'/'.join(fold.split('/')[-2:])}")
        self._logger.info(f"Flat command and images saved in {fold}.")

    def computeFlatCmd(self, n_modes: int | _ot.ArrayLike) -> _ot.ArrayLike:
        """
        Compute the command to apply to flatten the input shape.

        Parameters
        ----------
        n_modes : int | ArrayLike
            Number of modes used to compute the flat command. If int, it will
            compute the first n_modes of the command matrix. If list, it will
            compute the flat command for the given modes.

        Returns
        -------
        flat_cmd : ndarray
            Flat command.
        """
        self._logger.info("Computing flat command...")
        img = _np.ma.masked_array(self.shape2flat, mask=self._getMasterMask())
        _cmd = -_np.dot(img.compressed(), self._recMat)
        cmdMat = self._cmdMat.copy()
        if isinstance(n_modes, int):
            flat_cmd = cmdMat[:, :n_modes] @ _cmd[:n_modes]
        elif isinstance(n_modes, (_np.ndarray, list)):
            _cmdMat = _np.zeros((cmdMat.shape[0], len(n_modes)))
            _scmd = _np.zeros(len(n_modes))
            for i, mode in enumerate(n_modes):
                _cmdMat.T[i] = cmdMat.T[mode]
                _scmd[i] = _cmd[mode]
            flat_cmd = _cmdMat @ _scmd
        else:
            self._logger.error(f"`n_modes` must be either an int or a list of int: {type(n_modes)}")
            raise TypeError(
                f"`n_modes` must be either an int or a list of int: {type(n_modes)}"
            )
        self.flatCmd = flat_cmd.copy()
        return flat_cmd

    def loadImage2Shape(
        self, img: _ot.ImageData, compute: _ot.Optional[int | float] = None
    ) -> None:
        """
        (Re)Loader for the image to flatten.

        Parameters
        ----------
        img : ImageData
            Image to flatten.
        compute : int | float, optional
            If not None, it can be either the number of modes to discard from the
            reconstruction matrix computation (int) or the threshold value to discard
            computed eigenvalues for the reconstruction (float). Default is None.
        """
        self.shape2flat = self._alignImgAndCubeMasks(img)
        self._rec = self._rec.loadShape2Flat(self.shape2flat)
        self._logger.info("Image to shape loaded to Reconstructor class.")
        if compute is not None:
            self.computeRecMat(compute)

    def computeRecMat(self, threshold: _ot.Optional[int | float] = None):
        """
        Compute the reconstruction matrix for the loaded image.

        Parameters
        ----------
        threshold : int | float, optional
            If not None, it can be either the number of modes to discard from the
            reconstruction matrix computation (int) or the threshold value to discard
            computed eigenvalues for the reconstruction (float). Default is None.
        """
        print("Computing recontruction matrix...")
        self._logger.info("Starting reconstruction matrix computation...")
        self._recMat = self._rec.run(sv_threshold=threshold)

    def getSVDmatrices(self) -> tuple[_ot.ArrayLike, _ot.ArrayLike, _ot.ArrayLike]:
        """
        Returns the U, S, Vt matrices from the SVD decomposition of the interaction matrix.

        Returns
        -------
        U : ndarray
            Left singular vectors.
        S : ndarray
            Singular values.
        Vt : ndarray
            Right singular vectors (transposed).
        """
        return self._rec._intMat_U, self._rec._intMat_S, self._rec._intMat_Vt

    def plotEigenvalues(self) -> None:
        """
        Plots the eigenvalues of the interaction matrix.
        """
        import matplotlib.pyplot as plt

        if self._rec._intMat_S is None:
            raise ValueError("Reconstruction matrix not computed yet.")
        plt.figure()
        plt.semilogy(self._rec._intMat_S, "o-")
        plt.title("Eigenvalues of the interaction matrix")
        plt.xlabel("Mode number")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.show()

    def filterIntCube(
        self, zernModes: _ot.Optional[list[int] | _ot.ArrayLike] = None
    ) -> "Flattening":
        """
        Filter the interaction cube with the given zernike modes

        Parameters
        ----------
        zernModes : list of int | ArrayLike, optional
            Zernike modes to filter out this cube (if it's not already filtered).
            Default modes are [1,2,3] -> piston/tip/tilt.
        """
        try:
            import warnings

            warnings.warn(
                "filtering flag in `flag.txt` file is deprecated and will be removed in a future version of `opticalib`.",
                DeprecationWarning,
            )
            self._logger.warning(
                "filtering flag in `flag.txt` file is deprecated and will be removed in a future version of `opticalib`."
            )
            # Backwards compatibility for rebinning
            with open(
                _os.path.join(self._path, _ifp.flagFile), "r", encoding="utf-8"
            ) as f:
                flag = f.read()
            if " filtered " in flag:
                print("Cube already filtered, skipping...")
                return
        except FileNotFoundError:
            if self.filtered:
                self._logger.warning("Cube already filtered, skipping...")
                print("Cube already filtered, skipping...")
                return
            else:
                print("Filtering cube...")
                self._oldCube = self._intCube.copy()
                zern2fit = zernModes if zernModes is not None else [1, 2, 3]
                self._logger.info(f"Filtering cube of zernike modes {zern2fit}...")
                self._intCube, new_tn = _ifp.filterZernikeCube(self.tn, zern2fit)
                self.loadNewTn(new_tn)
                self.filtered = True
                self.filteredModes = zern2fit
        return self

    def loadNewTn(self, tn: str) -> None:
        """
        Load a new tracking number for the flattening.

        Parameters
        ----------
        tn : str
            Tracking number of the new data.
        """
        self.__update_tn(tn)
        self._reloadClass(tn)

    def _reloadClass(self, tn: str) -> None:
        """
        Reload function for the interaction cube

        Parameters
        ----------
        tn : str
            Tracking number of the new data.
        zernModes : list, optional
            Zernike modes to filter out this cube (if it's not already filtered).
            Default modes are [1,2,3] -> piston/tip/tilt.
        """
        self._cmdMat = self._loadCmdMat()
        self._rec = self._rec.loadInteractionCube(tn=tn)

    def _getMasterMask(self) -> _ot.ImageData:
        """
        Creates the intersection mask of the interaction cube.
        """
        self._logger.info("Creating master mask from interaction cube...")
        cubeMask = _np.sum(self._intCube.mask.astype(int), axis=2)
        master_mask = _np.zeros(cubeMask.shape, dtype=_np.bool_)
        master_mask[_np.where(cubeMask > 0)] = True
        return master_mask

    def _alignImgAndCubeMasks(self, img: _ot.ImageData) -> _ot.ImageData:
        """
        Aligns the image mask with the interaction cube mask.

        Parameters
        ----------
        img : ImageData
            Image to align with the interaction cube mask.

        Returns
        -------
        aligned_img : ImageData
            Aligned image.
        """
        self._logger.info("Aligning image and cube masks...")
        cubemask = self._getMasterMask()
        pad_shape = (
            (cubemask.shape[0] - img.shape[0]) // 2,
            (cubemask.shape[1] - img.shape[1]) // 2,
        )
        if not any([x <= 0 for x in pad_shape]):
            img = _np.ma.masked_array(
                _np.pad(img.data, pad_shape), mask=~_np.pad(~img.mask, pad_shape)
            )
        if img.shape != cubemask.shape:
            self._logger.info("Padding image to match cube mask shape...")
            xdiff = cubemask.shape[1] - img.shape[1]
            ydiff = cubemask.shape[0] - img.shape[0]
            nimg = _np.pad(img.data, ((ydiff, 0), (0, xdiff)))
            nmask = _np.pad(~img.mask, ((ydiff, 0), (0, xdiff)))
            img = _np.ma.masked_array(nimg, mask=~nmask)
        xci, yci = self.__get_mask_center(img.mask)
        xcm, ycm = self.__get_mask_center(cubemask)
        roll = (xcm - xci, ycm - yci)
        img = _np.roll(img, roll, axis=(0, 1))
        if self.filteredModes is not None:
            self._logger.info("Removing Zernike modes from the image to match the loaded calibration...")
            from opticalib.ground.modal_decomposer import ZernikeFitter

            zfit = ZernikeFitter(cubemask)
            self._logger.info(f"Filtered modes: {self.filteredModes}")
            img = zfit.removeZernike(img, self.filteredModes)
        return img

    def _loadIntCube(self) -> _ot.CubeData:
        """
        Interaction cube loader

        Return
        ------
        intCube : CubeData
            The interaction cube data array.
        """
        intCube = _osu.load_fits(_os.path.join(self._path, _ifp.cubeFile))
        try:
            import warnings

            warnings.warn(
                "filtering flag in `flag.txt` file is deprecated and will be removed in a future version of `opticalib`.",
                DeprecationWarning,
            )
            # Backwards compatibility for rebinning
            with open(_os.path.join(self._path, _ifp.flagFile), "r") as file:
                lines = file.readlines()
                flag = file.read()
            rebin = eval(lines[1].split("=")[-1])
            if " filtered " in flag:
                filtered = True
                fittedModes = eval(lines[2].split("=")[-1])
            else:
                filtered = False
                fittedModes = None
        except FileNotFoundError:
            rebin = intCube.header.get("REBIN", 1)
            filtered = intCube.header.get("FILTERED", False)
            fittedModes = eval(intCube.header.get("ZREMOVED", "None"))
        self.rebin = rebin
        self.filtered = filtered
        self.filteredModes = fittedModes
        return intCube

    def _loadCmdMat(self) -> _ot.MatrixLike:
        """
        Command matrix loader. It loads the saved command matrix of the loaded
        cube.

        Returns
        -------
        cmdMat : MatrixLike
            Command matrix of the cube, saved in the tn path.
        """
        cmdMat = _osu.load_fits(_os.path.join(self._path, _ifp.cmdMatFile))
        return cmdMat

    def _loadReconstructor(self) -> _ot.Reconstructor:
        """
        Builds the reconstructor object off the input cube

        Returns
        -------
        rec : Reconstructor
            Reconstructor class.
        """
        rec = _crec.ComputeReconstructor(self._intCube)
        return rec

    def _loadFrameCenter(self):
        """
        Center frame loader, useful for image registration.

        Returns
        -------
        frame_center : TYPE
            DESCRIPTION.

        """
        frame_center = _osu.load_fits("data")
        return frame_center

    def _registerShape(self, shape: tuple[int, int]) -> _ot.ImageData:
        xxx = None
        dp = _ifp.findFrameOffset(self.tn, xxx)
        # cannot work. we should create a dedicated function, not necessarily linked to IFF or flattening
        return dp

    def __get_mask_center(self, mask: _ot.MaskData) -> tuple[int, int]:
        """
        Computes the center of the mask, which is used to align images and cubes.

        Parameters
        ----------
        mask : MaskData
            Mask of the image or cube.

        Returns
        -------
        y_center, x_center : tuple
            Coordinates of the center of the mask.
        """
        ys, xs = _np.where(~mask)
        y_center = (ys.min() + ys.max()) // 2
        x_center = (xs.min() + xs.max()) // 2
        return y_center, x_center

    def __update_tn(self, tn: str) -> None:
        """
        Updates the tn and cube path if the tn is to change

        Parameters
        ----------
        tn : str
            New tracking number.
        """
        self.tn = tn
        self._path = _os.path.join(_ifp._intMatFold, self.tn)

    def __repr__(self) -> str:
        """
        String representation of the class instance.
        """
        tn = self.tn
        filtered = self.filteredModes if self.filtered else False
        r = self.rebin
        return f"Flattening(tn={tn}, filtered={filtered}, rebin={r})"
