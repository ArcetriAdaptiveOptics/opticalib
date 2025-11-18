"""
ALIGNMENT module
================
2024

Author(s):
----------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it

Description
-----------
This module provides the `Alignment` class and related functions for performing
alignment procedures, including calibration and correction.

How to Use it
-------------
This module contains the class `Alignment`, which manages, alone, both the calibration
and the correction of the alignment of the system. The class is initialized with the
mechanical and acquisition devices used for alignment. These devices, which, for example, in
the case of the M4 project are the OTT and the interferometer, are passed as arguments
and configured through the `configuration.yaml` file, under the `ALIGNMENT` section.

Usage Example
-------------
Given the OTT (with Parabola, Reference Mirror and M4 Hexapode) as mechanical device
and the interferometer as acquisition device, we can initialize the class as follows:

```python
    from opticalib.alignment import Alignment
    align = Alignment(ott, interf)
    # At this point the alignment is ready to be calibrated, given the command amplitude
    amps = [0,7, 10, 10, 6, 6, 4, 4] # example, verosimilar, amplitudes
    align.calibrate_alignment(amps)
    [...]
    "Ready for Alignment..."
```

At this point, the calibration is complete and and `InteractionMatrix.fits` file
was created, saved and stored in the Alignment class. It is ready to compute
and apply corrections.

```python
    modes2correct = [3,4] # Reference Mirror DoF
    zern2correct = [0,1] # tip $ tilt
    align.correct_alignment(modes2correct, zern2correct, apply=True)
```

If we already have an `InteractionMatrix.fits` file, we can load it and apply
corrections based off the loaded calibration. All to do is to load the calibration
to the class:

```python
    tn_cal = '20241122_160000' # example, tracking number
    align.load_calibration(tn_cal) # load the calibration
    align.correct_alignment(modes2correct, zern2correct, apply=True)
```

It can also be instanced with a calibration:

```python
    tn_cal = '20241122_160000' # example, tracking number
    align = Alignment(ott, interf, calibtn=tn_cal)
    align.correct_alignment(modes2correct, zern2correct, apply=True)
```

Notes
-----
Note that the calibration process can be done uploading to the class
a `calibrated cavity`, so that a different algorithm for the Zernike fitting is
performed. This can be done through the `load_fitting_surface` method.

```python
    cavity_tn = '20241122_160000' # example, tracking number
    align.load_fitting_surface(cavity_tn) # load the calibrated cavity
```

When working with segmented system (e.g. a segmented mirror), the Zernike modes
shall be computed as global coefficients, which are basically the average of the
local amplitude measured on each of the segment.

"""

import os as _os
import numpy as _np
from .core.root import folders as _fn
from .core.read_config import getAlignmentConfig as _gac
from .ground import logger as _logger, roi as roigen
from .ground.modal_decomposer import ZernikeFitter as _zfitter
from .ground.osutils import load_fits as _rfits, save_fits as _sfits, newtn as _ts
from . import typings as _ot
from .analyzer import pushPullReductionAlgorithm as _ppr

_sc = _gac()
_np.set_printoptions(precision=2, suppress=False)


class Alignment:
    """
    Class for the alignment procedure: calibration and correction.

    This class provides methods to perform alignment procedures using mechanical
    and acquisition devices. It handles the initialization of devices, reading
    calibration data, and executing alignment commands.

    Attributes
    ----------
    mdev : object or list
        The mechanical devices used for alignment. Can be either a single object
        which calls more devices or a list of single devices.
    ccd : object
        The acquisition devices used for alignment.
    cmdMat : numpy.ndarray
        The command matrix read from a FITS file, used for alignment commands.
    intMat : numpy.ndarray or None
        The interaction matrix, initialized as None.
    recMat : numpy.ndarray or None
        The reconstruction matrix, initialized as None.

    Methods
    -------
    correct_alignment(modes2correct, zern2correct, tn=None, apply=False, n_frames=15)
        Corrects the alignment of the system based on Zernike coefficients.
    calibrate_alignment(cmdAmp, n_frames=15, template=None, n_repetitions=1, save=True)
        Calibrates the alignment of the system using the provided command amplitude and template.
    read_positions(show=True)
        Reads the current positions of the devices.
    reload_calibrated_parabola(tn)
        Reloads the calibrated parabola from the given tracking number.

    """

    def __init__(
        self,
        mechanical_devices: _ot.GenericDevice | list[_ot.GenericDevice],
        acquisition_devices: _ot.InterferometerDevice | list[_ot.InterferometerDevice],
        calibtn: _ot.Optional[str] = None,
    ):
        """
        Initializes the Alignment class with mechanical and acquisition devices.

        Parameters
        ----------
        mechanical_devices : object or list of objects
            The mechanical devices used for alignment. Can be either
            a single object which calls more devices or a list of
            single devices.
        acquisition_devices : object
            The acquisition devices used for alignment.
        calibtn : str, optional
            The tracking number of the alignment calibration to be used.
        """
        self.mdev = mechanical_devices
        self.ccd = acquisition_devices
        self.cmdMat = _rfits(
            _os.path.join(_fn.CONTROL_MATRIX_FOLDER, _sc.commandMatrix)
        )
        self._calibtn = calibtn
        self.intMat = self.__loadIntMat(calibtn)
        self.recMat = None
        self._cmdAmp = None
        self._surface = (
            _rfits(_sc.fitting_surface) if not _sc.fitting_surface == "" else None
        )
        self._zfit = _zfitter(self._surface)
        self._moveFnc = self.__get_callables(self.mdev, _sc.devices_move_calls)
        self._readFnc = self.__get_callables(self.mdev, _sc.devices_read_calls)
        self._acquire = self.__get_callables(self.ccd, _sc.ccd_acquisition)
        self._devName = self.__get_dev_names(_sc.names, ndev=len(self._moveFnc))
        self._dof = [
            _np.array(dof) if not isinstance(dof, _np.ndarray) else dof
            for dof in _sc.dof
        ]
        self._dofTot = (
            _sc.devices_dof
            if isinstance(_sc.devices_dof, list)
            else [_sc.devices_dof] * len(self._moveFnc)
        )
        self._idx = _sc.slices
        self._zvec2fit = _np.arange(1, 11)
        self._zvec2use = _sc.zernike_to_use
        self._template = _sc.push_pull_template
        self._correct_cavity = True
        self._dataPath = _fn.ALIGNMENT_ROOT_FOLDER
        self._logPath = _os.path.join(_fn.LOGGING_ROOT_FOLDER, "alignment.log")
        self._txt = _logger.txtLogger(self._logPath.strip(".log") + "Record.txt")
        _logger.set_up_logger(self._logPath, 20)

    def correct_alignment(
        self,
        modes2correct: _ot.ArrayLike,
        zern2correct: _ot.ArrayLike,
        apply: bool = False,
        n_frames: int = 15,
    ) -> str | _ot.ArrayLike:
        """
        Corrects the alignment of the system based on Zernike coefficients.

        Parameters
        ----------
        modes2correct : array-like
            Indices of the modes to correct.
        zern2correct : array-like
            Indices of the Zernike coefficients to correct.
        tn : str, optional
            Tracking number of the intMat.fits to be used
        apply : bool, optional
            If True, the correction command will be applied to the system.
            If False (default), the correction command will be returned.
        n_frames : int, optional
            Number of frames acquired and averaged the alignment correction. Default is 15.

        Returns
        -------
        numpy.ndarray or str
            If `apply` is False, returns the correction command as a numpy array.
            If `apply` is True, applies the correction command and returns a string
            indicating that the alignment has been corrected along with the current
            positions.

        Notes
        -----
        This method acquires an image, calculates the Zernike coefficients, reads the
        interaction matrix from a FITS file, reduces the interaction matrix and command
        matrix based on the specified modes and Zernike coefficients, creates a
        reconstruction matrix, calculates the reduced command, and either applies the
        correction command or returns it.
        """
        _logger.log(f"{self.correct_alignment.__qualname__}")
        self._correct_cavity = True
        image = self._acquire[0](nframes=n_frames)
        zernike_coeff = self._zern_routine(image)
        if self.intMat is not None:
            intMat = self.intMat
        else:
            raise AttributeError(
                "No internal matrix found. Please calibrate the alignment first."
            )
        reduced_intMat = intMat[_np.ix_(zern2correct, modes2correct)]
        reduced_cmdMat = self.cmdMat[:, modes2correct]
        recMat = self._create_rec_mat(reduced_intMat)
        reduced_cmd = _np.dot(recMat, zernike_coeff[zern2correct])
        f_cmd = -_np.dot(reduced_cmdMat, reduced_cmd)
        print(f"Resulting Command: {f_cmd}")
        if apply:
            print("Applying correction command...")
            self._apply_command(f_cmd)
            print("Alignment Corrected\n")
            self.read_positions()
            return
        return f_cmd

    def calibrate_alignment(
        self,
        cmdAmp: int | float | _ot.ArrayLike,
        n_frames: int = 15,
        template: _ot.ArrayLike = None,
        n_repetitions: int = 1,
        save: bool = True,
    ) -> str:
        """
        Calibrate the alignment of the system using the provided command amplitude and template.

        Parameters
        ----------
        cmdAmp : int|float|arrayLike
            The command amplitude used for calibration.
        n_frames : int, optional
            The number of frames acquired and averaged for calibration. Default is 15.
        template : list, optional
            A list representing the template for calibration. If not provided, the default template will be used.
        n_repetitions : int, optional
            The number of repetitions for the calibration process. Default is 1.
        save : bool, optional
            If True, the resulting internal matrix will be saved to a FITS file. Default is False.

        Returns
        -------
        str
            A message indicating that the system is ready for alignment.

        Notes
        -----
        This method performs the following steps:
        1. Sets the command amplitude.
        2. Uses the provided template or the default template if none is provided.
        3. Produces a list of images based on the template and number of repetitions.
        4. Executes a Zernike routine on the image list to generate an internal matrix.
        5. Optionally saves the internal matrix to a FITS file.
        """
        self._correct_cavity = False
        _logger.log(f"Cavity correction: False", level="INFO")
        self._calibtn = _ts()
        _logger.log(f"{self.calibrate_alignment.__qualname__}")
        self._cmdAmp = cmdAmp
        template = template if template is not None else self._template
        imglist = self._images_production(template, n_frames, n_repetitions)
        intMat = self._zern_routine(imglist)
        self.intMat = intMat.copy()
        if save:
            tn = self._calibtn
            path = _os.path.join(_fn.ALIGN_CALIBRATION_ROOT_FOLDER, tn)
            if not _os.path.exists(path):
                _os.mkdir(path)
            filename = _os.path.join(path, "InteractionMatrix.fits")
            _sfits(filename, self.intMat, overwrite=True)
            _logger.log(f"{_sfits.__qualname__}")
            _logger.log(f"Calibration saved in '{filename}'", level="INFO")
            print(f"Calibration saved in '{filename}'\nReady for Alignment...")
        return tn

    def read_positions(self, show: bool = True) -> _ot.ArrayLike:
        """
        Reads the current positions of the devices.

        Returns
        -------
        pos : ArrayLike
            The list of current positions of the devices.
        """
        _logger.log(f"{self.read_positions.__qualname__}")
        logMsg = ""
        pos = []
        logMsg += "Current Positions\n"
        for fnc, dev_name in zip(self._readFnc, self._devName):
            temp = fnc()
            pos.append(_Command(temp))
            logMsg += f"{dev_name}" + " " * (16 - len(dev_name)) + f" : {temp}\n"
        logMsg += "-" * 30
        if show:
            print(logMsg)
        return pos

    def load_fitting_surface(self, filepath: str) -> None:
        """
        This function let you load the mask to use for zernike fitting. In the case of
        M$, for example, here the calibrated parabola is loaded, so that zernike modes are
        fitted using the parabola surface (right) instead of the Reference Mirror one
        (smaller, wrong)

        Parameters
        ----------
        filepath : str
            The file path to the parabola file.

        Returns
        -------
        str
            A message indicating the successful loading of the file.
        """
        surf = _rfits(filepath)
        self._surface = surf
        print(f"Correctly loaded '{filepath}'")

    def load_calibration(self, tn: str) -> None:
        """
        Loads the alignment calibration InteractionMatrix.fits based on the
        provided tracking number.

        Parameters
        ----------
        tn : str
            The tracking number of the calibration to be loaded.
        """
        self._calibtn = tn
        self.intMat = self.__loadIntMat(tn)
        print(f"Calibration loaded from '{tn}'")

    def _images_production(
        self, template: _ot.ArrayLike | list[int], n_frames: int, n_repetitions: int
    ) -> _ot.CubeData:
        """
        Acquire images based on the provided template and number of repetitions.

        Parameters
        ----------
        template : ArrayLike
            The template used for image production.
        n_frames : int
            The number of frames acquired and averaged for image production.
        n_repetitions : int
            The number of repetitions for image production.

        Returns
        -------
        n_results : CubeData
            The list of produced images.
        """
        results = []
        n_results = []
        for i in range(n_repetitions):
            logMsg = ""
            logMsg += f"Repetition n.{i}\n"
            print(logMsg)
            for k in range(self.cmdMat.shape[1]):
                logMsg2 = ""
                logMsg2 += f"Matrix Column {k+1} : {self.cmdMat.T[k]}"
                print(f"Matrix Column {k+1} : {self.cmdMat.T[k]}\n")
                imglist = self._img_acquisition(k, template, n_frames)
                # image = self._push_pull_redux(imglist, template) / self._cmdAmp[k]
                template.insert(0, 1)
                image = _ppr(
                    imglist, template, normalization=6 * self._cmdAmp[k]
                )  # TODO: 6 -> sum of template weights?
                template.pop(0)
                results.append(image)
            if n_repetitions != 1:
                n_results.append(results)
            else:
                n_results = results
        return n_results

    #### OLD ALGORITHM - TO BE DELETED LATER
    # def _zern_routine(
    #     self, imglist: list[_ot.ImageData] | _ot.CubeData
    # ) -> _ot.MatrixLike:
    #     """
    #     Creates the interaction matrix from the provided image list.

    #     Parameters
    #     ----------
    #     imglist : CubeData
    #         The list of images used to create the interaction matrix.

    #     Returns
    #     -------
    #     intMat : MatrixLike
    #         The interaction matrix created from the images.
    #     """
    #     _logger.log(f"{self._zern_routine.__qualname__}")
    #     coefflist = []
    #     if not isinstance(imglist, list):
    #         imglist = [imglist]
    #     for img in imglist:
    #         if self._surface is None:
    #             coeff, _ = _zern.zernikeFit(img, self._zvec2fit)
    #             _logger.log(f"{_zern.zernikeFit.__qualname__}")
    #         else:
    #             if self._correct_cavity is True:
    #                 img -= 2 * self._surface
    #             cir = _geo.qpupil(-1 * self._surface.mask + 1)
    #             mm = _geo.draw_mask(
    #                 self._surface.data * 0, cir[0], cir[1], 1.44 / 0.00076 / 2, out=0
    #             )  # e questo blocco potrebbe essere in una funzione chiamata all'avvio,
    #             # così si crea anche la auxmask. i parametri da definire in conf sarebbero 1.44 / 0.00076 / 2 == pix on radius
    #             # coeff, _ = _zern.zernikeFitAuxmask(img, mm, self._zvec2fit) #mod RB20250917: this part has been substituted with zern_on_roi below
    #             coeff = self._global_zern_on_roi(img, auxmask=mm)
    #             _logger.log(f"{_zern.zernikeFitAuxmask.__qualname__}")
    #         coefflist.append(coeff[self._zvec2use])
    #     if len(coefflist) == 1:
    #         coefflist = _np.array([c for c in coefflist[0]])
    #     intMat = _np.array(coefflist).T
    #     return intMat

    def _zern_routine(
        self, imglist: list[_ot.ImageData] | _ot.CubeData
    ) -> _ot.MatrixLike:
        """
        Creates the interaction matrix from the provided image list.

        Parameters
        ----------
        imglist : CubeData
            The list of images used to create the interaction matrix.

        Returns
        -------
        intMat : MatrixLike
            The interaction matrix created from the images.
        """
        _logger.log(f"{self._zern_routine.__qualname__}")
        coefflist = []
        if not isinstance(imglist, list):
            imglist = [imglist]
        for img in imglist:
            if self._surface is None:
                coeff, _ = self._zfit.fit(img, self._zvec2fit)
                _logger.log(f"{self._zfit.fit.__qualname__}")
            else:
                if self._correct_cavity is True:
                    img -= 2 * self._surface
                coeff = self._zfit.fitOnROi(img, self._zvec2fit, "global")
                _logger.log(f"{self._zfit.fitOnROi.__qualname__}")
            coefflist.append(coeff[self._zvec2use])
        if len(coefflist) == 1:
            coefflist = _np.array([c for c in coefflist[0]])
        intMat = _np.array(coefflist).T
        return intMat

    #### Deprecated - to be deleted later
    # def _global_zern_on_roi(
    #     self, img: _ot.ImageData, auxmask: _ot.Optional[_ot.ImageData] = None
    # ):
    #     """
    #     Computes Zernike coefficients over a segmented fitting area, i.e. a pupil
    #     mask divided into Regions Of Interest (ROI). The computation is based on
    #     the fitting of Zernike modes independently on each ROI; the coefficients
    #     are then averaged together to return the global Zernike mode amplitude.
    #     An auxiliary mask (optional) may be passed. Such auxiliary mask allows
    #     creating the Zernike modes (or more precisely the coordinates grid) over
    #     a user-defined area, instead over the image mask (default option for zernikeFit).

    #     Parameters
    #     ----------
    #     img : ImageData
    #         Image to fit the Zernike modes on, over the ROIs.
    #     auxmask : ImageData, optional
    #         Image of the auxiliary mask, where the fitting coordinates are constructed

    #     Returns
    #     -------
    #     zcoeff : array
    #         The vector of the Zernike coefficients, corresponding to the selected modes id,
    #         fitted over the auxiliary mask and all the ROIs, averaged together.
    #     """
    #     print("Searching for Regions of Interest in the frame...")
    #     roiimg = roigen.roiGenerator(img)
    #     nroi = len(roiimg)
    #     print("Found " + str(nroi) + " ROI")
    #     if auxmask is None:
    #         auxmask2use = img.mask
    #     else:
    #         auxmask2use = auxmask
    #     zcoeff = _np.zeros([nroi, len(self._zvec2fit)])
    #     for i in range(nroi):
    #         img2fit = _np.ma.masked_array(img.data, roiimg[i])
    #         cc, _ = _zern.zernikeFitAuxmask(img2fit, auxmask2use, self._zvec2fit)
    #         zcoeff[i, :] = cc
    #     zcoeff = zcoeff.mean(axis=0)
    #     print("Global Zernike coeff:")
    #     print(str(zcoeff))
    #     return zcoeff

    def _create_rec_mat(self, intMat: _ot.MatrixLike) -> _ot.MatrixLike:
        """
        Creates the reconstruction matrix off the inversion of the interaction
        matrix obtained in the alignment calibration procedure.

        Parameters
        ----------
        intMat : MatrixLike
            The interaction matrix used to create the reconstruction matrix.

        Returns
        -------
        recMat : MatrixLike
            Reconstruction matrix.
        """
        _logger.log(f"{self._create_rec_mat.__qualname__}")
        recMat = _np.linalg.pinv(intMat)
        self.recMat = recMat
        return recMat

    def _apply_command(self, fullCmd: _ot.ArrayLike) -> None:
        """
        Applies the full command to the devices.

        Parameters
        ----------
        fullCmd : list or ndarray
            Full command of the interaction matrix which commands all device's
            available motors.

        Returns
        -------
        None
        """
        device_commands = self._extract_cmds_to_apply(fullCmd)
        logMsg = ""
        for cmd, fnc, dev in zip(device_commands, self._moveFnc, self._devName):
            if cmd.to_ignore:
                logMsg += f"Skipping null command for {dev}\n"
            else:
                try:
                    logMsg += f"Commanding {cmd} to {dev}\n"
                    fnc(cmd.vect)
                    _logger.log(f"{fnc.__qualname__} : {cmd.vect}")
                except Exception as e:
                    print(e)
        logMsg += "-" * 30
        print(logMsg)

    def _extract_cmds_to_apply(self, fullCmd: _ot.ArrayLike) -> _ot.ArrayLike:
        """
        Extracts the commands to be applied from the full command.

        Parameters
        ----------
        fullCmd : ArrayLike
            The full command from which individual device commands are extracted.

        Returns
        -------
        device_commands : ArrayLike
            The list of commands to be applied to each device.
        """
        _logger.log(f"{self._extract_cmds_to_apply.__qualname__}")
        commands = []
        for d, dof in enumerate(self._dof):
            dev_cmd = _np.zeros(self._dofTot[d])
            dev_idx = fullCmd[self._idx[d]]
            for i, idx in enumerate(dev_idx):
                dev_cmd[dof[i]] = idx
            commands.append(_Command(dev_cmd))
        positions = self.read_positions(show=False)
        device_commands = []
        for pos, cmd in zip(positions, commands):
            res_cmd = pos + cmd
            device_commands.append(res_cmd)
        return device_commands

    def _img_acquisition(
        self, k: int, template: _ot.ArrayLike, n_frames: int
    ) -> _ot.CubeData:
        """
        Acquires images based on the given template.

        Parameters
        ----------
        k : int
            The index of the command matrix column.
        template : list
            The template used for image acquisition.
        n_frames : int
            The number of frames to be acquired.

        Returns
        -------
        imglist : CubeData
            The list of acquired images.
        """
        _logger.log(f"{self._img_acquisition.__qualname__}")
        _logger.log(f"{self._acquire[0].__qualname__}")
        imglist = [self._acquire[0](nframes=n_frames)]
        for t in template:
            logMsg = ""
            logMsg += f"t = {t}"
            cmd = self.cmdMat.T[k] * self._cmdAmp[k] * t
            logMsg += f" - Full Command : {cmd}"
            print(logMsg)
            self._apply_command(cmd)
            _logger.log(f"{self._acquire[0].__qualname__}")
            imglist.append(self._acquire[0](nframes=n_frames))
        return imglist

    def _push_pull_redux(
        self, imglist: _ot.CubeData, template: _ot.ArrayLike
    ) -> _ot.ImageData:
        """
        Reduces the push-pull images based on the given template.

        Parameters
        ----------
        imglist : ArrayLike
            The list of images to be reduced.
        template : ArrayLike
            The template used for image reduction.

        Returns
        -------
        image : ImageData
            The reduced image.
        """
        _logger.log(f"{self._push_pull_redux.__qualname__}")
        template.insert(0, 1)
        image = _np.zeros((imglist[0].shape[0], imglist[0].shape[1]))
        for x in range(1, len(imglist)):
            opd2add = imglist[x] * template[x] + imglist[x - 1] * template[x - 1]
            mask2add = _np.ma.mask_or(imglist[x].mask, imglist[x - 1].mask)
            if x == 1:
                master_mask = mask2add
            else:
                master_mask = _np.ma.mask_or(master_mask, mask2add)
            image += opd2add
        image = _np.ma.masked_array(image, mask=master_mask) / 6
        template.pop(0)
        return image

    def __loadIntMat(self, calibtn: str | None) -> _ot.MatrixLike:
        """
        Loads the interaction matrix from a FITS file based on the provided tracking number.

        Parameters
        ----------
        calibtn : str, optional
            The tracking number of the interaction matrix to be loaded.

        Returns
        -------
        intMat : MatrixLike
            The loaded interaction matrix.

        Raises
        ------
        FileNotFoundError
            If the interaction matrix file does not exist.
        """
        if calibtn is None:
            return None
        filename = _os.path.join(
            _fn.ALIGN_CALIBRATION_ROOT_FOLDER, calibtn, "InteractionMatrix.fits"
        )
        if not _os.path.exists(filename):
            raise FileNotFoundError(
                f"Interaction matrix file '{filename}' does not exist."
            )
        intMat = _rfits(filename)
        return intMat

    @staticmethod
    def __get_callables(
        devices: _ot.GenericDevice | list[_ot.GenericDevice], callables: list[str]
    ) -> list[_ot.Callable[..., _ot.Any]]:
        """
        Returns a list of callables for the instanced object, taken from the
        configuration.py file.

        Parameters
        ----------
        devices : object
            The device object for which callables are retrieved.
        callables : list
            The list of callable names to be retrieved.

        Returns
        -------
        functions : list
            List of callables, which interacts with the input object of the class.
        """
        if not isinstance(devices, list):
            devices = [devices]
        functions = []
        for dev in devices:
            for dev_call in callables:
                obj, *methods = dev_call.split(".")
                call = getattr(dev, obj)
                for method in methods:
                    call = getattr(call, method)
                functions.append(call)
        return functions

    @staticmethod
    def __get_dev_names(names: list[str], ndev: int) -> list[str]:
        """
        Returns the names of the devices.

        Parameters
        ----------
        names : list
            The list of device names.

        Returns
        -------
        names : list
            The list of device names.
        """
        dev_names = []
        try:
            for x in names:
                dev_names.append(x)
        except TypeError:
            for x in range(ndev):
                dev_names.append(f"Device {x}")
        return dev_names


class _Command:
    """
    The _Command class represents a command with a vector and a flag indicating
    whether the command should be ignored. It provides methods for initializing
    the command, combining it with other commands, and checking if it is null.

    Attributes:
        vect (_np.ndarray): The vector representing the command.
        to_ignore (bool): A flag indicating whether the command should be ignored.

    Methods:
        __init__(vector=None, to_ignore:bool=None):
            Initializes a new instance of the _Command class.
        __repr__():
            Returns a string representation of the _Command instance.
        __str__():
            Returns the string representation of the command vector.
        __add__(other):
            Combines the current command with another _Command instance.
        is_null():
            Determines whether the command is null, i.e., a sequence of zeros.
        _process_command_logic(P, C, S):
            Processes the command logic to determine the to_ignore flag.
    """

    def __init__(
        self, vector: _ot.ArrayLike = None, to_ignore: _ot.Optional[bool] = None
    ):
        """
        Initializes a new instance of the _Command class.

        Parameters
        ----------
        vector : list or _np.ndarray, optional
            The vector representing the command. If a list is provided, it will
            be converted to a numpy array.
        to_ignore : bool, optional
            A flag indicating whether the command should be ignored.
        """
        self.vect = _np.array(vector) if isinstance(vector, list) else vector
        self.to_ignore = to_ignore

    def __repr__(self):
        """
        Returns a string representation of the _Command instance.

        Returns
        -------
        str
            A string representation of the _Command instance.
        """
        if self.to_ignore is not None:
            return f"Command({self.vect}, to_ignore={self.to_ignore})"
        else:
            return f"Command({self.vect},)"

    def __str__(self):
        """
        Returns the string representation of the command vector.

        Returns
        -------
        str
            The string representation of the command vector.
        """
        return self.vect.__str__()

    def __add__(self, other: "_Command") -> "_Command":
        """
        Combines the current command with another _Command instance.

        Parameters
        ----------
        other : _Command
            Another instance of the _Command class.

        Returns
        -------
        _Command
            A new _Command instance with the combined vector and updated
            'to_ignore 'flag.

        Raises
        ------
        NotImplementedError
            If the vectors of the commands are not numpy arrays.
        """
        if not isinstance(other, _Command):
            return NotImplementedError
        if not isinstance(self.vect, _np.ndarray) and not isinstance(
            other.vect, _np.ndarray
        ):
            raise NotImplementedError(
                f"Operation not supported for operands types {type(self.vect)} and {type(other)}"
            )
        combined_vect = self.vect + other.vect
        to_ignore = self._process_command_logic(self, other, combined_vect)
        return _Command(combined_vect, to_ignore)

    @property
    def is_null(self) -> bool:
        """
        Determines whether the command is null, i.e., a sequence of zeros.

        Returns
        -------
        bool
            True if the command is null, False otherwise.
        """
        return _np.all(self.vect == 0)

    def _process_command_logic(
        self, P: "_Command", C: "_Command", S: "_Command"
    ) -> bool:
        """
        Processes the command logic to determine the to_ignore flag.

        Parameters
        ----------
        P : _Command
            The previous command instance.
        C : _Command
            The current command instance.
        S : _np.ndarray
            The sum of the vectors of the previous and current commands.

        Returns
        -------
        bool
            The decision for the to_ignore flag based on the command logic.
        """
        # P = current device position
        # C = received device command
        # S = sum of P and C - command to apply (absolute)
        # _________________________________________________#
        # If S = 0
        if _np.all(S == 0):
            # C ≠ 0 and P ≠ 0 → TO_NOT_IGNORE
            if not P.is_null and not C.is_null and _np.array_equal(C.vect, -1 * P.vect):
                decision = False
            # C = 0 and P = 0 → TO_IGNORE
            elif C.is_null and P.is_null:
                decision = True
        # If S ≠ 0
        else:
            # P ≠ 0 and C = 0 → TO_IGNORE
            if not P.is_null and C.is_null and _np.array_equal(S, P.vect):
                decision = True
            # C ≠ 0 and P ≠ 0 → TO_NOT_IGNORE
            elif not C.is_null and not P.is_null:
                decision = False
            # P = 0 and C ≠ 0 → TO_NOT_IGNORE
            elif P.is_null and not C.is_null and _np.array_equal(S, C.vect):
                decision = False
        return decision
