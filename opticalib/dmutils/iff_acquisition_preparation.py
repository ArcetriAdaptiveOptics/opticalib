"""
This module contains the IFFCapturePreparation class, a class which serves as a
preparator for the Influence Function acquisition, creating the timed command
matrix history that will be ultimately used.

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it

"""

import os as _os
import numpy as _np
from opticalib.ground import osutils as _osu
from opticalib.core import read_config as _rif
from opticalib.core.fitsarray import fits_array as _fa
from opticalib import typings as _ot
from .iff_processing import _getAcqInfo


class IFFCapturePreparation:
    """
    Class containing all the functions necessary to create the final timed
    command matrix history to be executed by M4

    Import and Initialization
    -------------------------
    Import the module and initialize the class with a deformable mirror object

    >>> from opticalib.dmutils.iff_acquisition_preparation import IFFCapturePreparation
    >>> from opticalib.devices import AlpaoDm
    >>> dm = AlpaoDm(88)
    >>> ifa = IFFCapturePreparation(dm)

    Methods
    -------
    createTimedCmdHistory

        Creates the final timed command matrix history. Takes 4 positional optional
        arguments, which will be read from a configuration file if not passed

    createCmdMatrixhistory

        Takes the modal base loaded into the class (which can be updated using
        the sub-method _updateModalBase) and returns the wanted command matrix
        with the dedired modes and amplitudes, which can be either passed on as
        arguments or read automatically from a configuration file.

        >>> # As example, wanting to update the modal base using a zonal one
        >>> ifa._updateModalBase('zonal')
        'Using zonal modes'

    createAuxCmdHistory

        Creates the auxiliary command matrix to attach to the command matrix
        history. This auxiliary matrix comprehends the trigger padding and the
        registration padding schemes. the parameters on how to create these
        schemes is written in a configuration file.

    getInfoToSave

        A function that returns a dictionary containing all the useful information
        to save, such as the command matrix used, the used mode list, the indexing
        the amplitudes, the used tamplate and the shuffle option.

    """

    def __init__(self, dm: _ot.DeformableMirrorDevice):
        """The Constructor"""
        # DM information
        if not _ot.isinstance_(dm, "DeformableMirrorDevice"):
            from opticalib.core.exceptions import DeviceError

            raise DeviceError(dm, "DeformableMirrorDevice")

        self._dm = dm
        self.mirrorModes = dm.mirrorModes
        self._NActs = dm.nActs

        # IFF info
        self.modalBaseId = None
        self._modesList = None
        self._modalBase = self.mirrorModes
        self._regActs = None
        self._cmdMatrix = None
        self._indexingList = None
        self._modesAmp = None
        self._template = None
        self._shuffle = False

        # Matrices
        self.timedCmdHistory = None
        self.cmdMatHistory = None
        self.auxCmdHistory = None
        self.triggPadCmdHist = None
        self.regPadCmdHist = None

    def createTimedCmdHistory(
        self,
        cmdMat: _ot.Optional[_ot.MatrixLike] = None,
        triggerMat: _ot.Optional[_ot.MatrixLike] = None,
        registrationMat: _ot.Optional[_ot.MatrixLike] = None,
        modesList: _ot.Optional[_ot.ArrayLike] = None,
        modesAmp: _ot.Optional[float | _ot.ArrayLike] = None,
        template: _ot.Optional[_ot.ArrayLike] = None,
        modalBase: str = None,
        shuffle: bool = False,
        n_repetitions: int = 1,
    ) -> _ot.MatrixLike:
        """
        Function that creates the final timed command history to be applied

        Parameters
        ----------
        cmdMat : MatrixLike
            Command matrix to be used. Default is None, that means the command
            matrix is created using the 'modesList' argument or the configuration
            file.
        triggerMat : MatrixLike
            Trigger matrix to be used. Default is None, that means the trigger
            matrix is created using the configuration file.
        registrationMat : MatrixLike
            Registration matrix to be used. Default is None, that means the
            registration matrix is created using the configuration file.
        modesList : int | ArrayLike
            List of selected modes to use. Default is None, that means all modes
            of the base command matrix are used.
        modesAmp : float
            Amplitude of the modes. Default is None, that means the value is
            loaded from the 'iffconfig.ini' file
        template : int | ArrayLike
            Template for the push-pull measures. List of 1 and -1. Default is
            None, which means the template is loaded from the 'iffcongig.ini' file.
        shuffle : boolean
            Decide wether to shuffle or not the modes order. Default is False
        modalBase : str, optional
            Modal base to use. Default is None, which means the modal base is
            loaded from the 'iffconfig.ini' file.
        n_repetitions : int
            Number of times the command matrix is repeated. Default is 1.

        Returns
        -------
        timedCmdHist : float | ArrayLike
            Final timed command history, including the trigger padding, the
            registration pattern and the command matrix history.
        """
        # Provide manually the cmdMatrixHistory
        if cmdMat is not None:
            _, _, infoIF = _getAcqInfo()
            trailing_zeros = _np.zeros((cmdMat.shape[0], infoIF["paddingZeros"]))
            self._cmdMatrix = cmdMat
            cmdMat = _np.hstack((cmdMat, trailing_zeros))
            self.cmdMatHistory = cmdMat
            self._modesList = modesList
            self._modesAmp = modesAmp
            self._template = template
            self._shuffle = shuffle
            self._indexingList = _np.arange(0, len(modesList), 1)
            self._n_repetitions = n_repetitions

        elif self.cmdMatHistory is None:
            self.createCmdMatrixHistory(
                modesList, modesAmp, template, modalBase, shuffle, n_repetitions
            )

        self.triggPadCmdHist = triggerMat.copy() if triggerMat is not None else None
        self.regPadCmdHist = registrationMat.copy() if registrationMat is not None else None

        # Create the auxiliary command history if needed
        if self.auxCmdHistory is None:
            self.createAuxCmdHistory()

        if self.auxCmdHistory is not None:
            cmdHistory = _np.hstack((self.auxCmdHistory, self.cmdMatHistory))
        else:
            cmdHistory = self.cmdMatHistory
            self._regActs = _np.array([])

        timing = _rif.getTiming()
        timedCmdHist = _np.repeat(cmdHistory, timing, axis=1)
        self.timedCmdHistory = timedCmdHist
        return timedCmdHist

    def getInfoToSave(self) -> dict[str, _ot.Any]:
        """
        Return the data to save as fits files, arranged in a dictionary

        Returns
        -------
        info : dict
            Dictionary containing all the vectors and matrices needed
        """
        info = {
            "timedCmdHistory": self.timedCmdHistory,
            "cmdMatrix": self._cmdMatrix,
            "modesVector": self._modesList,
            "regActs": self._regActs,
            "ampVector": self._modesAmp,
            "indexList": self._indexingList,
            "template": self._template,
            "shuffle": self._shuffle,
        }
        return info

    def createCmdMatrixHistory(
        self,
        modesList: _ot.Optional[_ot.ArrayLike] = None,
        modesAmp: _ot.Optional[float | _ot.ArrayLike] = None,
        template: _ot.Optional[_ot.ArrayLike] = None,
        modalBase: _ot.Optional[str] = None,
        shuffle: bool = False,
        n_repetitions: int = 1,
    ) -> _ot.MatrixLike:
        """
        Creates the command matrix history for the IFF acquisition.

        Parameters
        ----------
        modesList : ArrayLike
            List of selected modes to use. If no argument is passed, it will
            be loaded from the configuration file iffConfig.ini
        modesAmp : float | ArrayLike
            Amplitude of the modes to be commanded. If no argument is passed,
            it will be loaded from the configuration file iffConfig.ini
        template : ArrayLike
            Template for the push-pull application of the modes. If no argument
            is passed, it will be loaded from the configuration file iffConfig.ini
        modalBase : str, optional
            Modal base to use. Default is None, which means the modal base is
            loaded from the 'iffconfig.ini' file.
        shuffle : bool
            Decides to wether shuffle or not the order in which the modes are
            applied. Default is False
        n_repetitions : int
            Number of times the command matrix is repeated. 
            Default is 1.

        Returns
        -------
        cmd_matrixHistory : MatrixLike
            Command matrix history to be applied, with the correct push-pull
            application, following the desired template.
        """
        _, _, infoIF, _ = _getAcqInfo()
        modesList = _np.asarray(
            modesList if modesList is not None else infoIF.get("modes"),
            dtype=int
        )
        template = _np.asarray(
            template if template is not None else infoIF.get("template"),
            dtype=int
        )
        modesAmp = modesAmp if modesAmp is not None else infoIF.get("amplitude")
        zeroScheme = infoIF["zeros"]

        self._createCmdMatrix(modesList, modalBase)
        A, M = self._cmdMatrix.shape
        n_push_pull = len(template)

        if _np.size(modesAmp) == 1:
            modesAmp = _np.full(M, modesAmp)
        elif _np.size(modesAmp) != M:
            raise ValueError(
                f"Length of modesAmp ({_np.size(modesAmp)}) must be either 1 or"
                f" equal to the number of modes ({M})"
            )

        if shuffle is not False:            
            final_cmd_mat = _np.zeros(
                (A, M * n_repetitions)
            )
            final_mlist = _np.zeros(M * n_repetitions, dtype=int)
            final_ilist = _np.zeros(M * n_repetitions, dtype=int)
            final_amps  = _np.zeros(len(modesAmp) * n_repetitions)
            
            for R in range(n_repetitions):
                indexList = _np.arange(M)
                _np.random.shuffle(indexList)
                
                cmd_matrix = self._cmdMatrix[:, indexList]
                final_cmd_mat[:, R*M:(R+1)*M] = cmd_matrix
                final_ilist[R*M:(R+1)*M] = indexList
                final_mlist[R*M:(R+1)*M] = modesList[indexList]
                final_amps[R*M:(R+1)*M] = modesAmp[indexList]

        else:
            final_cmd_mat = _np.tile(self._cmdMatrix, (1, n_repetitions))
            final_mlist = _np.tile(modesList, n_repetitions)
            final_ilist = _np.tile(_np.arange(len(modesList)), n_repetitions)
            final_amps = _np.tile(modesAmp, n_repetitions)

        n_frame = len(final_mlist) * n_push_pull
        cmd_matrixHistory = _np.zeros(
            (self._NActs, n_frame + zeroScheme + infoIF["paddingZeros"])
        )
        n_modes = final_cmd_mat.shape[1]

        k = zeroScheme
        for i in range(n_modes):
            # for j in range(n_push_pull):
            #     cmd_matrixHistory.T[k] = final_cmd_mat[:, i] * template[j] * modesAmp[i]
            #     k += 1
            scaled_cmd = final_cmd_mat[:, i:i+1] * template[_np.newaxis, :] * final_amps[i]
            cmd_matrixHistory[:, k:k+n_push_pull] = scaled_cmd
            k += n_push_pull
        
        header = {
            'SHUFFLE': shuffle,
            'N_REP': n_repetitions,
        }

        cmdMatHist = _fa(
            cmd_matrixHistory,
            header=header
        )
        
        self._modesList = _fa(final_mlist, header=header)
        self._indexingList = _fa(final_ilist, header=header)
        self._modesAmp = _fa(final_amps, header=header)
        self._template = _fa(template)
        self._shuffle = shuffle
        self._n_repetitions = n_repetitions
        self.cmdMatHistory = cmdMatHist.copy()
        return cmdMatHist

    def createAuxCmdHistory(self) -> _ot.Optional[_ot.MatrixLike]:
        """
        Creates the initial part of the final command history matrix that will
        be passed to M4. This includes the Trigger Frame, the first frame to
        have a non-zero command, and the Padding Frame, two frames with high
        rms, useful for setting a start to the real acquisition.

        Result
        ------
        aus_cmdHistory : MatrixLike or None
            The auxiliary command history, which includes the trigger padding
            and the registration pattern. This matrix is used to create the
            final command history to be passed to the DM.
        """
        self._createTriggerPadding() if self.triggPadCmdHist is None else None
        self._createRegistrationPattern() if self.regPadCmdHist is None else None
        # if self.triggPadCmdHist is not None and self.regPadCmdHist is not None:
        #     aux_cmdHistory = _np.hstack((self.triggPadCmdHist, self.regPadCmdHist))
        # elif self.triggPadCmdHist is not None:
        #     aux_cmdHistory = self.triggPadCmdHist
        # elif self.regPadCmdHist is not None:
        #     aux_cmdHistory = self.regPadCmdHist
        # else:
        #     aux_cmdHistory = None
        matrices = [m for m in [self.triggPadCmdHist, self.regPadCmdHist] if m is not None]
        aux_cmdHistory = _np.hstack(matrices) if matrices else None
        self.auxCmdHistory = aux_cmdHistory

    def _createRegistrationPattern(self) -> None:
        """
        Creates the registration pattern to apply after the triggering and before
        the commands to apply for the IFF acquisition. The information about number
        of zeros, mode(s) and amplitude are read from the 'iffconfig.ini' file.
        """
        infoR = _rif.getIffConfig("REGISTRATION")
        if len(infoR["modes"]) == 0:
            self._regActs = infoR["modes"]
            return
        self._regActs = infoR["modes"]
        self._updateModalBase(infoR["modalBase"])
        zeroScheme = _np.zeros((self._NActs, infoR["zeros"]))
        regScheme = _np.zeros(
            (self._NActs, len(infoR["template"]) * len(infoR["modes"]))
        )
        k = 0
        for mode in infoR["modes"]:
            for t in range(len(infoR["template"])):
                regScheme.T[k] = (
                    self._modalBase.T[mode] * infoR["amplitude"] * infoR["template"][t]
                )
                k += 1
        regHist = _np.hstack((zeroScheme, regScheme))
        self.regPadCmdHist = regHist.copy()

    def _createTriggerPadding(self) -> None:
        """
        Function that creates the trigger padding scheme to apply before the
        registration padding scheme. The information about number of zeros,
        mode(s) and amplitude are read from the 'iffconfig.ini' file.
        """
        infoT = _rif.getIffConfig("TRIGGER")
        if len(infoT["modes"]) == 0:
            return
        self._updateModalBase(infoT["modalBase"])
        zeroScheme = _np.zeros((self._NActs, infoT["zeros"]))
        trigMode = self._modalBase[:, infoT["modes"]] * infoT["amplitude"]
        triggHist = _np.hstack((zeroScheme, trigMode))
        self.triggPadCmdHist = triggHist.copy()

    def _createCmdMatrix(
        self, mlist: int | _ot.ArrayLike, mbase: str = None
    ) -> None:
        """
        Cuts the modal base according the given modes list
        """
        infoIF = _rif.getIffConfig("IFFUNC")
        modalbase = mbase or infoIF["modalBase"]
        self._updateModalBase(modalbase)
        self._cmdMatrix = self._modalBase[:, mlist]

    def _updateModalBase(self, mbasename: _ot.Optional[str] = None) -> None:
        """
        Updates the used modal base

        Parameters
        ----------
        mbasename : str, optional
            Modal base name to be used. The default is None, which means
            the default `mirror` modal base is used. The other options are 'zonal',
            'hadamard' and a user-defined modal base, which must be a .fits file
            in the IFFUNCTIONS_ROOT_FOLDER folder.
        """
        if (mbasename is None) or (mbasename == "mirror"):
            self.modalBaseId = mbasename
            self._modalBase = self.mirrorModes
        elif mbasename == "zonal":
            self.modalBaseId = mbasename
            self._modalBase = self._createZonalMat()
        elif mbasename == "hadamard":
            self.modalBaseId = mbasename
            self._modalBase = self._createHadamardMat()
        elif mbasename == "mirror":
            self.modalBaseId = mbasename
            self._modalBase = self.mirrorModes
        else:
            self.modalBaseId = mbasename
            self._modalBase = self._createUserMat(mbasename)

    def _createUserMat(self, name: str = None) -> _ot.MatrixLike:
        """
        Create a user-defined modal base, given the name of the file.
        The file must be a .fits file, and it must be in the
        IFFUNCTIONS_ROOT_FOLDER folder.

        Parameters
        ----------
        name : str
            Name of the file to be used as modal base. The file must be a .fits
            file, and it must be in the IFFUNCTIONS_ROOT_FOLDER folder.

        Returns
        -------
        cmdBase : MatrixLike
            The command matrix to be used as modal base.

        """
        from opticalib.core.root import MODALBASE_ROOT_FOLDER

        if ".fits" not in name:
            name = name + ".fits"
        try:
            mbfile = _os.path.join(MODALBASE_ROOT_FOLDER, name)
            cmdBase = _osu.load_fits(mbfile)
        except FileNotFoundError as f:
            raise f((f"'{name}' not found in {MODALBASE_ROOT_FOLDER}"))
        print(f"Loaded user-defined modal base: `{name}`")
        return cmdBase

    def _createZonalMat(self) -> _ot.MatrixLike:
        """
        Create the zonal matrix to use as modal base, with size (nacts, nacts).

        Returns
        -------
        cmdBase : MatrixLike
            The zonal matrix, with size (nacts, nacts).

        """
        cmdBase = _np.eye(self._NActs)
        return cmdBase

    def _createHadamardMat(self) -> _ot.MatrixLike:
        """
        Create the hadamard matrix to use as modal base, with size
        (nacts, nacts), removed of piston mode.

        Returns
        -------
        cmdBase : MatrixLike
            The Hadamard matrix, with size (nacts, nacts), removed of
            the piston mode.
        """
        from scipy.linalg import hadamard
        import math

        numb = math.ceil(math.log(self._NActs, 2))
        hadm = hadamard(2**numb)  # 892, 1 segment
        cmdBase = hadm[1 : self._NActs + 1, 1 : self._NActs + 1]
        return cmdBase
