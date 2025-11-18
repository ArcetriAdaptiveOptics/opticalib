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
        self._shuffle = 0
        # Matrices
        self.timedCmdHistory = None
        self.cmdMatHistory = None
        self.auxCmdHistory = None
        self.triggPadCmdHist = None
        self.regPadCmdHist = None

    def createTimedCmdHistory(
        self,
        modesList: _ot.Optional[_ot.ArrayLike] = None,
        modesAmp: _ot.Optional[float | _ot.ArrayLike] = None,
        template: _ot.Optional[_ot.ArrayLike] = None,
        shuffle: bool = False,
    ) -> _ot.MatrixLike:
        """
        Function that creates the final timed command history to be applied

        Parameters
        ----------
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

        Returns
        -------
        timedCmdHist : float | ArrayLike
            Final timed command history, including the trigger padding, the
            registration pattern and the command matrix history.
        """
        self.createCmdMatrixHistory(modesList, modesAmp, template, shuffle)
        self.createAuxCmdHistory()
        if not self.auxCmdHistory is None:
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
        mlist: _ot.Optional[_ot.ArrayLike] = None,
        modesAmp: _ot.Optional[float | _ot.ArrayLike] = None,
        template: _ot.Optional[_ot.ArrayLike] = None,
        shuffle: bool = False,
    ) -> _ot.MatrixLike:
        """
        Creates the command matrix history for the IFF acquisition.

        Parameters
        ----------
        mlist : ArrayLike
            List of selected modes to use. If no argument is passed, it will
        modesAmp : float | ArrayLike
            Amplitude of the modes to be commanded. If no argument is passed,
            it will be loaded from the configuration file iffConfig.ini
        template : ArrayLike
            Template for the push-pull application of the modes. If no argument
            is passed, it will be loaded from the configuration file iffConfig.ini
        shuffle : bool
            Decides to wether shuffle or not the order in which the modes are
            applied. Default is False

        Returns
        -------
        cmd_matrixHistory : MatrixLike
            Command matrix history to be applied, with the correct push-pull
            application, following the desired template.
        """
        _, _, infoIF, _ = _getAcqInfo()
        if mlist is None:
            mlist = infoIF.get("modes")
        else:
            mlist = mlist
            infoIF["modes"] = mlist
        modesAmp = modesAmp if modesAmp is not None else infoIF.get("amplitude")
        template = template if template is not None else infoIF.get("template")
        zeroScheme = infoIF["zeros"]
        self._template = template
        self._createCmdMatrix(mlist)
        self._modesList = mlist
        nModes = self._cmdMatrix.shape[1]
        n_push_pull = len(template)
        if _np.size(modesAmp) == 1:
            modesAmp = _np.full(nModes, modesAmp)
        self._modesAmp = modesAmp
        if shuffle is not False:
            self._shuffle = shuffle
            cmd_matrix = _np.zeros((self._cmdMatrix.shape[0], self._cmdMatrix.shape[1]))
            modesList = _np.copy(self._modesList)
            _np.random.shuffle(modesList)
            k = 0
            for i in modesList:
                cmd_matrix.T[k] = self._cmdMatrix[i]
                k += 1
            self._indexingList = _np.arange(0, len(modesList), 1)
        else:
            cmd_matrix = self._cmdMatrix
            modesList = self._modesList
            self._indexingList = _np.arange(0, len(modesList), 1)
        n_frame = len(self._modesList) * n_push_pull
        cmd_matrixHistory = _np.zeros(
            (self._NActs, n_frame + zeroScheme + 5)
        )  # TODO -> fix it by reading a new configuration entry, like 'paddingZeros'
        k = zeroScheme
        for i in range(nModes):
            for j in range(n_push_pull):
                # k = zeroScheme + cmd_matrix.shape[1]*j + i
                cmd_matrixHistory.T[k] = cmd_matrix[:, i] * template[j] * modesAmp[i]
                k += 1
        self.cmdMatHistory = cmd_matrixHistory
        return cmd_matrixHistory

    def createAuxCmdHistory(self) -> _ot.MatrixLike:
        """
        Creates the initial part of the final command history matrix that will
        be passed to M4. This includes the Trigger Frame, the first frame to
        have a non-zero command, and the Padding Frame, two frames with high
        rms, useful for setting a start to the real acquisition.

        Result
        ------
        aus_cmdHistory : MatrixLike
            The auxiliary command history, which includes the trigger padding
            and the registration pattern. This matrix is used to create the
            final command history to be passed to the DM.
        """
        self._createTriggerPadding()
        self._createRegistrationPattern()
        if self.triggPadCmdHist is not None and self.regPadCmdHist is not None:
            aux_cmdHistory = _np.hstack((self.triggPadCmdHist, self.regPadCmdHist))
        elif self.triggPadCmdHist is not None:
            aux_cmdHistory = self.triggPadCmdHist
        elif self.regPadCmdHist is not None:
            aux_cmdHistory = self.regPadCmdHist
        else:
            aux_cmdHistory = None
        self.auxCmdHistory = aux_cmdHistory
        return aux_cmdHistory

    def _createRegistrationPattern(self) -> _ot.MatrixLike:
        """
        Creates the registration pattern to apply after the triggering and before
        the commands to apply for the IFF acquisition. The information about number
        of zeros, mode(s) and amplitude are read from the 'iffconfig.ini' file.

        Returns
        -------
        regHist : MatrixLike
            Registration pattern command history

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
        self.regPadCmdHist = regHist
        return regHist

    def _createTriggerPadding(self) -> _ot.MatrixLike:
        """
        Function that creates the trigger padding scheme to apply before the
        registration padding scheme. The information about number of zeros,
        mode(s) and amplitude are read from the 'iffconfig.ini' file.

        Returns
        -------
        triggHist : MatrixLike
            Trigger padding command history
        """
        infoT = _rif.getIffConfig("TRIGGER")
        if len(infoT["modes"]) == 0:
            return
        self._updateModalBase(infoT["modalBase"])
        zeroScheme = _np.zeros((self._NActs, infoT["zeros"]))
        trigMode = self._modalBase[:, infoT["modes"]] * infoT["amplitude"]
        triggHist = _np.hstack((zeroScheme, trigMode))
        self.triggPadCmdHist = triggHist
        return triggHist

    def _createCmdMatrix(self, mlist: int | _ot.ArrayLike) -> _ot.MatrixLike:
        """
        Cuts the modal base according the given modes list
        """
        infoIF = _rif.getIffConfig("IFFUNC")
        self._updateModalBase(infoIF["modalBase"])
        self._cmdMatrix = self._modalBase[:, mlist]
        return self._cmdMatrix

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
