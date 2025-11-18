import os
import numpy as np
from opticalib import folders as fn
from opticalib import typings as ot

try:
    from Microgate.adopt.AOClient import AO_CLIENT  # type: ignore
except ImportError:
    pass


mirrorModesFile = "ff_v_matrix.fits"
ffFile = "ff_matrix.fits"
actCoordFile = "ActuatorsCoordinates.fits"
nActFile = "nActuators.dat"


class BaseAdOpticaDm:
    """
    Base class for AdOptica DM devices.
    This class is intended to be inherited by specific device classes.
    """

    def __init__(self, tracknum: str = None):
        """The constructor"""
        """
        print(f"Initializing the M4AU with configuration: '{os.path.join(fn.MIRROR_FOLDER,tracknum)}'")
        self.dmConf      = os.path.join(fn.MIRROR_FOLDER,tracknum)
        """
        self._aoClient = AO_CLIENT(tracknum)
        self.ffm = (self._aoClient.aoSystem.sysConf.gen.FFWDSvdMatrix)[0]  #
        self.ff = self._aoClient.aoSystem.sysConf.gen.FFWDMatrix
        self._biasCmd = self._aoClient.aoSystem.sysConf.gen.biasVectors[0]
        self.nActs = self._initNActuators()
        self.mirrorModes = self._initMirrorModes()
        self.actCoord = self._initActCoord()
        self.workingActs = self._initWorkingActs()
        self._aoClient._connect()
        self._enumerateDevices()

    def getCounter(self):
        """
        Function which returns the current shape of the mirror.

        Returns
        -------
        shape: numpy.ndarray
            Current shape of the mirror.
        """
        fc = self._aoClient.getCounters()
        skipByCommand = fc.skipByCommand
        # .....
        return skipByCommand

    def get_force(self):
        """
        Function which returns the current force applied to the mirror.

        Returns
        -------
        force: numpy.ndarray
            Current force applied to the mirror actuators.

        """
        # micLibrary.getForce()
        force = self._aoClient.getForce()
        return force

    def plot_acts(self, amp: ot.Optional[ot.ArrayLike] = None, **kwargs):
        """
        Function which plots the actuators.

        Parameters
        ----------
        amp: ot.ArrayLike
            Amplitude to be plotted.
        **kwargs: dict
            Additional keyword arguments for plotting.
        """
        xA = self.actCoord[0:111, 0]
        yA = self.actCoord[0:111, 1]
        xB = self.actCoord[111:, 0]
        yB = self.actCoord[111:, 1]
        import matplotlib.pyplot as plt

        plt.figure()
        if amp is None:
            col = col2 = "black"
        elif amp.shape[0] == 222:
            col = amp[:111]
            col2 = amp[111:]
        else:
            col = col2 = amp
        plt.scatter(xA, yA, c=col, **kwargs)
        plt.scatter(xB, yB, c=col2, **kwargs)
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Actuators")
        plt.axis("equal")
        plt.colorbar()
        plt.show()

    def _initNActuators(self) -> int:
        """
        Function which reads the number of actuators of the DM from a configuration
        file.

        Returns
        -------
        nact: int
            number of actuators of the DM.
        """
        return self._aoClient.aoSystem.sysConf.gen.M2CMatrix.shape[-1]

    def _initMirrorModes(self):
        """
        Function which initialize the mirror modes by reading from a fits file.

        Returns
        -------
        mirrorModes: numpy.ndarray
            Mirror Modes Matrix.
        """
        cmdMat = np.zeros((self.nActs, 222))
        mirrorModes = np.array(
            self._aoClient.aoSystem.sysConf.gen.FFWDSvdMatrix[0]
        )  # (2,111,111)
        shell1 = np.zeros((111, 111))
        shell2 = np.zeros((111, 111))
        for m in range(111):
            shell1[:, m] = mirrorModes[0, :, m] / np.std(mirrorModes[0, :111, m])
            shell2[:, m] = mirrorModes[1, :, m] / np.std(mirrorModes[1, :, m])

        cmdMat[:111, :111] = shell1
        cmdMat[111:222, 111:222] = shell2
        return cmdMat

    def _initWorkingActs(self):
        """
        Function which initialize the working actuators by reading
        a list from a fits file.

        Returns
        -------
        workingActs: numpy.ndarray
            Working Actuators Matrix.
        """
        pass
        # fname = os.path.join(self.dmConf, mirrorModesFile)
        # if os.path.exists(fname):
        #     with pyfits.open(fname) as hdu:
        #         workingActs = hdu[0].data
        # else:
        #     workingActs = np.eye(self.nActs)
        # return workingActs

    def _initActCoord(self):
        """
        Reading the actuators coordinate from file
        """
        from opticalib import load_fits

        filepath = os.path.join(fn.CONFIGURATION_FOLDER, actCoordFile)
        coords = load_fits(filepath)
        return coords

    def _enumerateDevices(self):
        """
        Function which enumerates the connected devices.
        """
        self._aoClient.aoSystem.aoSubSystem0.deviceEnum()
