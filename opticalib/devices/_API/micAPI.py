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
nActFile = "n_actuators.dat"


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
        self.n_acts = self._init_n_actuators()
        self.mirrorModes = self._init_mirror_modes()
        self.act_coord = self._init_act_coord()
        self.workingActs = self._init_working_acts()
        self._aoClient._connect()
        self._enumerate_devices()

    def get_counter(self):
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
        # micLibrary.get_force()
        force = self._aoClient.get_force()
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
        xA = self.act_coord[0:111, 0]
        yA = self.act_coord[0:111, 1]
        xB = self.act_coord[111:, 0]
        yB = self.act_coord[111:, 1]
        import matplotlib.pyplot as plt

        plt.figure()
        if amp is None:
            col = col2 = "black"
        elif amp.shape[0] == 222:
            col = amp[:111]
            col2 = amp[111:]
        else:
            col = col2 = amp
        s = kwargs.pop("s", 250)
        eg = kwargs.pop("edgecolors", "gray")

        plt.gca().set_facecolor((0, 0, 0, 0.05))
        plt.scatter(xA, yA, c=col, s=s, edgecolors=eg, **kwargs)
        plt.scatter(xB, yB, c=col2, s=s, edgecolors=eg, **kwargs)
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("Actuators")
        plt.axis("equal")
        plt.colorbar()
        plt.show()

    def _init_n_actuators(self) -> int:
        """
        Function which reads the number of actuators of the DM from a configuration
        file.

        Returns
        -------
        nact: int
            number of actuators of the DM.
        """
        return self._aoClient.aoSystem.sysConf.gen.M2CMatrix.shape[-1]

    def _init_mirror_modes(self):
        """
        Function which initialize the mirror modes by reading from a fits file.

        Returns
        -------
        mirrorModes: numpy.ndarray
            Mirror Modes Matrix.
        """
        cmdMat = np.zeros((self.n_acts, 222))
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

    def _init_working_acts(self):
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
        #     workingActs = np.eye(self.n_acts)
        # return workingActs

    def _init_act_coord(self):
        """
        Reading the actuators coordinate from file
        """
        from opticalib import load_fits

        filepath = os.path.join(fn.CONFIGURATION_FOLDER, actCoordFile)
        coords = load_fits(filepath)
        return coords

    def _enumerate_devices(self):
        """
        Function which enumerates the connected devices.
        """
        self._aoClient.aoSystem.aoSubSystem0.deviceEnum()
