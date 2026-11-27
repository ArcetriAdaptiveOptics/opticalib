import os
import numpy as np
from opticalib.core.root import folders as fn
from opticalib.core import _types as ot
from ..base_devices import BaseDeformableMirror

try:
    from Microgate.adopt.AOClient import AO_CLIENT  # type: ignore
except ImportError:
    pass

actCoordFile = "ActuatorsCoordinates.fits"


class BaseAdOpticaDm(BaseDeformableMirror):
    """
    Base class for AdOptica DM devices.
    This class is intended to be inherited by specific device classes.
    """

    def __init__(self, tracknum: str | None = None):
        """
        Constructor for the Microgate AO_CLIENT wrapper

        Parameters
        ----------
        tracknum: str, optional
            Track number for the AO_CLIENT. If None, it will be resolved automatically.
        """
        # Resolve the TN
        tn = self._resolve_tracknum(tracknum)
        self._aoClient = AO_CLIENT(tn)

        # Retrieves the mirror's FF, M2C and Bias Vector
        # self.ffm = (self._aoClient.aoSystem.sysConf.gen.FFWDSvdMatrix)[0]  #
        self.ff = self._aoClient.aoSystem.sysConf.gen.FFWDMatrix
        self._modal_base = self._aoClient.aoSystem.sysConf.gen.M2CMatrix.copy()
        # self.mirror_modes = self._init_mirror_modes()
        self.mirror_modes = None
        self._bias_cmd = self._aoClient.aoSystem.sysConf.gen.biasVectors[0]
        self._cmd_history = None

        # Mirror properties
        self.n_acts = self._modal_base.shape[-1]
        self.act_coord = self._init_act_coord()
        self.workingActs = self._init_working_acts()

        # Monolithic defaults (overridden by segmented subclasses)
        self.is_segmented = False
        self.n_segments = 1
        self.n_acts_per_segment = self.n_acts
        self._init_slaving_ids()

        # Connect to the Ao SubSystems
        self._aoClient._connect()
        # self._enumerate_devices()

    def get_counter(self):
        """
        Function which returns the current skip counter of the mirror.

        Returns
        -------
        counter: int
            Current skip counter of the mirror.
        """
        return self._aoClient.getCounters()

    def get_force(self):
        """
        Function which returns the current force applied to the mirror.

        Returns
        -------
        force: numpy.ndarray
            Current force applied to the mirror actuators.

        """
        return self._aoClient.getForce()

    def recover_shape(self):
        """
        Recover the current shape of the mirror by invoking the mirror recovery
        procedure.
        """
        try:
            self._aoClient.mirrorRecovery()
        except Exception as e:
            print(f"Error during mirror recovery: {e}")

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

    def _check_current_modal_base(self) -> str:
        """
        Check the current modal base of the mirror.

        Returns
        -------
        'zonal' or 'modal'.
        """
        if np.array_equal(self._modal_base, np.eye(self.n_acts)):
            return "zonal"
        return "modal"

    def _has_m2c_changed(self) -> bool:
        """
        Check if the M2C matrix has changed from its initial state.

        Returns
        -------
        bool
            True if the M2C matrix has changed, False otherwise.
        """
        return not np.array_equal(
            self._modal_base, self._aoClient.aoSystem.sysConf.gen.M2CMatrix
        )

    def _get_skip_counter(self) -> int:
        """
        Get the current frame counter from the AO client

        Returns
        -------
        total_skipped_frames : int
            Current total skipped frames counter value
        """
        cc = self._aoClient.getCounters()
        values = []
        keys = ["skipByCommand", "skipByDeltaCommand", "skipByForce"]
        for key in keys:
            values.append(getattr(cc, key))
        total_skipped_frames = sum(values)
        return total_skipped_frames

    def _get_clip_counter(self) -> int:
        """
        Get the current clip counter from the AO client

        Returns
        -------
        total_clipped_frames : int
            Current total clipped frames counter value
        """
        cc = self._aoClient.getCounters()
        values = []
        keys = ["clipByCommand", "clipByDeltaCommand", "clipByForce"]
        for key in keys:
            values.append(getattr(cc, key))
        total_clipped_frames = sum(values)
        return total_clipped_frames

    def _init_n_actuators(self) -> int:
        """
        Function which reads the number of actuators of the DM from a configuration
        file.

        Returns
        -------
        nact: int
            number of actuators of the DM.
        """
        return

    # DP
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

    def _resolve_tracknum(self, tracknum: str | None) -> str | None:
        """
        Function which resolves the tn.
        """
        from opticalib.ground.osutils import is_tn

        if not tracknum is None:
            if tracknum.lower() == "default":
                try:
                    from Microgate.adopt.AOBaseConfig import baseConfig  # type: ignore

                    tn = baseConfig.SYS_CONF_TN
                except ImportError:
                    tn = None

            elif is_tn(tracknum):
                tn = tracknum

            else:
                tn = None

        else:
            tn = None

        return tn

    # def _enumerate_devices(self):
    #     """
    #     Function which enumerates the connected devices.
    #     """
    #     self._aoClient.aoSystem.aoSubSystem0.deviceEnum()
