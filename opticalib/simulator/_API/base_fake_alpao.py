import os
import xupy as xp
import numpy as np
from ... import typings as _t
from ..factory_functions import *
from ...core import root as _root
from ...core.read_config import getDmIffConfig as _dmc
from abc import ABC, abstractmethod
from opticalib.ground import osutils as osu
from scipy.interpolate import RBFInterpolator


class BaseFakeAlpao(ABC):
    """
    Base class for deformable mirrors.
    """

    def __init__(self, nActs: int):
        """
        Initializes the base deformable mirror with the number of actuators.
        """
        self._name = f"AlpaoDM{nActs}"
        self.mirrorModes = None
        self.nActs = nActs
        self._pxScale = pixel_scale(self.nActs)
        self.actCoords, self._mask = getAlpaoCoordsMask(self.nActs)
        self._scaledActCoords = self._scaleActCoords()
        self._iffCube = None
        self.IM = None
        self.ZM = None
        self.RM = None
        self._load_matrices()
        dmc = _dmc()
        self._slaveIds = dmc.get("slaveIds", [])
        self._borderIds = dmc.get("borderIds", [])

    @property
    def slaveIds(self):
        return self._slaveIds

    @property
    def borderIds(self):
        return self._borderIds

    @abstractmethod
    def set_shape(self, command: _t.ArrayLike, differential: bool = False):
        """
        Applies the DM to a wavefront.

        Parameters
        ----------
        command : np.array
            Wavefront to which the DM will be applied.

        differential : bool
            If True, the command is the differential wavefront.

        Returns
        -------
        np.array
            Modified wavefront.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        """
        Returns the current shape of the DM.

        Returns
        -------
        np.array
            Current shape of the DM.
        """
        raise NotImplementedError

    @abstractmethod
    def uploadCmdHistory(self, timed_command_history: _t.MatrixLike):
        """
        Uploads a history of commands to the DM.

        Parameters
        ----------
        timed_command_history : _t.MatrixLike
            A 2D array where each column represents a command to be applied to the DM.
        """
        raise NotImplementedError

    @abstractmethod
    def runCmdHistory(self):
        """
        Executes the uploaded command history on the DM.
        """
        raise NotImplementedError

    def _load_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(_root.SIM_DATA_FILE(self._name, "IF")):
            print(
                f"First time simulating DM {self.nActs}. Generating influence functions..."
            )
            self._simulate_Zonal_Iff_Acquisition()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(
                osu.load_fits(_root.SIM_DATA_FILE(self._name, "IF"))
            )
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(_root.SIM_DATA_FILE(self._name, "ZM")):
            n_zern = self.nActs
            print("Computing Zernike matrix...")
            self.ZM = xp.asnumpy(generateZernikeMatrix(n_zern, self._mask))
            osu.save_fits(_root.SIM_DATA_FILE(self._name, "ZM"), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = osu.load_fits(_root.SIM_DATA_FILE(self._name, "ZM"))

    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        if not all(
            [
                os.path.exists(_root.SIM_DATA_FILE(self._name, "IM")),
                os.path.exists(_root.SIM_DATA_FILE(self._name, "RM")),
            ]
        ):
            print("Computing interaction matrix...")
            im = xp.array(
                [
                    (self._iffCube[:, :, i].data)[self._mask == 0]
                    for i in range(self._iffCube.shape[2])
                ]
            )
            self.IM = xp.asnumpy(im)
            print("Computing reconstruction matrix...")
            self.RM = xp.asnumpy(xp.linalg.pinv(im))
            osu.save_fits(_root.SIM_DATA_FILE(self._name, "IM"), self.IM)
            osu.save_fits(_root.SIM_DATA_FILE(self._name, "RM"), self.RM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = osu.load_fits(_root.SIM_DATA_FILE(self._name, "IM"))
            print(f"Loaded reconstruction matrix.")
            self.RM = osu.load_fits(_root.SIM_DATA_FILE(self._name, "RM"))

    def _simulate_Zonal_Iff_Acquisition(self):
        """
        Simulate the influence functions by imposing 'perfect' zonal commands.

        Parameters
        ----------
        amps : float or np.ndarray, optional
            Amplitude(s) for the actuator commands. If a single float is provided,
            it is applied to all actuators. Default is 1.0.

        Returns
        -------
        np.ma.MaskedArray
            A masked cube of influence functions with shape (height, width, nActs).
        """
        # Get the number of actuators from the coordinates array.
        n_acts = self.actCoords.shape[1]
        max_x, max_y = self._mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((max_x * max_y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(max_x), max_y)
        pix_coords[:, 1] = np.tile(np.arange(max_y), max_x)
        act_pix_coords = self._scaleActCoords()
        # Create Empty cube for IFF
        img_cube = np.zeros((max_x, max_y, n_acts))

        # For each actuator, compute the influence function with a TPS interpolation.
        for k in range(n_acts):
            print(f"{k+1}/{n_acts}", end="\r", flush=True)
            # Create a command vector with a single nonzero element (ZONAL IFF).
            act_data = np.zeros(n_acts)
            act_data[k] = 1

            rbf = RBFInterpolator(
                act_pix_coords,  # Shape (n_acts, 2)
                act_data,  # Shape (n_acts,)
                kernel="thin_plate_spline",  # TPS
                smoothing=0.0,  # No smoothing
                degree=1,  # Polynomial degree for TPS
            )
            flat_img = rbf(pix_coords)

            img_cube[:, :, k] = flat_img.reshape((max_x, max_y))

        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self._mask, n_acts).reshape(img_cube.shape, order="F")
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = _root.SIM_DATA_FILE(self._name, "IF")
        osu.save_fits(fits_file, cube)
        self._iffCube = cube

    def _scaleActCoords(self):
        """
        Scales the actuator coordinates to the mirror's pixel scale.
        """
        max_x, max_y = self._mask.shape
        if not self.actCoords.shape[1] == 2:
            act_coords = self.actCoords.T  # shape: (n_acts, 2)
        act_pix_coords = np.zeros((self.nActs, 2), dtype=int)
        act_pix_coords[:, 0] = (
            act_coords[:, 1] / np.max(act_coords[:, 1]) * max_x
        ).astype(int)
        act_pix_coords[:, 1] = (
            act_coords[:, 0] / np.max(act_coords[:, 0]) * max_y
        ).astype(int)
        return act_pix_coords
