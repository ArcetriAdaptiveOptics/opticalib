import os
import xupy as xp
import numpy as np
from ..factory import *
from ...core import root as _root
from ...core import config as _rc
from .._API.base_simulated_dm import BaseSimulatedDM
from opticalib.ground import osutils as osu
from rbfgpu import RBFInterpolator


class BaseFakeAlpao(BaseSimulatedDM):
    """
    Base class for deformable mirrors.
    """

    def __init__(self, n_acts: int, force_recompute: bool = False):
        """
        Initializes the base deformable mirror with the number of actuators.
        """
        self._name = f"AlpaoDM{n_acts}"
        self.mirrorModes = None
        self.n_acts = n_acts
        self._pxScale = pixel_scale(self.n_acts)
        self._coords, self._mask = get_alpao_coords_mask(self.n_acts)
        super().__init__()
        self.scaled_coords = self._scale_act_coords()
        self._iffCube = None
        self.IM = None
        self.ZM = None
        self.RM = None
        self._load_matrices(force_recompute=force_recompute)
        try:
            dmc = _rc.get_device_config("DEFORMABLE.MIRRORS", self._name)
            self._slaveIds = dmc.get("slave_ids", [])
            self._borderIds = dmc.get("border_ids", [])
        except Exception:
            self._slaveIds = []
            self._borderIds = []

    @property
    def slave_ids(self):
        return self._slaveIds

    @property
    def border_ids(self):
        return self._borderIds

    def _load_matrices(self, force_recompute: bool = False):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if (
            not os.path.exists(_root.sim_data_file(self._name, "IF") + ".fits")
            or force_recompute
        ):
            print(
                f"First time simulating DM {self.n_acts}. Generating influence functions..."
            )
            self._simulate_iff()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(
                osu.load_fits(_root.sim_data_file(self._name, "IF"))
            )
        self._create_int_and_rec_matrices(force_recompute=force_recompute)
        self._create_zernike_matrix(force_recompute=force_recompute)

    def _create_zernike_matrix(self, force_recompute: bool = False):
        """
        Create the Zernike matrix for the DM.
        """
        if (
            not os.path.exists(_root.sim_data_file(self._name, "ZM") + ".fits")
            or force_recompute
        ):
            n_zern = self.n_acts
            print("Computing Zernike matrix...")
            self.ZM = xp.asnumpy(generate_zernike_matrix(n_zern, self._mask))
            osu.save_fits(_root.sim_data_file(self._name, "ZM"), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = osu.load_fits(_root.sim_data_file(self._name, "ZM"))

    def _create_int_and_rec_matrices(self, force_recompute: bool = False):
        """
        Create the interaction matrices for the DM.
        """
        if (
            not all(
                [
                    os.path.exists(_root.sim_data_file(self._name, "IM") + ".fits"),
                    os.path.exists(_root.sim_data_file(self._name, "RM") + ".fits"),
                ]
            )
            or force_recompute
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
            osu.save_fits(_root.sim_data_file(self._name, "IM"), self.IM)
            osu.save_fits(_root.sim_data_file(self._name, "RM"), self.RM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = osu.load_fits(_root.sim_data_file(self._name, "IM"))
            print(f"Loaded reconstruction matrix.")
            self.RM = osu.load_fits(_root.sim_data_file(self._name, "RM"))

    def _simulate_iff(self):
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
            A masked cube of influence functions with shape (height, width, n_acts).
        """
        # Get the number of actuators from the coordinates array.
        n_acts = self.n_acts
        max_x, max_y = self._mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((max_x * max_y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(max_x), max_y)
        pix_coords[:, 1] = np.tile(np.arange(max_y), max_x)
        act_pix_coords = self._scale_act_coords()
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
            flat_img = xp.asnumpy(rbf(pix_coords))

            img_cube[:, :, k] = flat_img.reshape((max_x, max_y))

        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self._mask, n_acts).reshape(img_cube.shape, order="F")
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = _root.sim_data_file(self._name, "IF")
        osu.save_fits(fits_file, cube)
        self._iffCube = cube

    def _scale_act_coords(self):
        """
        Scale actuator coordinates to the mirror's pixel grid while preserving
        their radial symmetry over the circular pupil.
        """
        radius = min(self._mask.shape) / 2.0
        return self.scale_coords_to_pupil(
            self._coords,
            mask=self._mask,
            radius=radius,
        )
