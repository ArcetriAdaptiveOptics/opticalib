import os
import xupy as xp
import numpy as np
from scipy.interpolate import RBFInterpolator

# from scipy.interpolate import Rbf
from opticalib import folders as fp
from opticalib import typings as _t
from opticalib.ground import geometry as geo
from opticalib import load_fits as lf, save_fits as sf
from opticalib.core.read_config import getDmIffConfig as _dmc
from opticalib.ground.modal_decomposer import ZernikeFitter as _ZF

join = os.path.join


class BaseFakeDp:

    def __init__(self):
        """The constuctor"""
        dmc = _dmc()
        self._name = "AdOpticaDP"
        self._rootDataDir = join(os.path.dirname(__file__), "AdOpticaData")
        self.mirrorModes = lf(os.path.join(self._rootDataDir, "dp_cmdmat.fits"))
        self.ff = lf(os.path.join(self._rootDataDir, "dp_ffwd.fits"))
        self.nActs = self.mirrorModes.shape[0]
        self._createDpMaskAndCoords()
        self.cmdHistory = None
        self._shape = np.ma.masked_array(self._mask * 0, mask=self._mask, dtype=float)
        self._idx0, self._idx1 = self._get_segments_idx()
        self._load_matrices()
        self._zern = _ZF(self._mask)
        self._actPos = [np.zeros(self.nActs // 2), np.zeros(self.nActs // 2)]
        self._ccalcurve = self._getCapsensCalibration()
        self._biasCmd = self._getBiasCmd()
        self.set_shape(np.zeros(self.nActs))  # initialize to flat + offset
        self._slaveIds = dmc.get("slaveIds", [])
        self._borderIds = dmc.get("borderIds", [])

    @property
    def slaveIds(self) -> list[int]:
        """List of indices of the slaved actuators."""
        return self._slaveIds

    @property
    def borderIds(self) -> list[int]:
        """List of indices of the border actuators."""
        return self._borderIds

    @property
    def actCoords(self) -> _t.ArrayLike:
        """Actuator coordinates in pixels."""
        return self._coords.copy()

    def _wavefront(self, **kwargs: dict[str, _t.Any]) -> _t.ImageData:
        """
        Current shape of the mirror's surface. Only used for the interferometer's
        live viewer (see `interferometer.py`).

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for customization.
            - zernike : int ,
                Zernike mode to be removed from the wavefront.
            - surf : bool ,
                If True, the shape is returned instead of
                the wavefront.
            - noisy : bool ,
                If True, adds noise to the wavefront.

        Returns
        -------
        wf : np.array
            Phase map of the interferometer.
        """
        zernike = kwargs.get("zernike", None)
        surf = kwargs.get("surf", True)
        noisy = kwargs.get("noisy", False)
        img = np.ma.masked_array(self._shape, mask=self._mask)
        if zernike is not None:
            img = self._zern.removeZernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi) if noisy else 0
            wf = np.sin(2 * np.pi / Ilambda * img.copy() + phi)
            A = np.std(img) / np.std(wf)
            wf *= A
            img = wf.copy()
            del wf
        dx, dy = 650 - img.shape[0], 650 - img.shape[1]
        if dx > 0 or dy > 0:
            pimg = np.pad(
                img.data,
                ((dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)),
                mode="constant",
                constant_values=0,
            )
            pmask = np.pad(
                img.mask,
                ((dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)),
                mode="constant",
                constant_values=1,
            )
            img = np.ma.masked_array(pimg, mask=pmask)
        return geo.rotate_image(img, angle_deg=-70)

    def _mirror_command(self, cmd: _t.ArrayLike, diff: bool, modal: bool):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        cmd : np.array
            Command to be processed by the deformable mirror.

        diff : bool
            If True, process the command differentially.

        Returns
        -------
        np.array
            Processed shape based on the command.
        """
        # cmd = self._applyCSCalibration(cmd)
        cmd0 = cmd[: self.nActs // 2].copy()
        cmd1 = cmd[self.nActs // 2 :].copy()
        tomove = [0, 1]
        cmds = [cmd0, cmd1]
        idxs = [self._idx0, self._idx1]
        for s, cmx, idx in zip(tomove, cmds, idxs):
            if modal:
                mode_img = np.dot(self.ZM[s], cmx)
                cmx = np.dot(mode_img, self.RM[s])
            cmd_amp = cmx
            if not diff:
                cmd_amp = cmx - self._actPos[s]
            self._shape[idx] += np.dot(cmd_amp, self.IM[s])
            self._actPos[s] += cmd_amp

    def _load_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(fp.SIM_DATA_FILE(self._name, "IF")):
            print(
                f"First time simulating {self._name}.\nGenerating influence functions..."
            )
            self._simulateDP()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = lf(fp.SIM_DATA_FILE(self._name, "IF"))
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(fp.SIM_DATA_FILE(self._name, "ZM")):
            n_zern = self.nActs // 2
            print("Computing Zernike matrix...")
            from ..factory_functions import generateZernikeMatrix

            zms = []
            for mask in [self._ms0, self._ms1]:
                zm = generateZernikeMatrix(n_zern, mask)
                zms.append(zm)
            ZM = np.dstack(zms)
            sf(fp.SIM_DATA_FILE(self._name, "ZM"), ZM)
            self.ZM = zms
        else:
            print(f"Loaded Zernike matrix.")
            zms = lf(fp.SIM_DATA_FILE(self._name, "ZM"))
            self.ZM = [zms[:, :, i] for i in range(zms.shape[-1])]

    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        imfile = fp.SIM_DATA_FILE(self._name, "IM")
        rmfile = fp.SIM_DATA_FILE(self._name, "RM")
        if not all([os.path.exists(imfile), os.path.exists(rmfile)]):
            print("Computing interaction matrix...")
            ims = []
            rms = []
            for mask, s in zip([self._ms0, self._ms1], [0, 1]):
                im = xp.array(
                    [
                        (self._iffCube[:, :, i, s].data)[mask == 0]
                        for i in range(self._iffCube.shape[2])
                    ],
                    dtype=xp.float,
                )
                ims.append(im)
                rms.append(xp.asnumpy(xp.linalg.pinv(im)))
            im = xp.dstack(ims)
            self.IM = [xp.asnumpy(ifm) for ifm in ims]
            sf(imfile, im)
            print("Computing reconstruction matrix...")
            self.RM = [rm for rm in rms]
            sf(rmfile, np.dstack(rms))
        else:
            print(f"Loaded interaction matrix.")
            ims = lf(imfile)
            self.IM = [ims[:, :, i] for i in range(ims.shape[-1])]
            print(f"Loaded reconstruction matrix.")
            rms = lf(rmfile)
            self.RM = [rms[:, :, i] for i in range(rms.shape[-1])]

    def _getCapsensCalibration(self):
        """
        Loads the capacitive sensors calibration data.
        """
        calcurve = np.random.uniform(0.9, 1.1, size=self.nActs)
        return calcurve

    def _getBiasCmd(self):
        """
        Loads the bias command for the DP.
        """
        biascmd = np.random.uniform(75e-9, 85e-9, size=self.nActs)
        return biascmd

    def _applyCSCalibration(self, cmd: _t.ArrayLike):
        """
        Applies the capacitive sensors calibration to the current command.
        """
        ncmd = cmd * self._ccalcurve
        if all(cmd == 0):
            ncmd += self._biasCmd
        return ncmd

    def _simulateDP(self):
        """
        Simulates the influence function of the DP by TPS interpolation
        """
        # get Segment0 mask and coordinates (scaled)
        s0x, s0y = self._coords[: self.nActs // 2, :].T  # segment 0 (upper)
        # self._ms0 = dp_mask[: dp_mask.shape[0] // 2, :] # segment 0 mask
        s0y -= self._ms0.shape[0]
        s0acts = np.stack((s0x, s0y)).T

        # get Segment1 mask and coordinates
        s1x, s1y = self._coords[self.nActs // 2 :, :].T  # segment 1 (lower)
        # self._ms1 = dp_mask[dp_mask.shape[0] // 2 :, :] # segment 1 mask
        s1acts = np.stack((s1x, s1y)).T

        # act_px_coords = self._coords.copy()
        iffcubes = []
        segacts = 111
        for act_px_coords, mask, sid in zip(
            [s0acts, s1acts], [self._ms0, self._ms1], [0, 1]
        ):
            print(f"Segment {sid}", end="\n")
            X, Y = mask.shape
            # Create pixel grid coordinates.
            pix_coords = np.zeros((X * Y, 2))
            pix_coords[:, 0] = np.tile(np.arange(Y), X)
            pix_coords[:, 1] = np.repeat(np.arange(X), Y)
            img_cube = np.zeros((X, Y, segacts))
            # For each actuator, compute the influence function with a TPS interpolation.
            for k in range(segacts):
                print(f"{k+1}/{segacts}", end="\r", flush=True)
                # Create a command vector with a single nonzero element.
                act_data = np.zeros(segacts)
                act_data[k] = 1.0

                rbf = RBFInterpolator(
                    act_px_coords,  # Shape (n_acts, 2)
                    act_data,  # Shape (n_acts,)
                    kernel="thin_plate_spline",  # TPS
                    smoothing=0.0,  # No smoothing
                    degree=1,  # Polynomial degree for TPS
                )
                flat_img = rbf(pix_coords)

                img_cube[:, :, k] = flat_img.reshape((X, Y))
            # Create a cube mask that tiles the local mirror mask for each actuator.
            cube_mask = np.tile(mask, segacts).reshape(img_cube.shape, order="F")
            cube = np.ma.masked_array(img_cube, mask=cube_mask)
            iffcubes.append(cube)
            # Save the cube to a FITS file.
        iffcubes = np.ma.stack(iffcubes, axis=3)
        fits_file = fp.SIM_DATA_FILE(self._name, "IF")
        sf(fits_file, iffcubes)
        self._iffCube = iffcubes

    def _createDpMaskAndCoords(self) -> tuple[_t.MaskData, _t.ArrayLike]:
        """
        Creates the mask and the actuator pixel coordinates for the DP
        """
        dp_coords = lf(join(self._rootDataDir, "dp_coords.fits"))

        # Get DP's shape vertex coordinates (only 1 segment)

        s0x, s0y = (
            dp_coords[dp_coords[:, 1] > dp_coords[:, 1].max() / 2].T * 1000
        )  # upper segment
        s1x, s1y = (
            dp_coords[dp_coords[:, 1] < dp_coords[:, 1].max() / 2].T * 1000
        )  # lower segment

        y0, y1, y2, y3 = s0y.min(), s0y.max(), s0y.max(), s0y.min()
        x0, x1, x2, x3 = (
            s0x[s0y == y0].min(),
            s0x[s0y == y1].min(),
            s0x[s0y == y2].max(),
            s0x[s0y == y3].max(),
        )

        # vertex coordinates of the upper segment
        cols = np.array([x0, x1, x2, x3])
        rows = np.array([y0, y1, y2, y3])

        ylm = s1y.max()
        gap = np.abs((ylm - y0) // 2).astype(np.int8)  # in px

        # rescale to mm/px
        cols = cols - cols.min()
        rows = rows - rows.min()

        sx = int(np.ceil(np.max(cols))) + 1
        sy = int(np.ceil(np.max(rows))) + 1

        # Creates the mask of the single segment
        mm = geo.draw_polygonal_mask((sy, sx), np.stack((cols, rows)).T)
        mm = np.pad(mm, ((gap, 0), (0, 0)), mode="constant", constant_values=1)

        # Creates the full DP mask by mirroring the single segment
        mask_dp = np.zeros((mm.shape[0] * 2, mm.shape[1]))
        mask_dp[mm.shape[0] :, :] = mm
        mask_dp += np.flipud(mask_dp)  # flipping up-down
        mask_dp = mask_dp.astype(bool)

        # padding the mask to avoid tangent frame
        final_mask = np.pad(mask_dp, 5, mode="constant", constant_values=1)

        # Reorganize actuator coordinates in shells
        final_coords = np.empty_like(dp_coords)
        final_coords[: self.nActs // 2, :] = (
            np.array([s0x, s0y]).T + 5
        )  # matching padding
        final_coords[self.nActs // 2 :, :] = (
            np.array([s1x, s1y]).T + 5
        )  # matching padding

        self._coords = final_coords.copy()
        self._mask = final_mask.copy()
        self._ms0 = final_mask[: final_mask.shape[0] // 2, :]
        self._ms1 = final_mask[final_mask.shape[0] // 2 :, :]
        # return final_mask, final_coords

    def _get_segments_idx(self):
        """
        Get the indices of the two segments in the DP mask
        """
        mid_row = self._mask.shape[0] // 2
        idx0 = np.where(self._mask[:mid_row, :] == 0)
        idx1 = np.where(self._mask[mid_row:, :] == 0)
        idx1 = (idx1[0] + mid_row, idx1[1])  # adjust row indices for segment 1
        return idx0, idx1
