import os
import xupy as xp
import numpy as _np
from ... import typings as _ot
from .. import factory_functions as ff
from ...core import root as _root
from ...ground.roi import roiGenerator
from ...ground import osutils as osu
from scipy.interpolate import RBFInterpolator

class BaseFakePTL():
    """
    Base class for petal mirror simulators.
    """

    def __init__(self, shape: tuple[int, int] = (1200,1200), outer_radius: int = 450, inner_radius: int | None = None):
        self._name = "FakePetalDM"
        self._outer_radius = outer_radius
        self._inner_radius = inner_radius
        self._mask, self._coords = ff.getPetalmirrorMaskAndCoords(
            shape,
            outer_radius,
            inner_radius
        )
        self._rois = roiGenerator(
            _np.ma.masked_array(
                self._mask*0, mask=self._mask
            )
        )
        _ = self._rois.pop(3)
        self._idxs = self._get_segs_idxs()
        self.nActs = self._coords.shape[0]
        self._actPos = _np.zeros(self.nActs)
        self._get_matrices()
        self._shape = _np.ma.masked_array(
            self._mask * 0, mask=self._mask, dtype=xp.float32
        )

    def _wavefront(self, **kwargs: dict[str, _ot.Any]) -> _ot.ImageData:
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
        img = _np.ma.masked_array(self._shape, mask=self._mask)
        if zernike is not None:
            img = self._zern.removeZernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = _np.random.uniform(-0.25 * _np.pi, 0.25 * _np.pi) if noisy else 0
            wf = _np.sin(2 * _np.pi / Ilambda * img.copy() + phi)
            A = _np.std(img) / _np.std(wf)
            wf *= A
            img = wf.copy()
            del wf
        dx, dy = 650 - img.shape[0], 650 - img.shape[1]
        if dx > 0 or dy > 0:
            pimg = _np.pad(
                img.data,
                ((dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)),
                mode="constant",
                constant_values=0,
            )
            pmask = _np.pad(
                img.mask,
                ((dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)),
                mode="constant",
                constant_values=1,
            )
            img = _np.ma.masked_array(pimg, mask=pmask)
        return img

        
    def _mirror_command(self, cmd: _ot.ArrayLike, diff: bool):
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
        K = _np.array(
            [
                [1,1,1],
                [0,1,-1],
                [0,-1,1],
            ]
        )

        for jj, idx in enumerate(self._idxs):
            cmd_amp = cmd[jj*3:(jj+1)*3] @ K
            if diff:
                cmd_amp -= - self._actPos[jj*3:(jj+1)*3]
            self._shape[idx] += _np.dot(cmd_amp, self.IM[jj])
            self._actPos[jj*3:(jj+1)*3] += cmd_amp
    
    def _get_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(_root.SIM_DATA_FILE(self._name, "IF")):
            print(
                f"First time simulating {self._name}.\nGenerating influence functions..."
            )
            self._simulate_iff()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = osu.load_fits(_root.SIM_DATA_FILE(self._name, "IF"))
        self._get_IM_RM()
        self._get_ZM()
        
    def _simulate_iff(self):
        """
        """
        iff_cubes = []

        for jj, roi in enumerate(self._rois):
            
            print(f"Simulating IFF for ROI {jj+1}/{len(self._rois)}")
            act_pix_coords = self._coords[jj*3:(jj+1)*3, :]  # Shape (n_seg_acts, 2)
            n_acts = act_pix_coords.shape[0]
            
            # Create pixel grid coordinates.
            X, Y = roi.shape
            pix_coords = _np.zeros((X * Y, 2))
            pix_coords[:, 0] = _np.tile(_np.arange(Y), X)
            pix_coords[:, 1] = _np.repeat(_np.arange(X), Y)
            img_cube = _np.zeros((X, Y, 3))
            
            # For each actuator, compute the influence function with a TPS interpolation.
            for k in range(n_acts):
                print(f"{k+1}/{n_acts}", end="\r", flush=True)
                # Create a command vector with a single nonzero element (ZONAL IFF).
                act_data = _np.zeros(n_acts)
                act_data[k] = 1

                rbf = RBFInterpolator(
                    act_pix_coords,  # Shape (n_acts, 2)
                    act_data,  # Shape (n_acts,)
                    kernel="thin_plate_spline",  # TPS
                    smoothing=0.0,  # No smoothing
                    degree=1,  # Polynomial degree for TPS
                )
                flat_img = rbf(pix_coords)

                img_cube[:, :, k] = flat_img.reshape((X, Y))
            
            cube_mask = _np.tile(roi, n_acts).reshape(img_cube.shape, order="F")
            cube = _np.ma.masked_array(img_cube, mask=cube_mask)
            iff_cubes.append(cube)

        iff_cubes = _np.ma.stack(iff_cubes, axis=3)
        fits_file = _root.SIM_DATA_FILE(self._name, "IF")
        osu.save_fits(fits_file, iff_cubes)
        self._iffCube = iff_cubes

    def _get_IM_RM(self):
        """
        """
        
        imfile = _root.SIM_DATA_FILE(self._name, "IM")
        rmfile = _root.SIM_DATA_FILE(self._name, "RM")
        if not all([os.path.exists(imfile), os.path.exists(rmfile)]):
            print("Computing interaction matrix...")
            ims = []
            rms = []
            for mask, s in zip(self._rois, range(6)):
                im = xp.array(
                    [
                        (self._iffCube[:, :, i, s].data)[mask == 0]
                        for i in range(self._iffCube.shape[2])
                    ],
                    dtype=xp.float,
                )
                ims.append(im)
                rms.append(xp.asnumpy(xp.linalg.pinv(im)))
            self.IM = [xp.asnumpy(ifm) for ifm in ims]
            osu.save_h5(
                {f's{i}': self.IM[i] for i in range(6)},
                imfile)
            print("Computing reconstruction matrix...")
            self.RM = [rm for rm in rms]
            osu.save_h5(
                {f's{i}': self.RM[i] for i in range(6)},
                rmfile)
        else:
            print(f"Loaded interaction matrix.")
            ims = osu.load_h5(imfile)
            self.IM = [ims[f's{i}'] for i in range(6)]
            print(f"Loaded reconstruction matrix.")
            rms = osu.load_h5(rmfile)
            self.RM = [rms[f's{i}'] for i in range(6)]
    
    def _get_ZM(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(_root.SIM_DATA_FILE(self._name, "ZM")):
            n_zern = 3
            print("Computing Zernike matrix...")
            from ..factory_functions import generateZernikeMatrix

            zms = []
            for mask in self._rois:
                zm = generateZernikeMatrix(n_zern, mask)
                zms.append(zm)
            osu.save_h5(
                {f's{i}': zms[i] for i in range(6)},
                _root.SIM_DATA_FILE(self._name, "ZM")
            )
            self.ZM = zms
        else:
            print(f"Loaded Zernike matrix.")
            zms = osu.load_h5(_root.SIM_DATA_FILE(self._name, "ZM"))
            self.ZM = [zms[f's{i}'] for i in range(6)]
            
    def _get_segs_idxs(self):
        """
        Get the indices of the segments in the mask.
        """
        idxs = []
        for roi in self._rois:
            idx = _np.where(roi == 0)
            idxs.append(idx)
        return idxs