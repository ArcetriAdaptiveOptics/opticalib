"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025
- Matteo Menessini

Description
-----------

"""

import numpy as _np
from os.path import join as _j
from opticalib import typings as _ot
from opticalib.ground import osutils as _osu
from scipy.interpolate import griddata as _gd
from opticalib.core.root import folders as _fn


class PupilCalibrator:
    """
    Class to calibrate a DM given a pupil diofferent from that of the calibration
    data loaded.
    """

    def __init__(self, tn: str, dm: _ot.DeformableMirrorDevice) -> None:
        """The Initiator"""
        self._dm = dm
        self._tn = tn
        self._iffCube: _ot.CubeData | None = None
        self.IFM: _ot.MatrixLike | None = None
        self._loadIFMatrix()

    @property
    def dmCoords(self) -> _ot.MatrixLike:
        """
        Returns the actuator coordinates of the DM.
        """
        return self._dm.actCoords

    def act_coordinates_tranformation(
        self, dm: _ot.DeformableMirrorDevice, img: _ot.Optional[_ot.ImageData] = None
    ) -> _ot.MatrixLike:
        # Get the dm's actuator coordinates, confronts them with the image coordinates
        # and returns the transformation matrix
        ## Sudo code here
        #
        ...

    def _getTranformationMatrix(self) -> _ot.MatrixLike: ...

    def _getCurrentActCoords(self) -> _ot.MatrixLike: ...

    def _getCalibrationActCoords(self): ...

    def maskTransform(self, mask: _ot.ImageData, geometry) -> _ot.ImageData:
        """
        Transforms the given mask to the given geometry

        Parameters
        ----------
        mask : ImageData
            The mask to be transformed
        geometry : Geometry
            The geometry to which the mask should be transformed

        Returns
        -------
        transformed_mask : ImageData
            The transformed mask in the given geometry
        """
        ...

    def remapIff(self, mask: _ot.ImageData, geometry) -> _ot.MatrixLike:
        """
        Fits the IFFs to the given mask and geometry

        Parameters
        ----------
        mask : ImageData
            The mask to be used for remapping the IFFs
        geometry : Geometry
            The geometry to which the IFFs should be remapped

        Returns
        -------
        remapped_IFF : MatrixLike
            The remapped influence functions matrix
        """
        IFM = self.IFM.copy()

    def fitShape2Command(
        self,
        target_shape: _ot.ImageData,
        mask: _ot.ImageData,
        remapped_IFF: _ot.MatrixLike,
    ) -> _ot.ArrayLike:
        """
        Computes the command to obtain target_shape on the input mask using the IFF

        Parameters
        ----------
        target_shape : ImageData
            The shape to be commanded to the mirror
        mask : ImageData of booleans
            The mask of the commanded shape
        remapped_IFF : MatrixLike
            The masked influence functions matrix

        Returns
        -------
        raw_cmd : ArrayLike
            Vector of actuator commands.
        """

        if _ot.isinstance_(target_shape, _ot.ImageData):
            mask = _np.logical_or(mask, target_shape.mask)
        masked_shape = target_shape[~mask]
        raw_cmd = remapped_IFF @ masked_shape
        return raw_cmd

    def slaveCoords(
        self, raw_cmd: _ot.ArrayLike, slave_ids: list[int], slaving_method: str = "zero"
    ) -> _ot.ArrayLike:
        """
        Computes the command to obtain target_shape
        on the input mask using the IFF

        Parameters
        ----------
        raw_cmd : ArrayLike
            Vector of actuator commands.
        slave_ids : list
            The list of slave actuator ids.
        slaving_method : str
            String for the slaving method to use:
            - 'spline' : thin plate spline interpolation
            - 'nearest' : nearest grid interpolation
            - 'zero' : set slaves to zero
            Default is 'zero'

        Returns
        -------
        cmd : ArrayLike
            Slaved actuator commands.
        """

        cmd = raw_cmd.copy()
        coords = self._dm.actCoords.copy()
        ids = _np.arange(len(cmd))
        is_master = _np.ones_like(cmd, dtype=bool)
        is_master[slave_ids] = False
        master_ids = ids[is_master]
        master_coords = coords[:, master_ids]
        match slaving_method:
            case "spline":
                ...
                #tps = (alpha=0.0)
                #tps.fit(master_coords.T, cmd[master_ids])
                #cmd = tps.transform(coords.T)
            case "nearest":
                cmd = _gd(
                    master_coords,
                    cmd[master_ids],
                    (coords[0], coords[1]),
                    method="nearest",
                )
            case "zero":
                cmd[slave_ids] *= 0
            case _:
                raise KeyError(
                    f"{slaving_method} is not an available slaving method. Available methods are: 'interp', 'nearest', 'zero'"
                )
        return cmd

    def _loadIFMatrix(self) -> None:
        """
        Loads the calibration Cube (IffCube) and computes the interaction matrix (IFM).
        """
        self._iffCube = _osu.load_fits(
            _j(_fn.INTMAT_ROOT_FOLDER, self._tn, "IMCube.fits")
        )
        from opticalib.ground.computerec import ComputeReconstructor

        rec = ComputeReconstructor(self._iffCube)
        self.IFM = rec._intMat.copy()
        del rec
