import os
import numpy as np

from opticalib.core import _types as _t
from opticalib.core.exceptions import CommandError
from ._API import *
from opticalib.ground import osutils as osu
from opticalib.core.root import folders as fp
from opticalib.ground.logger import SystemLogger as _SL


class PetalMirror(BaseFakePTL):
    """
    Simulator for the petal deformable mirror.
    """

    def __init__(self, **kwargs: dict[str, _t.Any]):
        self._live = False
        self._logger = _SL(__class__)
        super().__init__(**kwargs)
        self.is_segmented = True
        self.n_segments = 6
        self.n_acts_per_segment = 3
        self.mirrorModes = None

    @property
    def slave_ids(self) -> _t.ArrayLike:
        """
        Returns the list of slave actuator IDs for the deformable mirror.

        Returns
        -------
        numpy.ndarray
            List of slave actuator IDs.
        """
        return self._slaveIds

    @property
    def border_ids(self) -> _t.ArrayLike:
        """
        Returns the list of border actuator IDs for the deformable mirror.

        Returns
        -------
        numpy.ndarray
            List of border actuator IDs.
        """
        return self._borderIds

    def get_shape(self):
        """
        Returns the current amplitudes commanded to the dm's actuators.

        Returns
        -------
        numpy.ndarray
            Current amplitudes commanded to the dm's actuators.
        """
        return self._n2modes(self._act_pos.copy()) / np.tile(self._unit_calib, 6)


class AlpaoDm(BaseFakeAlpao):

    def __init__(self, n_acts: int, force_recompute: bool = False):
        self._logger = _SL(__class__)
        super(AlpaoDm, self).__init__(n_acts, force_recompute=force_recompute)
        self._idx = np.where(self._mask == 0)
        self._produce_random_shape()
        self.is_segmented = False

    @property
    def slave_ids(self) -> _t.ArrayLike:
        """
        Returns the list of slave actuator IDs for the deformable mirror.

        Returns
        -------
        numpy.ndarray
            List of slave actuator IDs.
        """
        return self._slaveIds

    @property
    def border_ids(self) -> _t.ArrayLike:
        """
        Returns the list of border actuator IDs for the deformable mirror.

        Returns
        -------
        numpy.ndarray
            List of border actuator IDs.
        """
        return self._borderIds

    def _mirror_command(
        self, cmd: _t.ArrayLike, differential: bool, **kwargs: dict[str, _t.Any]
    ):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        cmd : numpy.ndarray
            Actuators command to be applied to the deformable mirror.
        differential : bool
            If True, process the command as a differential command.
        modal : bool
            If True, process the command in modal (Zernike) space.

        Returns
        -------
        numpy.ndarray
            Processed shape based on the command.
        """
        scaled_cmd = cmd * 1e-5  # more realistic command
        modal = kwargs.get("modal", False)
        if modal:
            mode_img = np.dot(self.ZM, scaled_cmd)
            scaled_cmd = np.dot(mode_img, self.RM)
        cmd_amp = scaled_cmd
        super()._base_mirror_command(cmd_amp, differential=differential)

    def _wavefront(self, **kwargs: dict[str, _t.Any]) -> np.array:
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
        wf : numpy.ndarray
            Phase map of the interferometer.
        """
        zernike = kwargs.get("zernike", None)
        surf = kwargs.get("surf", True)
        noisy = kwargs.get("noisy", False)
        img = np.ma.masked_array(self._shape, mask=self._mask)
        if zernike is not None:
            img = self._zern.remove_zernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi) if noisy else 0
            wf = np.sin(2 * np.pi / Ilambda * img + phi)
            A = np.std(img) / np.std(wf)
            wf *= A
            img = wf
        return img

    def _produce_random_shape(self):
        """
        Produces a random shape for the deformable mirror initialization,
        by using a linear combination of Tip/Tilt and focus.

        Returns
        -------
        numpy.ndarray
            Random shape for the deformable mirror.
        """
        try:
            self._logger.info(f"Loading base shape for {self._name} from file")
            shape = osu.load_fits(
                os.path.join(fp.simulated_dm_path(self._name), f"baseShape.fits")
            )
            self._shape = np.ma.masked_array(shape)
        except FileNotFoundError:
            self._logger.info(
                f"No base shape file found for {self._name}, generating random shape"
            )
            mat = np.eye(self.n_acts)
            tx = mat[0]
            ty = mat[1]
            f = mat[3]
            rand = np.random.uniform
            cmd = (
                rand(0.05, 0.005) * ty
                + rand(0.05, 0.005) * tx
                + rand(0.005, 0.0005) * f
            )
            self.set_shape(cmd, modal=True)
            self._logger.info(f"Saving generated base shape for {self._name} to file")
            osu.save_fits(
                os.path.join(fp.simulated_dm_path(self._name), f"baseShape.fits"),
                self._shape,
            )
            self._act_pos = np.zeros(self.n_acts)

    def __repr__(self) -> str:
        return f"{__class__.__name__}(n_acts={self.n_acts})"


class DP(BaseFakeDp):
    """
    Simulator for the AdOptica Deformable Prototype (DP).
    """

    def __init__(self, **kwargs: dict[str, _t.Any]):
        """
        The constructor for the DPSimulator class.
        """
        self._live = False
        self._logger = _SL(__class__)
        super().__init__(**kwargs)
        self.is_segmented = True
        self.n_segments = 2
        self.n_acts_per_segment = 111

    def get_shape(self):
        """
        Returns the current amplitudes commanded to the dm's actuators.

        Returns
        -------
        numpy.ndarray
            Current amplitudes commanded to the dm's actuators.
        """
        cmd = np.concatenate((self._act_pos[0], self._act_pos[1]))
        return cmd

    def __repr__(self) -> str:
        return f"{self._name}(n_segments={self.n_segments}, n_acts_per_segment={self.n_acts_per_segment})"


class M4AU(BaseFakeM4):

    def __init__(self, **kwargs: dict[str, _t.Any]):
        self._logger = _SL(__class__)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"{self._name}(n_segments={self.n_segments}, n_acts_per_segment={self.n_acts_per_segment})"

    def _mirror_command(self, cmd: _t.ArrayLike, diff: bool, modal: bool):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        cmd : numpy.ndarray
            Command to be processed by the deformable mirror.

        diff : bool
            If True, process the command differentially.

        Returns
        -------
        numpy.ndarray
            Processed shape based on the command.
        """
        if not len(cmd) == self.n_acts:
            raise CommandError(
                f"Command length {len(cmd)} does not match the number of actuators {self.n_acts}."
            )

        nsag = self.n_acts_per_segment

        for seg in range(self.n_segments):
            idx = self._idx[seg]

            if modal and self.zm is not None:
                mode_img = np.dot(self.zm[seg], cmd[seg * nsag : (seg + 1) * nsag])
                cmd_amp = np.dot(mode_img, self.rm[seg])
            else:
                cmd_amp = cmd[seg * nsag : (seg + 1) * nsag]

            if not diff:
                cmd_amp = cmd_amp - self._act_pos[seg * nsag : (seg + 1) * nsag]

            self._shape[idx] += np.dot(cmd_amp, self.im[seg])
            self._act_pos[seg * nsag : (seg + 1) * nsag] += cmd_amp
