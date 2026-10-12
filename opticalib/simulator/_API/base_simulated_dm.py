""" """

import os
import numpy as np
from time import sleep
from matplotlib.pyplot import pause
from opticalib import folders as fp
from opticalib.ground import osutils as osu
from opticalib.core import _types as _ot, config as _rc
from opticalib.dmutils.slaving import compute_slaved_command_matrix, compute_slave_cmd
from opticalib.ground.modal_decomposer import ZernikeFitter as _ZF
from opticalib.devices._API.base_devices import BaseDeformableMirror


def _apply_slaving(
    dm: _ot.FakeDeformableMirrorDevice,
    cmd: _ot.ArrayLike | _ot.MatrixLike,
    slave: bool | str,
) -> _ot.ArrayLike:
    """
    Apply slaving to command if requested.
    """
    if isinstance(slave, str):
        method = slave
    elif slave:
        if len(dm.slave_ids) == 0:
            return cmd
        method = "zero-force" if len(dm.border_ids) == 0 else "minimum-rms"
    else:
        return cmd
    if _ot.isinstance_(cmd, "MatrixLike"):
        return compute_slaved_command_matrix(dm, cmd, method=method)
    return compute_slave_cmd(dm, cmd, method=method)


class BaseSimulatedDM(BaseDeformableMirror):

    def __init__(self):
        """
        Initialize the base simulated deformable mirror (DM).
        """
        self._live = False
        the_mask = self._mask if self._mask is not None else self.pupil
        self.pupil = self._mask = the_mask  # fill the one missing
        if all(v is None for v in (self.pupil, self._mask)):
            raise ValueError("At least one of 'pupil' or '_mask' must be defined.")
        self._zern = _ZF(the_mask)
        self._shape = np.ma.masked_array(the_mask * 0, mask=the_mask, dtype=np.float32)
        self._act_pos = np.zeros(self.n_acts)
        self._cmd_history = None

    @property
    def act_coords(self) -> _ot.ArrayLike:
        """
        Get the current actuator coordinates of the simulated deformable mirror (DM).

        Returns
        -------
        act_pos : _ot.ArrayLike
            The current actuator coordinates of the DM.
        """
        return self._coords.copy()

    def set_shape(
        self,
        cmd: _ot.ArrayLike,
        differential: bool = False,
        *,
        slave: bool | str = True,
        **kwargs: dict[str, _ot.Any],
    ) -> None:
        """
        Send a command to the simulated deformable mirror (DM).

        Parameters
        ----------
        cmd : _ot.ArrayLike
            The command to be sent to the DM.
        differential : bool, optional
            Whether the command is a differential command (default is False).
        slave : bool | str, optional
            Whether to apply slaving to the command (default is True).
            It can be a string representing the slaving method to be used. Options
            are "zero-force" and "minimum-rms".
        **kwargs : dict[str,_ot.Any]
            Additional keyword arguments to be passed to the mirror command.
        """
        cmd = _apply_slaving(self, cmd, slave=slave)
        self._logger.info(f"Sending mirror command to {self._name}")
        self._mirror_command(cmd, differential=differential, **kwargs)
        if self._live:
            sleep(0.15)
            pause(0.05)

    def get_shape(self) -> _ot.ArrayLike:
        """
        Get the current shape of the simulated deformable mirror (DM).

        Returns
        -------
        cmd : _ot.ArrayLike
            The current shape of the DM.
        """
        return self._act_pos.copy()

    def upload_cmd_history(
        self, cmdhist: _ot.MatrixLike, slaving: bool | str = False
    ) -> None:
        """
        Upload the command history to the simulated deformable mirror (DM).

        Parameters
        ----------
        cmdhist : _ot.MatrixLike
            The command history to be uploaded to the DM.
        slaving : bool | str, optional
            Whether to apply slaving to the command history (default is False).

        """
        self._logger.info(
            f"Uploading command history of shape {cmdhist.shape} to {self._name}"
        )
        if slaving:
            cmdhist = _apply_slaving(self, cmdhist, slave=slaving)
        self._cmd_history = cmdhist.copy()
        print(f"Command history of shape {cmdhist.shape} uploaded.")

    def run_cmd_history(
        self,
        wfs: _ot.Optional[_ot.InterferometerDevice | _ot.WFSDevice] = None,
        save: str = None,
        **setshape_kwargs: dict[str, _ot.Any],
    ):
        """
        Runs the command history on the deformable mirror.

        Parameters
        ----------
        wfs : InterferometerDevice | WFSDevice, optional
            Wavefront sensor object used to acquire the data.
        rebin : int
            Rebinning factor for the acquired phase map.
        modal : bool
            If True, the command history is modal.
        differential : bool
            If True, the command history is applied differentially
            to the initial shape.

        Returns
        -------
        tn :str
            Timestamp of the data saved.
        """
        differential = setshape_kwargs.get("differential", False)
        delay: float = _rc.get_iff_config(key=None).get("sequential_delay", 0.0)
        slaving: bool | str = setshape_kwargs.get("slave", False)

        if self._cmd_history is None:
            self._logger.error("No Command History found in memory!")
            raise Exception("No Command History to run!")
        else:
            self._logger.info(
                f"Running command history of shape {self._cmd_history.shape}"
            )

            if wfs is not None:
                datafold, tn = osu.create_data_folder(get_tn=True)
            else:
                tn = save
                datafold = os.path.join(fp.OPD_IMAGES_ROOT_FOLDER, tn)
                if not os.path.exists(datafold):
                    os.mkdir(datafold)

            print(f"{tn} - {self._cmd_history.shape[-1]} modes.")
            s = self.get_shape()
            for i, cmd in enumerate(self._cmd_history.T):
                print(f"{i+1}/{self._cmd_history.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd += s
                self.set_shape(cmd, slave=slaving)
                if wfs is not None:
                    sleep(delay)
                    img = wfs.acquire_map()
                    path = os.path.join(datafold, f"image_{i:05d}.fits")
                    osu.save_fits(path, img)
        self.set_shape(s)
        return tn

    def _base_mirror_command(self, cmd: _ot.ArrayLike, differential: bool = False):
        """
        This is a basic foundational method for sending commands to a simulated
        mirror.

        If the DM does not need specialized command handling, the user should
        implement the ``_mirror_command`` (called by ``set_shape``) as

        ..code-block:: python

            def _mirror_command(self, cmd: _ot.ArrayLike, differential:bool = False):
                super()._base_mirror_command(cmd, differential=differential)


        Parameters
        ----------
        cmd : _ot.ArrayLike
            Commanded shape for the mirror's actuators.
        differential : bool, optional
            If True, the command is applied differentially to the current shape.
        """
        cmd = np.asarray(cmd)
        if not differential:
            cmd = cmd - self._act_pos
        self._shape[~self.pupil] += cmd @ self.IM
        self._act_pos += cmd

    def _wavefront(self, **kwargs: dict[str, _ot.Any]) -> _ot.ArrayLike:
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

    def visualize_shape(
        self,
        cmd: _ot.ArrayLike | None = None,
        annotate: bool = True,
        **kwargs: dict[str, _ot.Any],
    ):
        """
        Visualizes the shape of the mirror by plotting the actuator coordinates.

        Parameters
        ----------
        cmd : _ot.ArrayLike, optional
            Commanded shape to be visualized on the mirror's actuators. If None,
            the current actuator positions are used.
        annotate : bool, optional
            If True, annotate the actuator coordinates with their indices. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments for customization of the plot.
        """
        from opticalib.visualization import coordplot

        c = (cmd or kwargs.pop("c", None)) or self._coords[:, 2]
        s = kwargs.pop("s", 65)
        marker = kwargs.pop("marker", "s")
        coordplot(
            self._coords[:, :2],
            annotate=annotate,
            c=c,
            s=s,
            marker=marker,
            xlabel="X [m]",
            ylabel="Y [m]",
            title=f"Morfeo DM{self._which} actuator coordinates",
            **kwargs,
        )

    def scale_coords_to_pupil(
        self,
        coords: _ot.ArrayLike | None = None,
        *,
        mask: np.ndarray | None = None,
        radius: float | None = None,
        center: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        Rescale actuator coordinates onto a circular pupil while preserving their
        relative radial distribution.

        Parameters
        ----------
        coords : _ot.ArrayLike, optional
            Array of actuator coordinates, where the first two columns are the
            x/y coordinates. If omitted, uses ``self._coords``.
        mask : np.ndarray, optional
            Boolean or numeric aperture mask used to infer the pupil radius when
            no explicit ``radius`` is supplied.
        radius : float, optional
            Radius of the target circular pupil in pixels.
        center : tuple[float, float], optional
            Center of the circular pupil. If omitted, it is inferred from the
            coordinate distribution or from ``mask``.

        Returns
        -------
        np.ndarray
            Rescaled coordinate array of shape ``(n_actuators, 2)``.
        """
        if coords is None:
            coords = getattr(self, "coords", None)
        if coords is None:
            raise ValueError("No actuator coordinates were provided.")

        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2:
            raise ValueError("Coordinates must be a 2D array.")
        if coords.shape[0] == 2 and coords.shape[1] != 2:
            coords = coords.T
        if coords.shape[1] < 2:
            raise ValueError("Coordinates must contain at least two spatial columns.")

        xy = coords[:, :2]
        mask = (self.pupil or self._mask) if mask is None else mask

        if center is None:
            if mask is not None:
                y0, x0 = np.asarray(mask.shape) / 2.0
            else:
                x0, y0 = xy.mean(axis=0)
        else:
            x0, y0 = center

        dx = xy[:, 0] - x0
        dy = xy[:, 1] - y0
        r = np.hypot(dx, dy)

        if radius is None:
            if mask is not None:
                radius = min(mask.shape) / 2.0
            else:
                radius = float(np.max(r)) if r.size else 0.0

        rmax = float(np.max(r)) if r.size else 0.0
        if rmax == 0.0:
            return np.column_stack(
                [
                    np.full(xy.shape[0], x0, dtype=float),
                    np.full(xy.shape[0], y0, dtype=float),
                ]
            )

        scale = radius / rmax
        return np.column_stack([x0 + dx * scale, y0 + dy * scale])

    def _mirror_command(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        """
        Internal method to handle the mirror command.

        Parameters
        ----------
        cmd : _ot.ArrayLike
            The command to be sent to the DM.
        differential : bool, optional
            Whether the command is a differential command (default is False).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
