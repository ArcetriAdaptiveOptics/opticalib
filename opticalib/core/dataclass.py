"""
Dataclasses used across opticalib.
"""

import os as _os
from dataclasses import dataclass as _dc
from opticalib import typings as _ot
from opticalib.ground import osutils as _osu
from opticalib.core.root import folders as _fn
from collections import OrderedDict as _OrderedDict


@_dc(init=True, repr=True)
class FlatData:
    """
    Dataclass for flattening data, to be used for loading flattening results.
    """

    tn: str

    def __post_init__(self):
        filelist = _osu.getFileList(self.tn, _fn.FLAT_ROOT_FOLDER)

        attr_map = {
            "flatPosition": "_cmd",
            "flatDeltaCommand": "_deltacmd",
            "imgstart": "_imgstart",
            "imgflat": "_imgflat",
            "modes2flat": "_modes2flat",
            "cavityOffset": "_cavityOffset",
            "flatCommand": "_flatCommand",
            "BiasCommand": "_biasCommand",
            "BiasForces": "_biasForces",
            "flatTotalForces": "_flatTotalForces",
        }

        for key, attr in attr_map.items():
            for file in filelist:
                if key in file:
                    setattr(
                        self,
                        attr,
                        _osu.load_fits(
                            _os.path.join(_fn.FLAT_ROOT_FOLDER, self.tn, file)
                        ),
                    )
                    break
                setattr(self, attr, None)  # If not found set to None

    @property
    def flat_cmd(self) -> _ot.ArrayLike:
        """
        Absolute flat command.
        """
        return self._cmd

    @property
    def delta_cmd(self) -> _ot.ArrayLike:
        """
        Delta flat command.
        """
        return self._deltacmd

    @property
    def start_image(self) -> _ot.ImageData:
        """
        Starting image.
        """
        return self._imgstart

    @property
    def flat_image(self) -> _ot.ImageData:
        """
        Flattened image.
        """
        return self._imgflat

    @property
    def flattened_modes(self) -> _ot.ArrayLike:
        """
        Modes that have been flattened.
        """
        return self._modes2flat

    @property
    def cavity_offset(self) -> _ot.ImageData:
        """
        Cavity offset used in the flattening.
        """
        return self._cavityOffset

    @property
    def bias_command(self) -> _ot.ArrayLike:
        """
        DM bias command.
        """
        return self._biasCommand

    @property
    def bias_forces(self) -> _ot.ArrayLike:
        """
        DM bias forces.
        """
        return self._biasForces

    @property
    def total_forces(self) -> _ot.ArrayLike:
        """
        Total forces on the DM relative to the flattening command.
        """
        return self._flatTotalForces

    def __repr__(self) -> str:
        return f"FlatData(tn={self.tn})"


@_dc(init=True, repr=True)
class IffData:
    """
    Dataclass for Influence Function Data loading.

    Loads, for a specific Tracking Number, the following data:
    - Amplitude vector
    - Command matrix
    - Modes vector
    - Registration actuators
    - Template
    - Timed command history
    - Modes images (loaded on demand)
    - Cube of images
    """

    tn: str

    def __post_init__(self):
        filelist = _osu.getFileList(self.tn, _fn.IFFUNCTIONS_ROOT_FOLDER)

        attr_map = {
            "ampVector": "_amplitude",
            "cmdMatrix": "_cmd_matrix",
            "modesVector": "_modes_vector",
            "regActs": "_reg_acts",
            "template": "_template",
            "timedCmdHistory": "_timed_cmd_history",
        }

        for key, attr in attr_map.items():
            for file in filelist:
                if key in file:
                    setattr(self, attr, _osu.load_fits(file))
                    break
                setattr(self, attr, None)  # If not found set to None

        del filelist

        self._modesfl = _osu.getFileList(
            self.tn, _fn.IFFUNCTIONS_ROOT_FOLDER, key="mode_"
        )
        self._modes = [None] * len(self._modesfl)
        self._cache: _OrderedDict[str, _ot.ImageData] = _OrderedDict()

        try:
            self._cube = _osu.load_fits(
                _os.path.join(_fn.INTMAT_ROOT_FOLDER, self.tn, "IMCube.fits")
            )
        except FileNotFoundError:
            self._cube = None

    def _get_mode(self, index: int) -> _ot.ImageData:
        """
        Get the mode image at the specified index, using caching to avoid
        redundant file loading.
        """
        if index < 0 or index >= len(self._modesfl):
            raise IndexError(f"Mode index {index} is out of range.")

        if self._modes[index] is not None:
            return self._modes[index]

        if index in self._cache:
            return self._cache[index]

        mode_file = self._modesfl[index]
        mode_image = _osu.load_fits(mode_file)
        self._modes[index] = mode_image

        # Cache the loaded mode image
        self._cache[index] = mode_image

        # Limit cache size to avoid excessive memory usage
        if len(self._cache) > 50:  # Arbitrary cache size limit
            self._cache.popitem(last=False)  # Remove the oldest cached item

        return mode_image

    @property
    def amplitude(self) -> _ot.ArrayLike:
        """
        Amplitude vector.
        """
        return self._amplitude

    @property
    def cube(self) -> _ot.ImageData:
        """
        Cube image.
        """
        return self._cube

    @property
    def command_matrix(self) -> _ot.ArrayLike:
        """
        Command matrix.
        """
        return self._cmd_matrix

    @property
    def modes_vector(self) -> _ot.ArrayLike:
        """
        Modes vector.
        """
        return self._modes_vector

    @property
    def registration_acts(self) -> _ot.ArrayLike:
        """
        Actuators used for IFF registration.
        """
        return self._reg_acts

    @property
    def template(self) -> _ot.ImageData:
        """
        Template image.
        """
        return self._template

    @property
    def timed_command_history(self) -> _ot.ArrayLike:
        """
        Timed command matrix history.
        """
        return self._timed_cmd_history

    def mode(self, index: int) -> _ot.ImageData:
        """
        Get the mode image at the specified index.
        """
        return self._get_mode(index)

    def __repr__(self) -> str:
        return (
            f"IffData(tn={self.tn}, "
            f"n_modes={len(self._modes_vector) if self._modes_vector is not None else 'N/A'}, "
            f"template={self._template if self._template is not None else 'N/A'})"
        )
