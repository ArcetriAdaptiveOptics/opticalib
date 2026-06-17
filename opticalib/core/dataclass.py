"""
Dataclasses used across opticalib.
"""

import os as _os
from dataclasses import dataclass as _dc
from opticalib import typings as _ot
from opticalib.ground import osutils as _osu
from opticalib.core.root import folders as _fn


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
