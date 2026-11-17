"""
Alignment procedure window
==========================

Calibration and correction of the optical alignment with
:class:`opticalib.procedures.Alignment` (``SYSTEM.ALIGNMENT`` section of the
configuration).
"""

from typing import List

from .base import ProcedureWindow, Step
from .params import DeviceParam, ExprParam, IntParam, VarParam

_MOVES = "This step moves the mechanical devices (e.g. the stages of the optics)."


class AlignmentWindow(ProcedureWindow):
    """Calibrate and correct the alignment of the optics."""

    TITLE = "Alignment"
    DESCRIPTION = (
        "Calibrate the interaction between the degrees of freedom of the "
        "optics and the Zernike modes of the wavefront, then compute and "
        "apply the correction. Devices, commands and modes come from the "
        "SYSTEM.ALIGNMENT section of the configuration."
    )

    def steps(self) -> List[Step]:
        """The alignment steps."""
        return [
            Step(
                "setup",
                "Set up",
                "Create the alignment object. The mechanical object must expose "
                "the move/read calls listed in SYSTEM.ALIGNMENT (e.g. "
                "ott.parabola.setPosition); the acquisition device must "
                "acquire phase maps.",
                [
                    DeviceParam(
                        "mech", "Mechanical devices", kinds=("other",), default="ott"
                    ),
                    DeviceParam(
                        "acq",
                        "Acquisition device",
                        kinds=("interferometer", "wfs", "camera"),
                        default="interf",
                    ),
                    VarParam("out", "Store the alignment object in", "align"),
                ],
                lambda v: f"{v['out']} = procedures.Alignment({v['mech']}, {v['acq']})",
            ),
            Step(
                "positions",
                "Read positions",
                "Print the current positions of the degrees of freedom.",
                [VarParam("align", "Alignment object", "align")],
                lambda v: f"{v['align']}.read_positions()",
            ),
            Step(
                "calibrate",
                "Calibrate",
                "Move each degree of freedom of the command matrix by ± its "
                "amplitude, measure the Zernike response and save the "
                "interaction matrix under a new tracking number in Alignment.",
                [
                    VarParam("align", "Alignment object", "align"),
                    ExprParam(
                        "amplitudes",
                        "Amplitudes (one per command)",
                        "[0.1, 0.1, 0.1, 0.1, 0.1]",
                    ),
                    IntParam(
                        "nframes", "Frames per measurement", 15, minimum=1, maximum=1000
                    ),
                ],
                lambda v: (
                    f"{v['align']}.calibrate_alignment(list({v['amplitudes']}), n_frames={v['nframes']})\n"
                    f"print('Calibration:', {v['align']}._calibtn)"
                ),
                outputs={"align_calib_tn": "{align}._calibtn"},
                confirm=_MOVES,
            ),
            Step(
                "compute",
                "Compute correction",
                "Measure the wavefront and compute the command that corrects "
                "the chosen modes with the chosen degrees of freedom "
                "(nothing moves).",
                [
                    VarParam("align", "Alignment object", "align"),
                    ExprParam(
                        "dofs", "Degrees of freedom (command indices)", "[0, 1, 2]"
                    ),
                    ExprParam("modes", "Modes to correct (indices)", "[0, 1, 2]"),
                    IntParam(
                        "nframes", "Frames to average", 15, minimum=1, maximum=1000
                    ),
                    VarParam("out", "Store the command in", "align_cmd"),
                ],
                lambda v: (
                    f"{v['out']} = {v['align']}.correct_alignment({v['dofs']}, {v['modes']}, "
                    f"n_frames={v['nframes']}, apply=False)\n"
                    f"print({v['out']})"
                ),
            ),
            Step(
                "apply",
                "Apply correction",
                "Measure again, compute the correction and move the degrees of "
                "freedom by it.",
                [
                    VarParam("align", "Alignment object", "align"),
                    ExprParam(
                        "dofs", "Degrees of freedom (command indices)", "[0, 1, 2]"
                    ),
                    ExprParam("modes", "Modes to correct (indices)", "[0, 1, 2]"),
                    IntParam(
                        "nframes", "Frames to average", 15, minimum=1, maximum=1000
                    ),
                ],
                lambda v: (
                    f"{v['align']}.correct_alignment({v['dofs']}, {v['modes']}, "
                    f"n_frames={v['nframes']}, apply=True)\n"
                    f"{v['align']}.read_positions()"
                ),
                confirm=_MOVES,
            ),
        ]
