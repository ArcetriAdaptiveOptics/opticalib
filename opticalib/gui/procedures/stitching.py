"""
Stitching procedure window
==========================

Sub-aperture scan with motorised stages
(:class:`opticalib.procedures.stitching.StitchAcquire`) and stitching of
the scan into one map (:class:`opticalib.procedures.stitching.StitchAnalysis`,
using the ``STITCHING`` section of the configuration).
"""

from typing import List

from .base import ProcedureWindow, Step
from .params import BoolParam, DeviceParam, ExprParam, FloatParam, IntParam, TnParam, VarParam

_IMPORT = "from opticalib.procedures.stitching import StitchAcquire, StitchAnalysis\n"


class StitchingWindow(ProcedureWindow):
    """Scan the sub-apertures of a large optic and stitch them together."""

    TITLE = "Stitching"
    DESCRIPTION = (
        "Move the stages over a grid of sub-apertures, acquire one map at "
        "each position and stitch them into a single map (pixel scale, "
        "rotation and home coordinates come from the STITCHING section of "
        "the configuration)."
    )

    def steps(self) -> List[Step]:
        """The scan and stitching steps."""
        return [
            Step(
                "grid",
                "Plan the grid",
                "Build a serpentine grid of positions around the current stage "
                "position (the stages are only read, not moved).",
                [
                    DeviceParam("dm", "Deformable mirror", kinds=("dm",), default="dm"),
                    DeviceParam("wfs", "Wavefront sensor", kinds=("interferometer", "wfs", "camera"), default="interf"),
                    DeviceParam("motors", "Stages", kinds=("other",), default="motors"),
                    IntParam("nstep", "Positions per axis", 3, minimum=1, maximum=100),
                    FloatParam("step_x", "Step along x [mm]", 3.0, minimum=0.0),
                    FloatParam("step_z", "Step along z [mm]", 3.0, minimum=0.0),
                    VarParam("out", "Store the coordinates in", "stitch_coords"),
                ],
                lambda v: (
                    _IMPORT
                    + f"stitch_acq = StitchAcquire({v['dm']}, {v['wfs']}, {v['motors']})\n"
                    f"{v['out']} = stitch_acq.get_coordinates_vector("
                    f"{v['nstep']}, step_in_mm=({v['step_x']}, {v['step_z']}), live_pos=True)\n"
                    f"print(len({v['out']}), 'positions:', {v['out']})"
                ),
                outputs={"n_positions": "len({out})"},
            ),
            Step(
                "scan",
                "Acquire the scan",
                "Move the stages through every position, acquire the maps and "
                "save them as one cube in OPDImages (positions in the header).",
                [
                    ExprParam("coords", "Positions", "stitch_coords"),
                    IntParam("nframes", "Frames per position", 1, minimum=1, maximum=1000),
                    BoolParam("homing", "Go home at the end", True),
                    VarParam("out", "Store the tracking number in", "tn_scan"),
                ],
                lambda v: (
                    f"{v['out']} = stitch_acq.acquire_single_scan("
                    f"{v['coords']}, nframes={v['nframes']}, homing={v['homing']})"
                ),
                outputs={"scan_tn": "{out}"},
                confirm="This step moves the stages through every position of the grid.",
            ),
            Step(
                "stitch",
                "Stitch",
                "Stitch the maps of a scan into one map, optionally removing a "
                "polynomial of the given degree from each sub-aperture.",
                [
                    TnParam("tn", "Scan tracking number", folder_attr="OPD_IMAGES_ROOT_FOLDER", state_key="scan_tn"),
                    ExprParam("deg", "Polynomial degree to remove", "", placeholder="none", optional=True),
                    VarParam("out", "Store the stitched map in", "stitched"),
                ],
                lambda v: (
                    _IMPORT
                    + f"{v['out']} = StitchAnalysis().stitch_single_scansion_cube({v['tn']}, deg={v['deg']})\n"
                    f"_gui.view({v['out']}, 'Stitched ' + {v['tn']})"
                ),
            ),
        ]
