"""
Segments phasing window
=======================

Differential piston measurement of segmented mirrors with the SPL sensor
(:class:`opticalib.procedures.SPL`: camera + tunable filter, ``PHASING``
section of the configuration).
"""

from typing import List

from .base import ProcedureWindow, Step, kwargs_code
from .params import ChoiceParam, DeviceParam, ExprParam, FloatParam, IntParam, TnParam, VarParam

_MOVES_FILTER = "This step moves the tunable filter."
# The PHASING section of the configuration (defaults of the SPL sensor).
_PHASING = "__import__('opticalib.core.config', fromlist=['_']).get_section_config('PHASING')"
_MODES = [("narrow", "Narrow"), ("medium", "Medium"), ("wide", "Wide")]


class SegmentsPhasingWindow(ProcedureWindow):
    """Measure the differential pistons of the segments with the SPL sensor."""

    TITLE = "Segments Phasing"
    DESCRIPTION = (
        "Acquire the PSFs of the segment borders over a sweep of wavelengths "
        "with the tunable filter, and compare their fringes with the "
        "templates to measure the differential pistons."
    )

    def steps(self) -> List[Step]:
        """The phasing steps."""
        return [
            Step(
                "setup",
                "Set up",
                "Create the SPL sensor. Without a camera or a filter, those "
                "named in the PHASING section of the configuration are used. "
                "The fringe templates are needed for the analysis.",
                [
                    DeviceParam("camera", "Camera", kinds=("camera",), default="cam", optional=True,
                                config_default=f"{_PHASING}['camera']"),
                    DeviceParam("filter", "Tunable filter", kinds=("other",), default="", optional=True,
                                config_default=f"{_PHASING}['filter']"),
                    TnParam("fringes", "Fringe templates", folder_attr="SPL_FRINGES_ROOT_FOLDER", optional=True),
                    VarParam("out", "Store the sensor in", "spl"),
                ],
                lambda v: (
                    f"{v['out']} = procedures.SPL(camera={v['camera']}, "
                    f"tunable_filter={v['filter']}, tnfringes={v['fringes']})"
                ),
            ),
            Step(
                "dark",
                "Dark frame",
                "Close the filter and acquire the dark frame used by the next "
                "acquisitions.",
                [
                    VarParam("spl", "SPL sensor", "spl"),
                    FloatParam("exptime", "Exposure time [s]", 0.01, minimum=0.0),
                    IntParam("nframes", "Frames", 1, minimum=1, maximum=1000),
                ],
                lambda v: f"spl_dark = {v['spl']}.acquire_dark_frame({v['exptime']}, nframes={v['nframes']})",
                confirm=_MOVES_FILTER,
            ),
            Step(
                "preview",
                "Preview detection",
                "Acquire one frame and show the PSFs the analysis would detect, "
                "to tune the exposure and the detection before a full sweep.",
                [
                    VarParam("spl", "SPL sensor", "spl"),
                    FloatParam("exptime", "Exposure time [s]", None, minimum=0.0, optional=True),
                    ChoiceParam("filter_mode", "Filter mode", [(None, "Unchanged")] + _MODES),
                    FloatParam("wavelength", "Wavelength [nm]", None, minimum=400, maximum=700, optional=True),
                    ExprParam("n_psfs", "Expected PSFs", "", placeholder="from configuration", optional=True,
                              config_default=f"{_PHASING}['expected_psfs']"),
                ],
                lambda v: (
                    f"{v['spl']}.preview_detection("
                    + kwargs_code(v, "exptime", "filter_mode", "wavelength", "n_psfs")
                    + ")"
                ),
                confirm=_MOVES_FILTER,
            ),
            Step(
                "acquire",
                "Acquire the sweep",
                "Acquire a dark frame, then one frame per wavelength of the "
                "sweep (400–700 nm), and save them under a new tracking number "
                "in SPL.",
                [
                    VarParam("spl", "SPL sensor", "spl"),
                    FloatParam("exptime", "Exposure time [s]", 0.01, minimum=0.0),
                    ChoiceParam("filter_mode", "Filter mode", _MODES),
                    ExprParam("wavelengths", "Wavelengths [nm]", "np.arange(440, 701, 20)"),
                    IntParam("nframes", "Frames per wavelength", 1, minimum=1, maximum=1000),
                    VarParam("out", "Store the tracking number in", "tn_spl"),
                ],
                # The dark frame comes first: without one, acquire() would take
                # it after setting the filter mode and sweep with the filter closed.
                lambda v: (
                    f"{v['spl']}.acquire_dark_frame({v['exptime']})\n"
                    f"{v['out']} = {v['spl']}.acquire({v['exptime']}, filter_mode={v['filter_mode']}, "
                    f"lambda_vector={v['wavelengths']}, nframes={v['nframes']})"
                ),
                outputs={"spl_tn": "{out}"},
                confirm=_MOVES_FILTER,
            ),
            Step(
                "analyse",
                "Analyse",
                "Extract the PSFs of a sweep, fit their fringes against the "
                "templates and print the differential piston of each one [nm].",
                [
                    VarParam("spl", "SPL sensor", "spl"),
                    TnParam("tn", "Sweep tracking number", folder_attr="SPL_DATA_ROOT_FOLDER", state_key="spl_tn"),
                    ExprParam("n_psfs", "Expected PSFs", "", placeholder="from configuration", optional=True,
                              config_default=f"{_PHASING}['expected_psfs']"),
                    VarParam("out", "Store the pistons in", "spl_pistons"),
                ],
                lambda v: (
                    f"{v['out']} = {v['spl']}.analysis({v['tn']}"
                    + (f", n_psfs={v['n_psfs']}" if v["n_psfs"] != "None" else "")
                    + ")\n"
                    f"print('Differential pistons [nm]:', {v['out']})\n"
                    f"{v['spl']}.plot_comparison({v['tn']})"
                ),
                outputs={"pistons": "{out}"},
            ),
        ]
