"""
Deformable-mirror calibration window
====================================

Influence-function acquisition (:func:`opticalib.procedures.iff_data_acquisition`),
processing into an interaction matrix
(:func:`opticalib.dmutils.iff_processing.process`) and flattening
(:class:`opticalib.dmutils.flattening.Flattening`).
"""

from typing import Any, Dict, List

from .base import ProcedureWindow, Step, kwargs_code
from .params import (
    BoolParam,
    ChoiceParam,
    DeviceParam,
    ExprParam,
    IntParam,
    TextParam,
    TnListParam,
    TnParam,
    VarParam,
)

_WFS = ("interferometer", "wfs")
_CALIBRATION = ("acquire", "process", "svd", "flatten", "loop")
_MOVES_DM = "This step moves the deformable mirror."

# Defaults of the acquisition, as read by iff_data_acquisition from the
# INFLUENCE.FUNCTIONS/IFFUNC section.
_IFFUNC = "__import__('opticalib.core.config', fromlist=['_']).get_iff_config('IFFUNC')"

# Where get_iff_config() reads the INFLUENCE.FUNCTIONS defaults from.
_IFF_CONFIG_CHECK = (
    "__import__('os').path.normpath(__import__('os').environ.get('AOCONF', '')) == "
    "__import__('os').path.normpath(__import__('os').path.join("
    "folders.CONFIGURATION_FOLDER, 'configuration.yaml'))"
)


def _modal_base(values: Dict[str, str]) -> str:
    """The modal base argument: a custom file wins over the choice."""
    return values["base_file"] if values["base_file"] != "None" else values["base"]


def _after_process(window: "DeformableMirrorCalibrationWindow", results: Dict[str, Any], values: Dict[str, str]) -> None:
    """The interaction matrix is saved under the tracking number of the acquisition."""
    import ast

    window.set_state("intmat_tn", ast.literal_eval(values["tn"]))


class DeformableMirrorCalibrationWindow(ProcedureWindow):
    """Acquire influence functions, build the interaction matrix and flatten the DM."""

    TITLE = "Deformable Mirror Calibration"
    DESCRIPTION = (
        "Acquire the influence functions of the DM, process them into an "
        "interaction matrix, look at its singular values and flatten the "
        "mirror, once or in closed loop. Empty optional fields use the "
        "INFLUENCE.FUNCTIONS section of the configuration."
    )
    CHECKS = [
        (
            _IFF_CONFIG_CHECK,
            "Influence-function acquisitions read their defaults from "
            "SysConfig/configuration.yaml in the data folder, which is not the "
            "configuration loaded by the GUI; the acquisition fails if that file does "
            "not exist. Open an experiment created with 'calpy --create' (its "
            "configuration lives in SysConfig), or copy the configuration there.",
        )
    ]

    def steps(self) -> List[Step]:
        """The calibration steps, then the post-processing of the interaction matrix."""
        steps = self._steps_list()
        for step in steps:
            step.section = "Calibration" if step.key in _CALIBRATION else "Post-processing"
        return steps

    def _steps_list(self) -> List[Step]:
        return [
            Step(
                "acquire",
                "Acquire influence functions",
                "Apply the modal command history to the DM and acquire one "
                "wavefront per command. Creates a tracking number in "
                "IFFunctions (commands) and OPDImages (frames).",
                [
                    DeviceParam("dm", "Deformable mirror", kinds=("dm",), default="dm"),
                    DeviceParam("wfs", "Wavefront sensor", kinds=_WFS, default="interf"),
                    ExprParam("modes", "Modes", "", placeholder="from configuration", optional=True,
                              config_default=f"{_IFFUNC}['modes_list']"),
                    ExprParam("amplitude", "Amplitude", "", placeholder="from configuration", optional=True,
                              config_default=f"{_IFFUNC}['amplitude']"),
                    ExprParam("template", "Push-pull template", "", placeholder="from configuration", optional=True,
                              config_default=f"{_IFFUNC}['template']"),
                    ChoiceParam(
                        "base", "Modal base",
                        [(None, "From configuration"), ("zonal", "Zonal (one actuator at a time)"),
                         ("hadamard", "Hadamard"), ("mirror", "Mirror modes of the DM")],
                        config_default=f"{_IFFUNC}['modal_base']",
                    ),
                    TextParam("base_file", "Custom modal base file", placeholder="file in ModalBases (overrides the choice)", optional=True),
                    BoolParam("shuffle", "Shuffle the modes", False),
                    IntParam("repetitions", "Repetitions", 1, minimum=1, maximum=100),
                    VarParam("out", "Store the tracking number in", "tn_iff"),
                ],
                lambda v: (
                    f"{v['out']} = ifm.iff_data_acquisition({v['dm']}, {v['wfs']}"
                    + "".join(
                        f", {name}={code}"
                        for name, code in (
                            ("modeslist", v["modes"]),
                            ("amplitude", v["amplitude"]),
                            ("template", v["template"]),
                            ("modalbase", _modal_base(v)),
                        )
                        if code != "None"
                    )
                    + f", shuffle={v['shuffle']}, n_repetitions={v['repetitions']})"
                ),
                outputs={"iff_tn": "{out}"},
                confirm=_MOVES_DM,
            ),
            Step(
                "process",
                "Process",
                "Reduce the frames of an acquisition into one influence function "
                "per mode and save the interaction-matrix cube in INTMatrices "
                "(same tracking number), needed for flattening.",
                [
                    TnParam("tn", "Acquisition tracking number", folder_attr="IFFUNCTIONS_ROOT_FOLDER", state_key="iff_tn"),
                    BoolParam("register", "Register the frames", False),
                    IntParam("rebin", "Rebin factor", 1, minimum=1, maximum=16),
                    IntParam("nworkers", "Parallel workers", 2, minimum=1, maximum=64),
                ],
                lambda v: (
                    f"ifp.process({v['tn']}, register={v['register']}, save=True, "
                    f"rebin={v['rebin']}, nworkers={v['nworkers']})"
                ),
                after=_after_process,
            ),
            Step(
                "svd",
                "Singular values",
                "Load the interaction matrix, measure the current mirror shape "
                "and plot the singular values of the interaction matrix, to "
                "choose how many modes to discard when flattening.",
                [
                    TnParam("tn", "Interaction matrix tracking number", folder_attr="INTMAT_ROOT_FOLDER", state_key="intmat_tn"),
                    DeviceParam("dm", "Deformable mirror", kinds=("dm",), default="dm"),
                    DeviceParam("wfs", "Wavefront sensor", kinds=_WFS, default="interf"),
                    IntParam("nframes", "Frames to average", 5, minimum=1, maximum=1000),
                    VarParam("out", "Store the flattening object in", "flat"),
                ],
                lambda v: (
                    f"{v['out']} = dmutils.Flattening({v['tn']}, dm={v['dm']}, wfs={v['wfs']})\n"
                    f"{v['out']}.load_image2_shape({v['wfs']}.acquire_map({v['nframes']}))\n"
                    f"{v['out']}.compute_rec_mat()\n"
                    f"_, flat_sv, _ = {v['out']}.get_svd_matrices()\n"
                    f"figure('Singular values ' + {v['tn']})\n"
                    f"semilogy(np.asarray(flat_sv), 'o-'); xlabel('mode'); ylabel('singular value')\n"
                    f"title(f'{{len(flat_sv)}} modes')"
                ),
                outputs={"n_modes": "len(flat_sv)"},
            ),
            Step(
                "flatten",
                "Apply flat command",
                "Measure the mirror, compute the command that flattens it with "
                "the interaction matrix, apply it and measure again. The maps "
                "before and after are shown; the data are saved in Flattening.",
                [
                    VarParam("flat", "Flattening object", "flat"),
                    ExprParam("modes2flat", "Modes to flatten", "", placeholder="all actuators (int or list)", optional=True),
                    ExprParam("modes2discard", "Singular values to discard", "", placeholder="none (int: smallest N)", optional=True),
                    IntParam("nframes", "Frames to average", 5, minimum=1, maximum=1000),
                    VarParam("out", "Store the tracking number in", "tn_flat"),
                ],
                lambda v: (
                    f"{v['out']} = {v['flat']}.apply_flat_command("
                    + kwargs_code(v, "modes2flat", "modes2discard", "nframes")
                    + ")\n"
                    f"_gui.view(osu.load_fits(join(folders.FLAT_ROOT_FOLDER, {v['out']}, 'imgstart.fits')), 'Before flattening')\n"
                    f"_gui.view(osu.load_fits(join(folders.FLAT_ROOT_FOLDER, {v['out']}, 'imgflat.fits')), 'After flattening')"
                ),
                outputs={"flat_tn": "{out}"},
                confirm=_MOVES_DM,
            ),
            Step(
                "loop",
                "Closed-loop flattening",
                "Repeat the flattening several times; each iteration is saved "
                "under its own tracking number and the final map is shown.",
                [
                    VarParam("flat", "Flattening object", "flat"),
                    IntParam("iterations", "Iterations", 3, minimum=1, maximum=100),
                    ExprParam("modes2flat", "Modes to flatten", "", placeholder="all actuators (int or list)", optional=True),
                    ExprParam("modes2discard", "Singular values to discard", "", placeholder="none (int: smallest N)", optional=True),
                    IntParam("nframes", "Frames to average", 5, minimum=1, maximum=1000),
                    VarParam("out", "Store the tracking numbers in", "tn_flat_loop"),
                ],
                lambda v: (
                    f"{v['out']} = []\n"
                    f"for _i in range({v['iterations']}):\n"
                    f"    {v['out']}.append({v['flat']}.apply_flat_command("
                    + kwargs_code(v, "modes2flat", "modes2discard", "nframes")
                    + "))\n"
                    f"    print(f'{{_i + 1}}/{v['iterations']}')\n"
                    f"_gui.view(osu.load_fits(join(folders.FLAT_ROOT_FOLDER, {v['out']}[-1], 'imgflat.fits')), 'After closed loop')"
                ),
                outputs={"flat_tn": "{out}[-1]"},
                confirm=_MOVES_DM,
            ),
            Step(
                "filter",
                "Filter Zernike modes",
                "Remove Zernike modes (e.g. piston, tip, tilt) from the "
                "interaction matrix; the filtered matrix gets a new tracking number.",
                [
                    TnParam("tn", "Interaction matrix tracking number", folder_attr="INTMAT_ROOT_FOLDER", state_key="intmat_tn"),
                    ExprParam("zernikes", "Zernike modes", "[1, 2, 3]"),
                    VarParam("out", "Store the tracking number in", "tn_filtered"),
                ],
                lambda v: f"_, {v['out']} = ifp.remove_zernike_from_iff({v['tn']}, zern_modes={v['zernikes']}, save=True)",
                outputs={"intmat_tn": "{out}"},
            ),
            Step(
                "stack",
                "Stack cubes",
                "Stack the interaction matrices of several tracking numbers (e.g. "
                "the segments of a segmented mirror) into one cube, with its "
                "command matrix and modes; the stack gets a new tracking number.",
                [
                    TnListParam("tns", "Tracking numbers to stack", folder_attr="INTMAT_ROOT_FOLDER", minimum=2),
                    VarParam("out", "Store the tracking number in", "tn_stacked"),
                ],
                lambda v: f"{v['out']} = ifp.stack_cubes({v['tns']})",
                outputs={"intmat_tn": "{out}"},
            ),
            Step(
                "roi",
                "ROI processing",
                "Process each influence function within the region of interest "
                "of the actuated segment, using the other regions as reference "
                "(tip/tilt detrend, mean or median removal, nulling of the other "
                "regions); the result gets a new tracking number.",
                [
                    TnParam("tn", "Interaction matrix tracking number", folder_attr="INTMAT_ROOT_FOLDER", state_key="intmat_tn"),
                    IntParam("roi", "Active ROI (index)", 0, minimum=0, maximum=1000),
                    BoolParam("tt_detrend", "Remove tip/tilt using the other ROIs", False),
                    BoolParam("mean_subtraction", "Subtract the mean of the active ROI", False),
                    BoolParam("median_subtraction", "Subtract the median of the active ROI", False),
                    BoolParam("roinull", "Set the other ROIs to zero", False),
                    ExprParam("fitting_mask", "Fitting mask", "", placeholder="none", optional=True),
                    VarParam("out", "Store the tracking number in", "tn_roi"),
                ],
                lambda v: (
                    f"{v['out']} = ifp.cube_roi_processing({v['tn']}, {v['roi']}, "
                    + kwargs_code(v, "fitting_mask", "tt_detrend", "mean_subtraction", "median_subtraction", "roinull")
                    + ")"
                ),
                outputs={"intmat_tn": "{out}"},
            ),
        ]
