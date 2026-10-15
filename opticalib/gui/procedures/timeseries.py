"""
Timeseries procedure window
===========================

Acquire a sequence of frames (:class:`opticalib.procedures.TimeSeries`) and
analyse its stability with :mod:`opticalib.analyzer.timeseries`.
"""

from typing import List

from .base import ProcedureWindow, Step, kwargs_code
from .params import (
    BoolParam,
    DeviceParam,
    ExprParam,
    FloatParam,
    IntParam,
    TnParam,
    VarParam,
)

_SERIES = "OPD_SERIES_ROOT_FOLDER"


class TimeseriesWindow(ProcedureWindow):
    """Acquire and analyse a time series of frames."""

    TITLE = "Timeseries"
    DESCRIPTION = (
        "Acquire a sequence of frames with an interferometer or a camera, then "
        "study its stability: average, frame-to-frame differences and noise "
        "versus time lag."
    )

    def steps(self) -> List[Step]:
        """The acquisition and analysis steps."""
        return [
            Step(
                "acquire",
                "Acquire the series",
                "Acquire frames one after the other and save them under a new "
                "tracking number in OPDSeries. Frames are named after the "
                "second they are taken, so the delay must be at least 1 s.",
                [
                    DeviceParam(
                        "device",
                        "Device",
                        kinds=("interferometer", "wfs", "camera"),
                        default="interf",
                    ),
                    IntParam("nframes", "Frames", 20, minimum=2, maximum=100000),
                    FloatParam("delay", "Delay between frames [s]", 1.0, minimum=1.0),
                    VarParam("out", "Store the tracking number in", "tn_ts"),
                ],
                lambda v: (
                    f"{v['out']} = procedures.TimeSeries({v['device']})"
                    f".acquire_time_series({v['nframes']}, delay={v['delay']})"
                ),
                outputs={"ts_tn": "{out}"},
            ),
            Step(
                "average",
                "Average",
                "Average the frames of a series (optionally a range of them) "
                "and show the result.",
                [
                    TnParam(
                        "tn", "Tracking number", folder_attr=_SERIES, state_key="ts_tn"
                    ),
                    IntParam("first", "First frame", 0, minimum=0),
                    IntParam("last", "Last frame (-1: all)", -1, minimum=-1),
                    BoolParam("thresh", "Average only valid pixels", False),
                    VarParam("out", "Store the average in", "ts_avg"),
                ],
                lambda v: (
                    f"{v['out']} = az.timeseries.average_frames({v['tn']}, "
                    f"{kwargs_code(v, 'first', 'last', 'thresh')})\n"
                    f"_gui.view({v['out']}, 'Average ' + {v['tn']})"
                ),
            ),
            Step(
                "diff",
                "Running difference",
                "Subtract pairs of frames separated by the gap and plot the RMS "
                "of each difference: a stable system gives a flat curve.",
                [
                    TnParam(
                        "tn", "Tracking number", folder_attr=_SERIES, state_key="ts_tn"
                    ),
                    IntParam("gap", "Gap [frames]", 2, minimum=1),
                    BoolParam("zernikes", "Remove the default Zernike modes", False),
                    VarParam("out", "Store the RMS values in", "ts_stds"),
                ],
                lambda v: (
                    f"ts_diffs, {v['out']} = az.timeseries.running_diff("
                    f"{v['tn']}, gap={v['gap']}, remove_zernikes={v['zernikes']})\n"
                    f"figure('Running difference ' + {v['tn']})\n"
                    f"plot({v['out']}, '.-'); xlabel('difference'); ylabel('RMS')\n"
                    f"title('Running difference, gap {v['gap']}')"
                ),
            ),
            Step(
                "noise",
                "Noise vs time lag",
                "Structure function of the noise: RMS of the difference of "
                "frames separated by each time lag, after removing the chosen "
                "Zernike modes.",
                [
                    TnParam(
                        "tn", "Tracking number", folder_attr=_SERIES, state_key="ts_tn"
                    ),
                    ExprParam("taus", "Time lags [frames]", "np.arange(1, 6)"),
                    ExprParam("zernikes", "Zernike modes to remove", "[1, 2, 3]"),
                    VarParam("out", "Store the RMS values in", "ts_noise"),
                ],
                lambda v: (
                    f"{v['out']} = az.timeseries.noise_strfunct("
                    f"{v['tn']}, np.asarray({v['taus']}), zernike_vector={v['zernikes']})\n"
                    f"figure('Noise vs time lag ' + {v['tn']})\n"
                    f"plot(np.asarray({v['taus']}), {v['out']}, 'o-'); xlabel('lag [frames]'); ylabel('RMS')"
                ),
            ),
        ]
