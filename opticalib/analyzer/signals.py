"""
Module: signals
===============

Author(s)
---------
- Pietro Ferraiuolo

Description
-----------

Module containing functions for signal processing and frequency analysis within
the Opticalib framework.
"""

import numpy as _np
from numpy import fft as _fft
# scipy.signal provides detrend and get_window functions not available in numpy.fft
from scipy import signal as _signal
from .. import typings as _ot


def extract_frequency_spectrum(
    signal: _np.ndarray,
    sample_rate: float = 1.0,
    axis: int = -1,
    window: str | None = None,
    detrend: str | None = None,
    scaling: str = "density",
) -> dict[str, _np.ndarray]:
    """
    Extract the frequency spectrum from an input signal using Fast Fourier Transform.

    This function computes the frequency domain representation of a signal, handling
    multi-dimensional arrays and providing options for spectral windowing and detrending.

    Parameters
    ----------
    signal : ndarray
        Input signal array (can be 1D, 2D, or higher dimensional).
    sample_rate : float, optional
        Sampling rate of the signal in Hz (default: 1.0). Determines frequency axis scaling.
    axis : int, optional
        Axis along which to compute the FFT (default: -1, last axis).
    window : str or None, optional
        Window function to apply before FFT to reduce spectral leakage:
        - None: No windowing (rectangular window)
        - "hann": Hann window (default, good general purpose)
        - "hamming": Hamming window
        - "blackman": Blackman window (excellent side-lobe suppression)
        - "tukey": Tukey window
    detrend : str or None, optional
        Detrending method applied before FFT:
        - None: No detrending
        - "constant": Remove mean (default)
        - "linear": Remove linear trend
    scaling : str, optional
        Scaling of the power spectral density:
        - "density": Power spectral density (default)
        - "magnitude": Magnitude spectrum

    Returns
    -------
    dict with keys:
        - "frequencies" : 1D ndarray
            Frequency axis in Hz
        - "magnitude" : ndarray
            Magnitude spectrum (same shape as input except on FFT axis)
        - "power" : ndarray
            Power spectral density
        - "phase" : ndarray
            Phase spectrum in radians
        - "fft" : ndarray
            Raw complex FFT output

    Examples
    --------
    >>> import numpy as np
    >>> # Create a test signal: sum of 5 Hz and 10 Hz sinusoids
    >>> fs = 100  # 100 Hz sampling rate
    >>> t = np.arange(0, 1, 1/fs)
    >>> signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t)
    >>>
    >>> result = extract_frequency_spectrum(signal, sample_rate=fs)
    >>> frequencies = result['frequencies']
    >>> power = result['power']
    >>>
    >>> # Find dominant frequencies
    >>> peak_idx = np.argsort(power)[-2:]  # Top 2 peaks
    >>> print(f"Dominant frequencies: {frequencies[peak_idx]}")
    """

    # Input validation
    signal = _np.asarray(signal)
    if signal.size == 0:
        raise ValueError("Input signal cannot be empty")

    # Normalize axis
    if axis < 0:
        axis = signal.ndim + axis
    if axis < 0 or axis >= signal.ndim:
        raise ValueError(
            f"Axis {axis} out of bounds for array of dimension {signal.ndim}"
        )

    # Apply detrending
    if detrend is not None:
        signal = _signal.detrend(signal, axis=axis, type=detrend)

    # Apply windowing to reduce spectral leakage
    if window is not None:
        # Create window with correct shape
        window_shape = [signal.shape[i] if i == axis else 1 for i in range(signal.ndim)]
        window_array = _signal.get_window(window, signal.shape[axis])
        window_array = window_array.reshape(window_shape)
        signal = signal * window_array

    # Compute FFT
    fft_result = _fft.fft(signal, axis=axis)

    # Compute magnitude and phase
    magnitude = _np.abs(fft_result)
    phase = _np.angle(fft_result)

    # Compute frequency axis (one-sided spectrum)
    n_samples = signal.shape[axis]
    frequencies = _fft.fftfreq(n_samples, d=1 / sample_rate)

    # Compute power spectral density
    power = magnitude**2 / n_samples

    # Apply window correction factors if applicable
    if window is not None:
        # Correct for window power loss
        window_power = _np.sum(window_array**2) / n_samples
        power = power / window_power

    # For scaling: convert to one-sided spectrum if considering positive frequencies
    if scaling == "density":
        # Two-sided to one-sided conversion for positive frequencies
        power_one_sided = _np.copy(power)
        slices = [slice(None)] * signal.ndim
        slices[axis] = slice(1, n_samples // 2)
        power_one_sided[tuple(slices)] *= 2
    else:
        power_one_sided = power

    return {
        "frequencies": frequencies,
        "magnitude": magnitude,
        "power": power_one_sided,
        "phase": phase,
        "fft": fft_result,
    }


def extract_amplitude_spectrum(
    signal: _np.ndarray,
    sample_rate: float = 1.0,
    positive_freqs_only: bool = True,
    **kwargs: dict[str, _ot.Any],
) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Simplified interface for extracting amplitude spectrum (magnitude vs frequency).

    This is a convenience wrapper around extract_frequency_spectrum for the most
    common use case: getting the amplitude spectrum.

    Parameters
    ----------
    signal : ndarray
        Input signal array
    sample_rate : float, optional
        Sampling rate in Hz (default: 1.0)
    positive_freqs_only : bool, optional
        If True (default), return only positive frequencies (0 to Nyquist)
    **kwargs
        Additional arguments passed to extract_frequency_spectrum

    Returns
    -------
    frequencies : ndarray
        Frequency axis
    amplitude : ndarray
        Amplitude (magnitude) spectrum

    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> freqs, amplitude = extract_amplitude_spectrum(signal, sample_rate=100)
    """
    result = extract_frequency_spectrum(signal, sample_rate=sample_rate, **kwargs)

    freqs = result["frequencies"]
    amp = result["magnitude"]

    if positive_freqs_only:
        axis = -1  # Default axis
        pos_idx = freqs >= 0
        slices = [slice(None)] * len(amp.shape)
        slices[axis] = pos_idx
        return freqs[pos_idx], amp[tuple(slices)]

    return freqs, amp


# TODO: To remove in favor of the more general one above
def spectrum(
    signal: _ot.ArrayLike, dt: float = 1, show: bool = False
) -> tuple[_ot.ArrayLike, _ot.ArrayLike]:
    """
    Computes the one-dimensional power spectrum of a signal or a set of signals.

    Parameters
    ----------
    signal : ndarray
        Input signal or signals.
    dt : float, optional
        Time spacing between samples. The default is 1.
    show : bool, optional
        If True, displays the power spectrum. The default is None.

    Returns
    -------
    spe : float | ndarray
        Power spectrum of the input signal(s).
    freq : float | ArrayLike
        Frequency bins corresponding to the power spectrum.
    """
    nsig = signal.shape
    thedim = 0 if _np.size(nsig) == 1 else 1
    spe = _fft.rfft(signal, axis=thedim, norm="ortho")
    nn = _np.sqrt(spe.shape[thedim])  # modRB
    spe = (_np.abs(spe)) / nn
    freq = _fft.rfftfreq(signal.shape[thedim], d=dt)
    if _np.size(nsig) == 1:
        spe[0] = 0
    else:
        spe[:, 0] = 0
    # if show:
    #     _plt.figure()
    #     for i in range(0, len(spe)):
    #         _plt.plot(freq, spe[i, :], label=f"Channel {i}")
    #     _plt.xlabel(r"Frequency $[\mathrm{Hz}]$")
    #     _plt.ylabel("PS Amplitude")
    #     _plt.legend(loc="best")
    #     _plt.show()
    return spe, freq


__all__ = ["extract_frequency_spectrum", "extract_amplitude_spectrum", "spectrum"]
