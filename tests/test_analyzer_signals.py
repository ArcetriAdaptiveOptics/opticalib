"""
Tests for opticalib.analyzer.signals module.
"""

import pytest
import numpy as np
from opticalib.analyzer import signals


class TestExtractFrequencySpectrum:
    """Test extract_frequency_spectrum function."""

    def test_basic_1d_signal(self):
        """Test with a basic 1D signal."""
        fs = 100.0
        t = np.arange(0, 1, 1 / fs)
        signal = np.sin(2 * np.pi * 10 * t)
        result = signals.extract_frequency_spectrum(signal, sample_rate=fs)

        assert isinstance(result, dict)
        assert "frequencies" in result
        assert "magnitude" in result
        assert "power" in result
        assert "phase" in result
        assert "fft" in result

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        n = 100
        signal = np.random.randn(n)
        result = signals.extract_frequency_spectrum(signal, sample_rate=1.0)

        assert result["frequencies"].shape == (n,)
        assert result["magnitude"].shape == (n,)
        assert result["power"].shape == (n,)
        assert result["phase"].shape == (n,)
        assert result["fft"].shape == (n,)

    def test_dominant_frequency_detected(self):
        """Test that dominant frequency is correctly identified."""
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        freq_hz = 50.0
        signal = np.sin(2 * np.pi * freq_hz * t)

        result = signals.extract_frequency_spectrum(signal, sample_rate=fs)

        freqs = result["frequencies"]
        power = result["power"]
        # Find the peak frequency (positive side)
        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        pos_power = power[pos_mask]
        peak_freq = pos_freqs[np.argmax(pos_power)]
        assert abs(peak_freq - freq_hz) < 2.0  # within 2 Hz

    def test_empty_signal_raises(self):
        """Test that empty signal raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            signals.extract_frequency_spectrum(np.array([]))

    def test_invalid_axis_raises(self):
        """Test that invalid axis raises ValueError."""
        signal = np.random.randn(100)
        with pytest.raises(ValueError, match="out of bounds"):
            signals.extract_frequency_spectrum(signal, axis=5)

    def test_with_hann_window(self):
        """Test with Hann window."""
        signal = np.random.randn(100)
        result = signals.extract_frequency_spectrum(signal, window="hann")
        assert "power" in result
        assert result["power"].shape == (100,)

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        signal = np.random.randn(100)
        result = signals.extract_frequency_spectrum(signal, window="hamming")
        assert "power" in result

    def test_with_blackman_window(self):
        """Test with Blackman window."""
        signal = np.random.randn(100)
        result = signals.extract_frequency_spectrum(signal, window="blackman")
        assert "power" in result

    def test_with_detrend_constant(self):
        """Test with constant detrending."""
        signal = np.random.randn(100) + 5.0  # Add DC offset
        result = signals.extract_frequency_spectrum(signal, detrend="constant")
        assert "power" in result

    def test_with_detrend_linear(self):
        """Test with linear detrending."""
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t) + 2 * t  # Add linear trend
        result = signals.extract_frequency_spectrum(signal, detrend="linear")
        assert "power" in result

    def test_magnitude_scaling(self):
        """Test magnitude scaling option."""
        signal = np.random.randn(100)
        result = signals.extract_frequency_spectrum(signal, scaling="magnitude")
        assert "power" in result

    def test_2d_signal(self):
        """Test with a 2D signal (multiple channels)."""
        signal = np.random.randn(5, 100)
        result = signals.extract_frequency_spectrum(signal, sample_rate=100.0, axis=1)
        assert result["frequencies"].shape == (100,)
        assert result["magnitude"].shape == (5, 100)

    def test_sample_rate_affects_frequencies(self):
        """Test that sample rate affects the frequency axis."""
        n = 100
        signal = np.random.randn(n)
        result_1hz = signals.extract_frequency_spectrum(signal, sample_rate=1.0)
        result_100hz = signals.extract_frequency_spectrum(signal, sample_rate=100.0)

        # Nyquist frequency should differ
        assert result_1hz["frequencies"].max() < result_100hz["frequencies"].max()

    def test_fft_output_is_complex(self):
        """Test that FFT output is complex."""
        signal = np.random.randn(100)
        result = signals.extract_frequency_spectrum(signal)
        assert np.iscomplexobj(result["fft"])


class TestExtractAmplitudeSpectrum:
    """Test extract_amplitude_spectrum function."""

    def test_basic_return_types(self):
        """Test that function returns correct types."""
        signal = np.random.randn(100)
        freqs, amp = signals.extract_amplitude_spectrum(signal, sample_rate=100.0)

        assert isinstance(freqs, np.ndarray)
        assert isinstance(amp, np.ndarray)

    def test_positive_freqs_only(self):
        """Test that positive_freqs_only=True returns only non-negative frequencies."""
        signal = np.random.randn(100)
        freqs, amp = signals.extract_amplitude_spectrum(
            signal, sample_rate=100.0, positive_freqs_only=True
        )
        assert np.all(freqs >= 0)

    def test_all_freqs(self):
        """Test that positive_freqs_only=False returns all frequencies."""
        signal = np.random.randn(100)
        freqs_all, amp_all = signals.extract_amplitude_spectrum(
            signal, sample_rate=100.0, positive_freqs_only=False
        )
        freqs_pos, amp_pos = signals.extract_amplitude_spectrum(
            signal, sample_rate=100.0, positive_freqs_only=True
        )
        # All-freq version should have more elements
        assert len(freqs_all) > len(freqs_pos)

    def test_shapes_match(self):
        """Test that frequencies and amplitude have the same shape."""
        signal = np.random.randn(100)
        freqs, amp = signals.extract_amplitude_spectrum(signal, sample_rate=100.0)
        assert freqs.shape == amp.shape

    def test_amplitude_non_negative(self):
        """Test that amplitude values are non-negative."""
        signal = np.random.randn(100)
        freqs, amp = signals.extract_amplitude_spectrum(signal)
        assert np.all(amp >= 0)


class TestSpectrum:
    """Test spectrum function."""

    def test_basic_1d_signal(self):
        """Test with a basic 1D signal."""
        signal = np.random.randn(100)
        spe, freq = signals.spectrum(signal, dt=1.0)

        assert isinstance(spe, np.ndarray)
        assert isinstance(freq, np.ndarray)
        assert spe.shape == freq.shape

    def test_2d_signal(self):
        """Test with a 2D signal."""
        signal = np.random.randn(5, 100)
        spe, freq = signals.spectrum(signal, dt=1.0)

        assert isinstance(spe, np.ndarray)
        assert isinstance(freq, np.ndarray)

    def test_frequency_range(self):
        """Test that frequencies are in the expected range."""
        n = 100
        dt = 0.01  # 100 Hz sampling rate
        signal = np.random.randn(n)
        spe, freq = signals.spectrum(signal, dt=dt)

        # Nyquist frequency should be 1/(2*dt) = 50 Hz
        assert freq.max() <= 1 / (2 * dt) + 1e-10

    def test_dc_component_zeroed(self):
        """Test that DC component (index 0) is zeroed out for 1D."""
        signal = np.random.randn(100)
        spe, freq = signals.spectrum(signal)
        assert spe[0] == 0.0

    def test_dc_component_zeroed_2d(self):
        """Test that DC component is zeroed out for 2D."""
        signal = np.random.randn(3, 100)
        spe, freq = signals.spectrum(signal)
        # For 2D, first column should be zero
        np.testing.assert_array_equal(spe[:, 0], 0.0)

    def test_amplitude_non_negative(self):
        """Test that spectrum amplitudes are non-negative."""
        signal = np.random.randn(100)
        spe, freq = signals.spectrum(signal)
        assert np.all(spe >= 0)

    def test_dt_affects_frequency_scale(self):
        """Test that dt parameter affects the frequency scale."""
        signal = np.random.randn(100)
        _, freq1 = signals.spectrum(signal, dt=1.0)
        _, freq2 = signals.spectrum(signal, dt=0.1)

        # With smaller dt, max frequency should be larger
        assert freq2.max() > freq1.max()
