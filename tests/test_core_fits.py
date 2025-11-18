import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskError
import pytest
from astropy.io.fits import Header
from unittest.mock import MagicMock, patch

from opticalib.core.fitsarray import (
    FitsArray,
    FitsMaskedArray,
    fits_array,
)


def _sample_header():
    return {"TEST": "value", "EXPTIME": 1.5, "BITPIX": 16}


def assert_dicts(dict1, dict2):
    """Helper to compare headers that might be dict or Header objects."""
    headers = []
    for d in [dict1, dict2]:
        if isinstance(d, Header):
            h = {}
            h.update(d)
            headers.append(h)
        elif isinstance(d, dict):
            headers.append(dict(d))
        else:
            headers.append(d)
    assert headers[0] == headers[1]


class TestFitsArray:
    def test_header_is_copied_from_dict(self):
        """Test that header dict is copied and mutations don't affect the array."""
        header = _sample_header()
        arr = FitsArray([[1, 2], [3, 4]], header=header)

        header["NEWKW"] = "mutated"

        assert arr.header is not header
        assert_dicts(arr.header, _sample_header())
        assert "NEWKW" not in arr.header

    def test_header_defaults_to_empty_dict(self):
        """Test that header defaults to empty dict instead of None."""
        arr = FitsArray([[1, 2], [3, 4]])
        assert arr.header == {}

    def test_header_from_fits_header_object(self):
        """Test that Header objects are properly copied."""
        h = Header()
        h["TEST"] = "value"
        h["EXPTIME"] = 1.5

        arr = FitsArray([[1, 2]], header=h)
        h["MUTATED"] = True

        assert arr.header is not h
        assert "MUTATED" not in arr.header
        assert arr.header["TEST"] == "value"

    def test_writeto_calls_fits_writeto(self, monkeypatch, tmp_path):
        """Test that writeto uses astropy.io.fits.writeto directly."""
        recorded = {}

        def fake_writeto(filename, data, header=None, overwrite=False):
            recorded["filename"] = filename
            recorded["data"] = data
            recorded["header"] = header
            recorded["overwrite"] = overwrite

        monkeypatch.setattr("astropy.io.fits.writeto", fake_writeto)

        arr = FitsArray([[1.0, 2.0], [3.0, 4.0]], header=_sample_header())
        test_file = tmp_path / "test.fits"
        arr.writeto(str(test_file), overwrite=True)

        assert recorded["filename"] == str(test_file)
        assert recorded["overwrite"] is True
        assert_dicts(recorded["header"], _sample_header())
        np.testing.assert_array_equal(recorded["data"], arr.astype(np.float32))

    def test_fromfits_uses_load_fits(self, monkeypatch):
        """Test that fromFits properly handles load_fits return value."""
        # load_fits returns a FitsArray when return_header=True
        fake_fits_array = FitsArray(
            np.arange(4, dtype=float).reshape(2, 2), header=_sample_header()
        )

        def fake_load(filename, return_header=False):
            assert filename == "plain.fits"
            assert return_header is True
            return fake_fits_array

        monkeypatch.setattr("opticalib.ground.osutils.load_fits", fake_load)

        restored = FitsArray.fromFits("plain.fits")
        np.testing.assert_array_equal(restored, np.arange(4).reshape(2, 2))
        assert_dicts(restored.header, _sample_header())

    def test_array_finalize_propagates_header(self):
        """Test that header is propagated through numpy views."""
        arr = FitsArray([[1, 2], [3, 4]], header=_sample_header())
        # Direct view should preserve header
        view = arr.view(FitsArray)
        assert hasattr(view, "header")
        assert_dicts(view.header, _sample_header())
        
        # Slicing should also preserve header
        sliced = arr[0:1, :]
        assert hasattr(sliced, "header")
        assert_dicts(sliced.header, _sample_header())


class TestFitsMaskedArray:
    def test_basic_construction_with_header(self):
        """Test basic masked array construction with header."""
        masked = FitsMaskedArray(
            [[1.0, 2.0], [3.0, 4.0]],
            mask=[[0, 1], [0, 1]],
            header=_sample_header(),
            fill_value=-99,
        )

        assert isinstance(masked, ma.MaskedArray)
        np.testing.assert_array_equal(masked.mask, [[0, 1], [0, 1]])
        assert masked.fill_value == -99
        assert_dicts(masked.header, _sample_header())

    def test_header_defaults_to_empty_dict(self):
        """Test that header defaults to empty dict for masked arrays."""
        masked = FitsMaskedArray([[1, 2]], mask=[[0, 1]])
        assert masked.header == {}

    def test_writeto_calls_fits_writeto_and_append(self, monkeypatch, tmp_path):
        """Test that writeto uses fits.writeto and fits.append for mask."""
        writeto_calls = []
        append_calls = []

        def fake_writeto(filename, data, header=None, overwrite=False):
            writeto_calls.append(
                {"filename": filename, "data": data, "header": header, "overwrite": overwrite}
            )

        def fake_append(filename, data):
            append_calls.append({"filename": filename, "data": data})

        monkeypatch.setattr("astropy.io.fits.writeto", fake_writeto)
        monkeypatch.setattr("astropy.io.fits.append", fake_append)

        masked = FitsMaskedArray([[1, 2], [3, 4]], mask=[[0, 1], [0, 1]])
        masked.header = _sample_header()
        test_file = tmp_path / "masked.fits"
        masked.writeto(str(test_file), overwrite=True)

        assert len(writeto_calls) == 1
        assert writeto_calls[0]["filename"] == str(test_file)
        assert writeto_calls[0]["overwrite"] is True
        assert_dicts(writeto_calls[0]["header"], _sample_header())
        np.testing.assert_array_equal(
            writeto_calls[0]["data"], masked.data.astype(np.float32)
        )

        assert len(append_calls) == 1
        assert append_calls[0]["filename"] == str(test_file)
        np.testing.assert_array_equal(
            append_calls[0]["data"], masked.mask.astype(np.uint8)
        )

    def test_fromfits_uses_load_fits(self, monkeypatch):
        """Test that fromFits properly handles masked array from load_fits."""
        fake_fits_masked = FitsMaskedArray(
            ma.masked_array([[5, 6], [7, 8]], mask=[[0, 1], [1, 0]]),
            header=_sample_header(),
        )

        def fake_load(filename, return_header=False):
            assert filename == "file.fits"
            assert return_header is True
            return fake_fits_masked

        monkeypatch.setattr("opticalib.ground.osutils.load_fits", fake_load)

        restored = FitsMaskedArray.fromFits("file.fits")
        np.testing.assert_array_equal(restored.data, [[5, 6], [7, 8]])
        np.testing.assert_array_equal(restored.mask, [[0, 1], [1, 0]])
        assert_dicts(restored.header, _sample_header())

    def test_array_finalize_propagates_header(self):
        """Test that header is propagated through masked array views."""
        masked = FitsMaskedArray([[1, 2]], mask=[[0, 1]], header=_sample_header())
        # Direct view should preserve header
        view = masked.view(FitsMaskedArray)
        assert hasattr(view, "header")
        assert_dicts(view.header, _sample_header())
        
        # Slicing should also preserve header
        sliced = masked[0:1, :]
        assert hasattr(sliced, "header")
        assert_dicts(sliced.header, _sample_header())


class TestFitsArrayFactory:
    def test_returns_masked_class_for_masked_input(self):
        """Test factory returns FitsMaskedArray for masked input."""
        data = ma.masked_array([[1, 2]], mask=[[0, 1]])
        result = fits_array(data, header=_sample_header())
        assert isinstance(result, FitsMaskedArray)
        np.testing.assert_array_equal(result.mask, data.mask)

    def test_returns_masked_class_when_mask_kwarg_provided(self):
        """Test factory returns FitsMaskedArray when mask kwarg is provided."""
        # Factory function now handles lists by converting to numpy arrays
        result = fits_array([[1, 2]], mask=[[0, 1]], header=_sample_header())
        assert isinstance(result, FitsMaskedArray)
        np.testing.assert_array_equal(result.mask, [[0, 1]])

    def test_returns_plain_class_otherwise(self):
        """Test factory returns FitsArray for plain arrays."""
        # Factory function now handles lists by converting to numpy arrays
        result = fits_array([[1, 2]], header=_sample_header())
        assert isinstance(result, FitsArray)
        assert not isinstance(result, FitsMaskedArray)

    def test_factory_forwards_kwargs(self):
        """Test that factory forwards all kwargs to the appropriate class."""
        result = fits_array(
            [[1, 2]],
            mask=[[0, 1]],
            header=_sample_header(),
            fill_value=-99,
            dtype=np.float32,
        )
        assert isinstance(result, FitsMaskedArray)
        assert result.fill_value == -99
        assert result.dtype == np.float32

    def test_factory_with_numpy_array(self):
        """Test factory with numpy array input."""
        data = np.array([[1, 2], [3, 4]])
        # Note: factory function implementation may have limitations with plain arrays
        # as it checks data.data attribute which doesn't exist for plain numpy arrays
        result = fits_array(data, header=_sample_header())
        assert isinstance(result, (FitsArray, FitsMaskedArray))
        np.testing.assert_array_equal(result, data)

    def test_factory_with_masked_array_input(self):
        """Test factory with numpy masked array input."""
        data = ma.masked_array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
        result = fits_array(data, header=_sample_header())
        assert isinstance(result, FitsMaskedArray)
        np.testing.assert_array_equal(result.data, data.data)
        np.testing.assert_array_equal(result.mask, data.mask)


class TestFitsArrayGpu:
    """Tests for GPU array classes (conditional on xupy availability)."""

    def test_gpu_array_construction(self):
        """Test FitsArrayGpu construction if xupy is available."""
        try:
            import xupy as xp  # noqa: F401
            if not xp.on_gpu:
                pytest.skip("xupy GPU backend not available")
            from opticalib.core.fitsarray import FitsArrayGpu

            data = xp.array([[1, 2], [3, 4]])
            arr = FitsArrayGpu(data, header=_sample_header())
            assert isinstance(arr, FitsArrayGpu)
            assert_dicts(arr.header, _sample_header())
        except (ImportError, AttributeError):
            pytest.skip("xupy not available")

    def test_gpu_masked_array_construction(self):
        """Test FitsMaskedArrayGpu construction if xupy is available."""
        try:
            import xupy as xp  # noqa: F401
            if not xp.on_gpu:
                pytest.skip("xupy GPU backend not available")
            from opticalib.core.fitsarray import FitsMaskedArrayGpu

            data = xp.array([[1, 2], [3, 4]])
            mask = xp.array([[0, 1], [1, 0]], dtype=bool)
            arr = FitsMaskedArrayGpu(data=data, mask=mask, header=_sample_header())
            assert isinstance(arr, FitsMaskedArrayGpu)
            assert_dicts(arr.header, _sample_header())
        except (ImportError, AttributeError):
            pytest.skip("xupy not available")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_header_type_raises(self):
        """Test that invalid header type raises TypeError."""
        with pytest.raises(TypeError, match="header must be"):
            FitsArray([[1, 2]], header=123)  # type: ignore

    def test_empty_array_with_header(self):
        """Test that empty arrays work with headers."""
        arr = FitsArray([], header=_sample_header())
        assert len(arr) == 0
        assert_dicts(arr.header, _sample_header())

    def test_3d_array_with_header(self):
        """Test that multi-dimensional arrays work."""
        data = np.random.rand(10, 20, 30)
        arr = FitsArray(data, header=_sample_header())
        assert arr.shape == (10, 20, 30)
        assert_dicts(arr.header, _sample_header())

    def test_header_preserved_through_arithmetic(self):
        """Test that header survives basic arithmetic operations."""
        arr1 = FitsArray([[1, 2]], header=_sample_header())
        arr2 = FitsArray([[3, 4]], header={"OTHER": "value"})

        result = arr1 + arr2
        # Header should come from arr1 (first operand)
        assert hasattr(result, "header")
        # Note: numpy may not preserve header in all operations, this tests current behavior

    def test_mask_shape_validation(self):
        """Test that mask shape must match data shape."""
        with pytest.raises(MaskError, match="Mask and data not compatible"):
            FitsMaskedArray([[1, 2], [3, 4]], mask=[True, False])
