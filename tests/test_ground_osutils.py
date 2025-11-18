"""
Tests for opticalib.ground.osutils module.
"""
import pytest
import os
import tempfile
import shutil
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from opticalib.ground import osutils
from opticalib.core import fitsarray as _fa


class TestIsTn:
    """Test is_tn function."""

    def test_is_tn_valid(self):
        """Test with valid tracking number."""
        # Generate a valid tracking number
        from opticalib.ground.osutils import newtn
        tn = newtn()
        assert osutils.is_tn(tn) is True

    def test_is_tn_invalid_length(self):
        """Test with invalid length."""
        assert osutils.is_tn("12345678_12345") is False  # Too short
        assert osutils.is_tn("12345678901234567890") is False  # Too long

    def test_is_tn_invalid_format(self):
        """Test with invalid format."""
        assert osutils.is_tn("12345678-123456") is False  # Wrong separator
        assert osutils.is_tn("abcdefgh_ijklmn") is False  # Non-numeric

    def test_is_tn_invalid_date(self):
        """Test with invalid date."""
        assert osutils.is_tn("20251301_120000") is False  # Invalid month
        assert osutils.is_tn("20240230_120000") is False  # Invalid day


class TestNewtn:
    """Test newtn function."""

    def test_newtn_format(self):
        """Test that newtn returns correct format."""
        tn = osutils.newtn()
        assert len(tn) == 15
        assert tn[8] == "_"
        assert osutils.is_tn(tn) is True

    def test_newtn_unique(self):
        """Test that newtn generates unique values."""
        tn1 = osutils.newtn()
        import time
        time.sleep(1.1)  # Sleep for more than 1 second to ensure different timestamp
        tn2 = osutils.newtn()
        assert tn1 != tn2


class TestFindTracknum:
    """Test findTracknum function."""

    def test_find_tracknum_not_found(self, temp_dir, monkeypatch):
        """Test finding tracking number that doesn't exist."""
        monkeypatch.setattr(osutils, "_OPTDATA", temp_dir)
        
        # Create some folders but not the tracking number
        os.makedirs(os.path.join(temp_dir, "OPDImages"), exist_ok=True)
        
        result = osutils.findTracknum("20240101_120000")
        assert result == []

    def test_find_tracknum_found(self, temp_dir, monkeypatch):
        """Test finding tracking number."""
        monkeypatch.setattr(osutils, "_OPTDATA", temp_dir)
        
        # Create folder structure with tracking number
        tn = "20240101_120000"
        folder = os.path.join(temp_dir, "OPDImages")
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, tn), exist_ok=True)
        
        result = osutils.findTracknum(tn)
        assert result == "OPDImages"

    def test_find_tracknum_complete_path(self, temp_dir, monkeypatch):
        """Test finding tracking number with complete path."""
        monkeypatch.setattr(osutils, "_OPTDATA", temp_dir)
        
        tn = "20240101_120000"
        folder = os.path.join(temp_dir, "OPDImages")
        os.makedirs(folder, exist_ok=True)
        tn_path = os.path.join(folder, tn)
        os.makedirs(tn_path, exist_ok=True)
        
        result = osutils.findTracknum(tn, complete_path=True)
        assert result == tn_path


class TestGetFileList:
    """Test getFileList function."""

    def test_get_file_list_from_folder(self, temp_dir):
        """Test getting file list from folder."""
        # Create test files
        test_files = ["file1.fits", "file2.fits", "file3.fits"]
        for fname in test_files:
            with open(os.path.join(temp_dir, fname), "w") as f:
                f.write("test")
        
        file_list = osutils.getFileList(fold=temp_dir)
        # Filter out directories from the list
        file_list = [f for f in file_list if os.path.isfile(f)]
        assert len(file_list) == 3
        for fname in test_files:
            assert any(fname in f for f in file_list)

    def test_get_file_list_with_key(self, temp_dir):
        """Test getting file list with key filter."""
        # Create test files
        test_files = ["mode_0000.fits", "mode_0001.fits", "other.fits"]
        for fname in test_files:
            with open(os.path.join(temp_dir, fname), "w") as f:
                f.write("test")
        
        file_list = osutils.getFileList(fold=temp_dir, key="mode_")
        assert len(file_list) == 2
        assert all("mode_" in f for f in file_list)


class TestLoadFits:
    """Test load_fits function."""

    def test_load_fits_basic(self, temp_dir):
        """Test loading basic FITS file."""
        # Create a test FITS file
        data = np.random.randn(50, 50).astype(np.float32)
        fits_file = os.path.join(temp_dir, "test.fits")
        fits.writeto(fits_file, data)
        
        loaded = osutils.load_fits(fits_file)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_load_fits_with_mask(self, temp_dir):
        """Test loading FITS file with mask."""
        # Create a test FITS file with mask
        data = np.random.randn(50, 50).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        mask[:10, :10] = True
        
        fits_file = os.path.join(temp_dir, "test_masked.fits")
        fits.writeto(fits_file, data)
        fits.append(fits_file, mask.astype(np.uint8))
        
        loaded = osutils.load_fits(fits_file)
        assert isinstance(loaded, ma.MaskedArray)
        np.testing.assert_array_almost_equal(loaded.data, data)
        np.testing.assert_array_equal(loaded.mask, mask)

    def test_load_fits_with_header(self, temp_dir):
        """Test loading FITS file with header."""
        data = np.random.randn(50, 50).astype(np.float32)
        header = fits.Header()
        header['TESTKEY'] = 'testvalue'
        fits_file = os.path.join(temp_dir, "test_header.fits")
        fits.writeto(fits_file, data, header=header)

        result = osutils.load_fits(fits_file, return_header=True)
        assert isinstance(result, _fa.FitsArray)
        assert 'TESTKEY' in result.header
        assert result.header['TESTKEY'] == 'testvalue'

class TestSaveFits:
    """Test save_fits function."""

    def test_save_fits_basic(self, temp_dir):
        """Test saving basic FITS file."""
        data = np.random.randn(50, 50).astype(np.float32)
        fits_file = os.path.join(temp_dir, "test_save.fits")
        
        osutils.save_fits(fits_file, data)
        
        assert os.path.exists(fits_file)
        loaded = fits.getdata(fits_file)
        np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_save_fits_with_mask(self, temp_dir):
        """Test saving FITS file with mask."""
        data = np.random.randn(50, 50).astype(np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        mask[:10, :10] = True
        masked_data = ma.masked_array(data, mask=mask)
        
        fits_file = os.path.join(temp_dir, "test_save_masked.fits")
        osutils.save_fits(fits_file, masked_data)
        
        assert os.path.exists(fits_file)
        loaded = osutils.load_fits(fits_file)
        assert isinstance(loaded, ma.MaskedArray)
        np.testing.assert_array_almost_equal(loaded.data, data, decimal=5)

    def test_save_fits_with_header(self, temp_dir):
        """Test saving FITS file with header."""
        data = np.random.randn(50, 50).astype(np.float32)
        header = {'TESTKEY': 'testvalue'}
        fits_file = os.path.join(temp_dir, "test_save_header.fits")
        
        osutils.save_fits(fits_file, data, header=header)
        
        # Verify header exists by reading directly with astropy
        with fits.open(fits_file) as hdul:
            assert 'TESTKEY' in hdul[0].header
            assert hdul[0].header['TESTKEY'] == 'testvalue'
        
        # Also verify data loads correctly
        loaded = osutils.load_fits(fits_file, return_header=False)
        assert loaded is not None

    def test_save_fits_overwrite(self, temp_dir):
        """Test saving FITS file with overwrite."""
        data1 = np.random.randn(50, 50).astype(np.float32)
        data2 = np.random.randn(50, 50).astype(np.float32)
        fits_file = os.path.join(temp_dir, "test_overwrite.fits")
        
        osutils.save_fits(fits_file, data1)
        osutils.save_fits(fits_file, data2, overwrite=True)
        
        loaded = osutils.load_fits(fits_file)
        np.testing.assert_array_almost_equal(loaded, data2, decimal=5)


class TestHeaderFromDict:
    """Test _header_from_dict function."""

    def test_header_from_dict_basic(self):
        """Test converting dictionary to FITS header."""
        dict_header = {
            'KEY1': 'value1',
            'KEY2': 42,
            'KEY3': 3.14
        }
        
        header = osutils._header_from_dict(dict_header)
        assert isinstance(header, fits.Header)
        assert header['KEY1'] == 'value1'
        assert header['KEY2'] == 42
        assert header['KEY3'] == 3.14

    def test_header_from_dict_with_comment(self):
        """Test converting dictionary with comments to FITS header."""
        dict_header = {
            'KEY1': ('value1', 'comment1'),
            'KEY2': (42, 'comment2')
        }
        
        header = osutils._header_from_dict(dict_header)
        assert header['KEY1'] == 'value1'
        assert header['KEY2'] == 42

    def test_header_from_dict_already_header(self):
        """Test passing already a FITS header."""
        original_header = fits.Header()
        original_header['KEY1'] = 'value1'
        
        header = osutils._header_from_dict(original_header)
        assert header is original_header


class TestEnsureOnCpu:
    """Test _ensure_on_cpu function."""

    def test_ensure_on_cpu_numpy_array(self):
        """Test with numpy array."""
        data = np.random.randn(50, 50)
        result = osutils._ensure_on_cpu(data)
        assert result is data

    def test_ensure_on_cpu_masked_array(self):
        """Test with numpy masked array."""
        data = ma.masked_array(np.random.randn(50, 50))
        result = osutils._ensure_on_cpu(data)
        assert result is data

