"""
Tests for opticalib.core.exceptions module.
"""
import pytest
from opticalib.core.exceptions import (
    DeviceNotFoundError,
    DeviceError,
    MatrixError,
    CommandError,
)


class TestDeviceNotFoundError:
    """Test DeviceNotFoundError exception."""

    def test_device_not_found_error_creation(self):
        """Test that DeviceNotFoundError can be created."""
        error = DeviceNotFoundError("TestDevice")
        assert isinstance(error, Exception)
        assert isinstance(error, DeviceNotFoundError)

    def test_device_not_found_error_message(self):
        """Test that DeviceNotFoundError has correct message."""
        device_name = "TestDevice"
        error = DeviceNotFoundError(device_name)
        assert str(error) == f"Device '{device_name}' not found in the configuration file."

    def test_device_not_found_error_inheritance(self):
        """Test that DeviceNotFoundError inherits from Exception."""
        error = DeviceNotFoundError("TestDevice")
        assert isinstance(error, Exception)


class TestDeviceError:
    """Test DeviceError exception."""

    def test_device_error_creation(self):
        """Test that DeviceError can be created."""
        error = DeviceError("TestDevice", "DM")
        assert isinstance(error, Exception)
        assert isinstance(error, DeviceError)

    def test_device_error_message(self):
        """Test that DeviceError has correct message."""
        device_name = "TestDevice"
        device_type = "DM"
        error = DeviceError(device_name, device_type)
        assert str(error) == f"Device '{device_name}' is not a valid {device_type}."

    def test_device_error_inheritance(self):
        """Test that DeviceError inherits from Exception."""
        error = DeviceError("TestDevice", "DM")
        assert isinstance(error, Exception)


class TestMatrixError:
    """Test MatrixError exception."""

    def test_matrix_error_creation(self):
        """Test that MatrixError can be created."""
        error = MatrixError("Test message")
        assert isinstance(error, Exception)
        assert isinstance(error, MatrixError)

    def test_matrix_error_message(self):
        """Test that MatrixError has correct message."""
        message = "Matrix is singular"
        error = MatrixError(message)
        assert str(error) == message

    def test_matrix_error_inheritance(self):
        """Test that MatrixError inherits from Exception."""
        error = MatrixError("Test message")
        assert isinstance(error, Exception)


class TestCommandError:
    """Test CommandError exception."""

    def test_command_error_creation(self):
        """Test that CommandError can be created."""
        error = CommandError("Test message")
        assert isinstance(error, Exception)
        assert isinstance(error, CommandError)

    def test_command_error_message(self):
        """Test that CommandError has correct message."""
        message = "Invalid command"
        error = CommandError(message)
        assert str(error) == message

    def test_command_error_inheritance(self):
        """Test that CommandError inherits from Exception."""
        error = CommandError("Test message")
        assert isinstance(error, Exception)


class TestExceptionUsage:
    """Test that exceptions can be raised and caught."""

    def test_raise_device_not_found_error(self):
        """Test raising DeviceNotFoundError."""
        with pytest.raises(DeviceNotFoundError) as exc_info:
            raise DeviceNotFoundError("TestDevice")
        assert "TestDevice" in str(exc_info.value)

    def test_raise_device_error(self):
        """Test raising DeviceError."""
        with pytest.raises(DeviceError) as exc_info:
            raise DeviceError("TestDevice", "DM")
        assert "TestDevice" in str(exc_info.value)
        assert "DM" in str(exc_info.value)

    def test_raise_matrix_error(self):
        """Test raising MatrixError."""
        with pytest.raises(MatrixError) as exc_info:
            raise MatrixError("Matrix is singular")
        assert "Matrix is singular" in str(exc_info.value)

    def test_raise_command_error(self):
        """Test raising CommandError."""
        with pytest.raises(CommandError) as exc_info:
            raise CommandError("Invalid command")
        assert "Invalid command" in str(exc_info.value)

