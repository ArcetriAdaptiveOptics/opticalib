"""
Tests for opticalib.core.exceptions module.
"""

import pytest
from opticalib.core.exceptions import (
    DeviceNotFoundError,
    DeviceError,
    DeviceAttributeError,
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
        message = f"Device '{device_name}' not found in the configuration file."
        # The raw message is preserved in args
        assert error.args[0] == message
        # KeyError.__str__ returns the repr of its single argument
        assert str(error) == repr(message)

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


class TestDeviceAttributeError:
    """Test DeviceAttributeError exception."""

    def test_device_attribute_error_creation(self):
        """Test that DeviceAttributeError can be created with message only."""
        error = DeviceAttributeError("Missing attribute")
        assert isinstance(error, Exception)
        assert isinstance(error, DeviceAttributeError)

    def test_device_attribute_error_inheritance(self):
        """Test that DeviceAttributeError inherits from AttributeError."""
        error = DeviceAttributeError("Missing attribute")
        assert isinstance(error, AttributeError)
        assert not isinstance(error, KeyError)

    def test_device_attribute_error_message(self):
        """Test that the message is returned by str()."""
        message = "Device has no attribute 'foo'"
        error = DeviceAttributeError(message)
        assert str(error) == message
        assert error.args == (message,)

    def test_device_attribute_error_defaults(self):
        """Test that name and obj default to None."""
        error = DeviceAttributeError("Missing attribute")
        assert error.name is None
        assert error.obj is None

    def test_device_attribute_error_with_attribute(self):
        """Test that the attribute name is stored in `name`."""
        error = DeviceAttributeError("Missing attribute", attribute="foo")
        assert error.name == "foo"
        assert error.obj is None
        assert str(error) == "Missing attribute"

    def test_device_attribute_error_with_obj(self):
        """Test that the object is stored in `obj`."""
        device = object()
        error = DeviceAttributeError("Missing attribute", obj=device)
        assert error.obj is device
        assert error.name is None

    def test_device_attribute_error_with_attribute_and_obj(self):
        """Test passing both attribute and obj, positionally and by keyword."""
        device = object()
        by_kw = DeviceAttributeError("Missing", attribute="foo", obj=device)
        by_pos = DeviceAttributeError("Missing", "foo", device)
        for error in (by_kw, by_pos):
            assert error.name == "foo"
            assert error.obj is device
            assert str(error) == "Missing"

    def test_raise_and_catch_as_attribute_error(self):
        """Test that it can be caught as a built-in AttributeError."""
        device = object()
        with pytest.raises(AttributeError) as exc_info:
            raise DeviceAttributeError("Missing", attribute="foo", obj=device)
        assert isinstance(exc_info.value, DeviceAttributeError)
        assert exc_info.value.name == "foo"
        assert exc_info.value.obj is device

    def test_getattr_default_catches_it(self):
        """Test that getattr/hasattr treat it as an AttributeError."""

        class Dev:
            def __getattr__(self, item: str):
                raise DeviceAttributeError(f"no {item}", attribute=item, obj=self)

        dev = Dev()
        assert getattr(dev, "foo", "default") == "default"
        assert not hasattr(dev, "foo")


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
