"""
Custom exceptions — OptiCalib error hierarchy
=============================================

Exception classes for predictable error conditions: missing devices,
failed configuration lookups, singular matrices, timeouts, and invalid
command vectors.  All inherit from Python's built-in :class:`Exception`
so they can be caught with a single ``except opticalib.core.exceptions.*``
clause if needed.

"""


class DeviceNotFoundError(Exception):
    """Exception raised when a device is not found in the
    configuration file.
    """

    def __init__(self, device_name: str):
        super().__init__(f"Device '{device_name}' not found in the configuration file.")


class DeviceError(Exception):
    """Exception raised when a device is not found in the
    configuration file.
    """

    def __init__(self, device_name: str, device_type: str):
        super().__init__(f"Device '{device_name}' is not a valid {device_type}.")


class DeviceAttributeError(Exception):
    """Exception raised when a device attribute is missing or invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class MatrixError(Exception):
    """Exception raised when a matrix is not valid."""

    def __init__(self, message: str):
        super().__init__(message)


class CommandError(Exception):
    """Exception raised when a command is not valid."""

    def __init__(self, message: str):
        super().__init__(message)


class BufferError(Exception):
    """Exception raised when a buffer operation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class PathError(Exception):
    """Exception raised when a path operation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class ReconnectionError(Exception):
    """
    Exception raised when a device reconnection fails after multiple
    attempts.
    """

    def __init__(self, message: str):
        super().__init__(message)
