"""
This module defines custom exceptions used in the opticalib system.
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
