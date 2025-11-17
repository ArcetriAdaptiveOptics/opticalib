"""
TYPINGS module
==============
2025

Author(s)
---------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it

Description
-----------
This module defines custom type aliases and protocols for type hinting
within the `opticalib` package. It includes protocols for matrix-like
objects, image data, cube data, interferometer devices, and deformable
mirror devices. Additionally, it provides a custom `isinstance_` function
to check if an object conforms to these protocols.

"""

from typing import (
    Union,
    Optional,
    Any,
    TypeVar,
    TypeAlias,
    Callable,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)
import collections.abc
import numpy as _np
from numpy.typing import ArrayLike, DTypeLike
from astropy.io.fits import Header

if TYPE_CHECKING:
    from .ground.computerec import ComputeReconstructor

Reconstructor: TypeAlias = Union["ComputeReconstructor", None]

Number: TypeAlias = Union[int, float, complex]


@runtime_checkable
class _MatrixProtocol(Protocol):
    def shape(self) -> tuple[int, int]: ...
    def __getitem__(self, key: Any) -> Any: ...


@runtime_checkable
class _ImageDataProtocol(_MatrixProtocol, Protocol):
    def data(self) -> ArrayLike: ...
    def mask(self) -> ArrayLike: ...
    def __array__(self) -> ArrayLike: ...


@runtime_checkable
class _CubeProtocol(Protocol):
    def shape(self) -> tuple[int, int, int]: ...
    def data(self) -> ArrayLike: ...
    def mask(self) -> ArrayLike: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __array__(self) -> ArrayLike: ...


MatrixLike = TypeVar("MatrixLike", bound=_MatrixProtocol)
MaskData = TypeVar("MaskData", bound=_MatrixProtocol)
ImageData = TypeVar("ImageData", bound=_ImageDataProtocol)
CubeData = TypeVar("CubeData", bound=_CubeProtocol)


@runtime_checkable
class _InterfProtocol(Protocol):
    def acquire_map(
        self, nframes: int, delay: int | float, rebin: int
    ) -> ImageData: ...
    def capture(self, numberOfFrames: int, folder_name: str = None) -> str: ...
    def produce(self, tn: str): ...


InterferometerDevice = TypeVar("InterferometerDevice", bound=_InterfProtocol)


@runtime_checkable
class _DMProtocol(Protocol):
    @property
    def nActs(self) -> int: ...
    def set_shape(self, cmd: MatrixLike, differential: bool) -> None: ...
    def get_shape(self) -> ArrayLike: ...
    def uploadCmdHistory(self, cmdhist: MatrixLike) -> None: ...
    def runCmdHistory(
        self,
        interf: Optional[InterferometerDevice],
        delay: int | float,
        save: Optional[str],
        differential: bool,
    ) -> str: ...


DeformableMirrorDevice = TypeVar("DeformableMirrorDevice", bound=_DMProtocol)

GenericDevice = TypeVar("GenericDevice")


def array_str_formatter(array: ArrayLike | list[ArrayLike]) -> str | list[str]:
    """
    Formats an array-like object into a string representation.

    Parameters
    ----------
    arr : ArrayLike os list[ArrayLike]
        The array-like object to be formatted.

    Returns
    -------
    array_strs : str
        The string representation of the array(s).
    """
    if isinstance(array, list):
        if not all([isinstance(l, _np.ndarray) for l in array]):
            array = [_np.array(l) for l in array]
    else:
        array = [array]
    array_strs = []
    for arr in array:
        if arr.dtype == int:
            separator = ","
        else:
            separator = ", "
        if any([a >= 1e3 for a in arr]) or any([a <= 1e-3 for a in arr]):
            array_strs.append(
                _np.array2string(
                    arr,
                    separator=separator,
                    precision=2,
                    formatter={"float_kind": lambda x: f"{x:.2e}"},
                )
            )
        else:
            array_strs.append(
                _np.array2string(
                    arr,
                    separator=separator,
                    precision=3,
                    formatter={"float_kind": lambda x: f"{x:.2f}"},
                )
            )
    return array_strs[0] if len(array_strs) == 1 else array_strs


################################
## Custom `isinstance` checks ##
################################


class InstanceCheck:
    """
    A class to check if an object is an instance of a specific type.
    """

    @staticmethod
    def is_matrix_like(obj: Any) -> bool:
        """
        Check if the object is a matrix-like object.
        Returns True if obj is a 2D matrix-like object, otherwise False.
        """
        if not isinstance(obj, _ImageDataProtocol):
            if isinstance(obj, _MatrixProtocol):
                if isinstance(obj, _np.ndarray) and obj.ndim == 2:
                    return True
                if isinstance(obj, collections.abc.Sequence):
                    try:
                        first_row = obj[0]
                    except (IndexError, TypeError):
                        return False
                    if not isinstance(first_row, collections.abc.Sequence):
                        return False
                    row_len = len(first_row)
                    return all(
                        isinstance(row, collections.abc.Sequence)
                        and len(row) == row_len
                        for row in obj
                    )
        return False

    @staticmethod
    def is_mask_like(obj: Any) -> bool:
        """
        Check if the object is a mask-like object.
        Returns True if obj is a 2D mask-like object, otherwise False.
        """
        if not isinstance(obj, _MatrixProtocol):
            return False
        try:
            shape = obj.shape
        except Exception:
            return False
        # Ensure shape is a tuple of length 2
        if not (isinstance(shape, tuple) and len(shape) == 2):
            return False
        if not any(
            [
                obj.dtype.type == _np.bool_,
                obj.dtype.type == _np.uint8,
                obj.dtype.type == _np.int_,
            ]
        ):
            return False
        if not _np.sum(obj) <= shape[0] * shape[1]:
            return False
        return True

    @staticmethod
    def is_image_like(obj: Any, ndim: int = 2) -> bool:
        """
        Check if the object is an image-like object.
        Returns True if obj is a 2D image ArrayLike object with a mask,
        otherwise False.
        """
        if not isinstance(obj, _ImageDataProtocol):
            return False
        try:
            shape = obj.shape
            mask = obj.mask
            data = obj.data
        except Exception:
            return False
        # Ensure shape is a tuple of length ndim (default 2)
        if not (isinstance(shape, tuple) and len(shape) == ndim):
            return False
        # Check mask shape
        if hasattr(mask, "shape"):
            mask_shape = mask.shape if not callable(mask.shape) else mask.shape()
            if mask_shape != shape:
                return False
        else:
            try:
                if len(mask) != shape[0]:
                    return False
                if any(len(row) != shape[1] for row in mask):
                    return False
            except Exception:
                return False
        # Check data shape
        if hasattr(data, "shape"):
            data_shape = data.shape if not callable(data.shape) else data.shape()
            if data_shape != shape:
                return False
        else:
            try:
                if len(data) != shape[0]:
                    return False
                if any(len(row) != shape[1] for row in data):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def is_cube_like(obj: Any) -> bool:
        """
        Check if the object is a cube-like object.
        Returns True if obj is a 3D cube ArrayLike object with a mask,
        otherwise False.
        """
        return InstanceCheck.is_image_like(obj, ndim=3)

    @staticmethod
    def generic_check(obj: Any, class_name: str) -> bool:
        """
        Generic check for any object type.
        Returns True if obj is an instance of the specified class, otherwise False.
        """
        generic_class_map = {
            "DeformableMirrorDevice": _DMProtocol,
            "InterferometerDevice": _InterfProtocol,
        }
        if class_name not in generic_class_map:
            raise ValueError(f"Class {class_name} not found in the current context.")
        return isinstance(obj, generic_class_map[class_name])

    @classmethod
    def isinstance_(cls, obj: Any, class_name: str) -> bool:
        """
        Custom `isinstance` wrapper: checks if the object is an instance of a
        specific class.

        Parameters
        ----------
        class_name: str
            The name of the class to check against.

        obj: Any
            The object to check.

        Returns
        -------
        bool
            True if obj is an instance of the specified class, otherwise False.
        """
        checks: dict[str, Callable[..., bool]] = {
            "MatrixLike": cls.is_matrix_like,
            "MaskData": cls.is_mask_like,
            "ImageData": cls.is_image_like,
            "CubeData": cls.is_cube_like,
            "InterferometerDevice": cls.generic_check,
            "DeformableMirrorDevice": cls.generic_check,
        }
        if class_name not in checks:
            raise ValueError(f"Unknown class name: {class_name}")
        try:
            check = checks[class_name](obj)
        except TypeError:
            check = checks[class_name](obj, class_name)
        return check


isinstance_ = InstanceCheck.isinstance_
