"""
Type System — structural type aliases and device protocols for OptiCalib
========================================================================

OptiCalib is deliberately *hardware agnostic*.  Instead of coupling
algorithms to concrete classes, this module defines **structural type
aliases** and **Protocol classes** that describe the *shape* and
*capabilities* an object must have.

Two complementary layers:

**Data aliases** — ``MatrixLike``, ``MaskData``, ``ImageData``,
``CubeData``, ``FitsData`` — capture what array-shaped payloads look like
(dimensionality, presence of a mask, FITS serialisability).  Every alias
is a :class:`~typing.TypeVar` bound to a :class:`~typing.Protocol`, so
static type checkers verify compatibility without any inheritance
relationship.

**Device protocols** — ``InterferometerDevice``, ``WFSDevice``,
``CameraDevice``, ``DeformableMirrorDevice`` and their ``Fake*``
counterparts — specify the *methods* a device must expose.  Any object
that structurally matches the protocol can be passed where the alias is
expected, whether it is a real driver, a simulator, or a test double.

The module is re-exported publicly as ``opticalib.typings``.

Because the aliases are structural (not nominal), ``isinstance(x,
ImageData)`` does **not** work.  Use :func:`isinstance_` instead — it
dispatches to the correct runtime check for the given name.

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
    from ..ground.reconstructor import ComputeReconstructor

#################################
## DATA TYPES AND TYPE ALIASES ##
#################################
#: A reconstructor object used to convert raw interferometer frames into
#: phase maps, or ``None`` when no reconstruction is needed.
Reconstructor: TypeAlias = Union["ComputeReconstructor", None]
#: Any plain Python number, i.e. an ``int``, a ``float`` or a ``complex``.
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


class _FitsArrayProtocol(_MatrixProtocol, Protocol):
    def writeto(
        self,
        filename: str,
        overwrite: bool = False,
    ) -> None: ...
    @classmethod
    def fromfits(
        cls,
        filename: str,
    ) -> Any: ...


class _FitsMaskedArrayProtocol(_FitsArrayProtocol, Protocol):
    def mask(self) -> ArrayLike: ...


@runtime_checkable
class _CubeProtocol(Protocol):
    def shape(self) -> tuple[int, int, int]: ...
    def data(self) -> ArrayLike: ...
    def mask(self) -> ArrayLike: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __array__(self) -> ArrayLike: ...


#: A generic 2-D matrix, i.e. any object with a ``shape`` that can be
#: indexed, such as a ``numpy.ndarray`` or a nested list. Used for command matrices,
#: command histories and other plain numeric tables.
MatrixLike = TypeVar("MatrixLike", bound=_MatrixProtocol)
#: A 2-D boolean (or integer 0/1) mask, typically marking which pixels lie
#: outside the pupil or region of interest.
MaskData = TypeVar("MaskData", bound=_MatrixProtocol)
#: A single 2-D image with a mask attached, such as a phase map returned by
#: an interferometer. In practice this is a ``numpy.ma.MaskedArray``.
ImageData = TypeVar("ImageData", bound=_ImageDataProtocol)
#: A 3-D stack of masked images, with shape ``(ny, nx, n_frames)``, such as
#: the set of influence functions measured for a deformable mirror.
CubeData = TypeVar("CubeData", bound=_CubeProtocol)
#: An array (masked or not) that can be saved to and loaded from a FITS file,
#: e.g. a :class:`~opticalib.core.fitsarray.FitsArray`.
FitsData = TypeVar("FitsData", _FitsArrayProtocol, _FitsMaskedArrayProtocol)


####################################
## DEVICE PROTOCOLS AND TYPE VARS ##
####################################
@runtime_checkable
class _InterfProtocol(Protocol):
    def acquire_map(
        self, nframes: int, delay: int | float, rebin: int
    ) -> ImageData: ...
    def acquire_full_frame(self, **kwargs: dict[str, Any]) -> ImageData: ...
    def capture(self, numberOfFrames: int, folder_name: str = None) -> str: ...
    def produce(self, tn: str): ...


@runtime_checkable
class _CameraProtocol(Protocol):
    def acquire_frames(self, **kwargs: dict[str, Any]) -> ImageData: ...
    def set_exptime(self, exptime: int | float) -> None: ...
    def get_exptime(self) -> int | float: ...


@runtime_checkable
class _WFSProtocol(Protocol):
    def acquire_map(
        self, nframes: int, delay: int | float, rebin: int
    ) -> ImageData: ...
    def acquire_pupil(self, **kwargs: dict[str, Any]) -> ImageData: ...
    def acquire_detector(self, **kwargs: dict[str, Any]) -> ImageData: ...


#: Any interferometer, real or simulated, that can acquire phase maps and
#: capture/produce raw frame sequences.
InterferometerDevice = TypeVar("InterferometerDevice", bound=_InterfProtocol)
#: Any camera that can acquire frames and get/set its exposure time.
CameraDevice = TypeVar("CameraDevice", bound=_CameraProtocol)
#: Any wavefront sensor that can acquire phase maps, pupil images and raw
#: detector frames.
WFSDevice = TypeVar("WFSDevice", bound=_WFSProtocol)


@runtime_checkable
class _DMProtocol(Protocol):
    @property
    def n_acts(self) -> int: ...
    def set_shape(self, cmd: MatrixLike, differential: bool) -> None: ...
    def get_shape(self) -> ArrayLike: ...
    def upload_cmd_history(
        self, cmdhist: MatrixLike, *, slave: bool | str = False
    ) -> None: ...
    def run_cmd_history(
        self,
        wfs: Optional[InterferometerDevice | WFSDevice],
        delay: int | float,
        save: Optional[str],
        differential: bool,
    ) -> str: ...


@runtime_checkable
class _FakeDMProtocol(_DMProtocol, Protocol):
    @property
    def _mask(self) -> MaskData: ...
    @property
    def _zern(self) -> Any: ...
    def _wavefront(self, **kwargs) -> ArrayLike: ...


@runtime_checkable
class _FakeInterfProtocol(_InterfProtocol, Protocol):
    def live(
        self,
    ) -> tuple: ...
    def toggle_surface_view(self) -> None: ...
    def toggle_acquisition_live_freeze(self) -> None: ...
    def toggle_live_noise(self) -> None: ...
    def live_info(self) -> None: ...
    def toggle_shape_removal(self, modes: list[int]) -> None: ...


#: Any deformable mirror, real or simulated, that can apply a shape, read it
#: back, and run a timed history of commands.
DeformableMirrorDevice = TypeVar("DeformableMirrorDevice", bound=_DMProtocol)
#: A simulated deformable mirror. It is a ``DeformableMirrorDevice`` that
#: also exposes its internal mask, Zernike generator and computed wavefront.
FakeDeformableMirrorDevice = TypeVar(
    "FakeDeformableMirrorDevice", bound=_FakeDMProtocol
)
#: A simulated interferometer. It is an ``InterferometerDevice`` that also
#: offers live-view controls (surface view, noise, shape removal, ...).
FakeInterferometerDevice = TypeVar(
    "FakeInterferometerDevice", bound=_FakeInterfProtocol
)

#: Any device object, with no required methods. Used where a function works
#: with whichever instrument it is given.
GenericDevice = TypeVar("GenericDevice")

#######################
## UTILITY FUNCTIONS ##
#######################


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
            "WFSDevice": _WFSProtocol,
            "CameraDevice": _CameraProtocol,
            "FakeDeformableMirrorDevice": _FakeDMProtocol,
            "FakeInterferometerDevice": _FakeInterfProtocol,
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
            "CameraDevice": cls.generic_check,
            "DeformableMirrorDevice": cls.generic_check,
            "FakeDeformableMirrorDevice": cls.generic_check,
            "FakeInterferometerDevice": cls.generic_check,
            "WFSDevice": cls.generic_check,
        }
        if class_name not in checks:
            raise ValueError(f"Unknown class name: {class_name}")
        try:
            check = checks[class_name](obj)
        except TypeError:
            check = checks[class_name](obj, class_name)
        return check


isinstance_ = InstanceCheck.isinstance_

######################
## Helper Functions ##
######################

def get_device_type(device: object) -> str:
    if isinstance_(device, "InterferometerDevice"):
        return "interf"
    elif isinstance_(device, "WFSDevice"):
        return "wfs"
    elif isinstance_(device, "CameraDevice"):
        return "camera"
    return device._name