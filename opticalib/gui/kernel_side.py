"""
Kernel-side helpers of CalpyGUI
===============================

This module runs **inside the IPython kernel** started by the GUI (never in
the GUI process) and must not import Qt.  :func:`install` binds it in the
user namespace as ``_gui``, so GUI actions can be expressed as plain Python
that is echoed in the console, e.g.::

    _gui.view(img, "img")

It provides:

* figure publication: every open matplotlib figure that changed is rendered
  to PNG and sent to the GUI plot panel (after each cell, and on
  ``plt.show()`` / ``plt.pause()``);
* :func:`view`: show an array in the GUI interactive image viewer;
* :func:`workspace` and :func:`folders`: JSON summaries queried by the GUI
  to populate the workspace and data browser panels.

Messages to the GUI are ``display_data`` outputs carrying one of the custom
MIME types below; the GUI console intercepts them.
"""

import base64 as _base64
import io as _io
import itertools as _itertools
import json as _json
import os as _os
import reprlib as _reprlib
import sys as _sys
import tempfile as _tempfile
import types as _types
from typing import Any, Callable, Dict, Iterable, List, Optional

#: MIME type of a matplotlib figure published to the GUI.
FIGURE_MIME = "application/vnd.calpy.figure+json"
#: MIME type of an array sent to the GUI image viewer.
IMAGE_MIME = "application/vnd.calpy.image+json"

#: Resolution of the PNG figures sent to the GUI.
FIGURE_DPI = 120

#: Matplotlib backend of the kernel (see :mod:`opticalib.gui.kernel_backend`).
BACKEND = "module://opticalib.gui.kernel_backend"

# Data folders shown in the GUI data browser, as (label, root attribute).
_DATA_FOLDERS = [
    ("OPD images", "OPD_IMAGES_ROOT_FOLDER"),
    ("OPD series", "OPD_SERIES_ROOT_FOLDER"),
    ("IF functions", "IFFUNCTIONS_ROOT_FOLDER"),
    ("Interaction matrices", "INTMAT_ROOT_FOLDER"),
    ("Flattening", "FLAT_ROOT_FOLDER"),
    ("Modal bases", "MODALBASE_ROOT_FOLDER"),
    ("Alignment calibration", "ALIGN_CALIBRATION_ROOT_FOLDER"),
    ("Alignment results", "ALIGN_RESULTS_ROOT_FOLDER"),
    ("SPL", "SPL_DATA_ROOT_FOLDER"),
    ("Logging", "LOGGING_ROOT_FOLDER"),
]


# Device kinds, as (kind, methods of the class), most specific first; they
# mirror the device protocols of opticalib.core._types.
_DEVICE_METHODS = [
    ("dm", ("set_shape", "get_shape")),
    ("wfs", ("acquire_map", "acquire_pupil")),
    ("interferometer", ("acquire_map",)),
    ("camera", ("acquire_frames",)),
]

_figure_uids = _itertools.count(1)
_published_uids = set()
_view_counter = _itertools.count(1)
_baseline: Dict[str, int] = {}
_repr = _reprlib.Repr()
_repr.maxstring = 60
_repr.maxother = 60


def _tmp_dir() -> str:
    """Return the folder used to hand arrays over to the GUI."""
    path = _os.environ.get("CALPY_GUI_TMP")
    if not path:
        path = _os.path.join(_tempfile.gettempdir(), "calpygui")
    _os.makedirs(path, exist_ok=True)
    return path


def _default_publisher(data: Dict[str, Any]) -> None:
    """Send *data* to the frontends as a ``display_data`` message."""
    from IPython.display import publish_display_data

    publish_display_data(data)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_title(fig, num: Any) -> str:
    """Return a short title for a matplotlib figure."""
    label = fig.get_label()
    if label:
        return label
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        return suptitle.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return f"Figure {num}"


def _figure_is_empty(fig) -> bool:
    """Whether *fig* has nothing worth showing yet."""
    return not (fig.axes or fig.images or fig.texts or fig.lines or fig.patches)


def publish_figures(
    figures: Optional[Iterable[Any]] = None,
    publish: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> int:
    """
    Send new or changed matplotlib figures to the GUI.

    Parameters
    ----------
    figures : iterable of Figure, optional
        Figures to publish even if they did not change.
    publish : callable, optional
        Function receiving the ``display_data`` payload (used by tests);
        defaults to IPython's ``publish_display_data``.

    Returns
    -------
    int
        Number of figures published.
    """
    if "matplotlib" not in _sys.modules:
        return 0
    from matplotlib._pylab_helpers import Gcf

    publish = publish or _default_publisher
    forced = list(figures or [])
    count = 0
    for manager in Gcf.get_all_fig_managers():
        canvas = manager.canvas
        fig = canvas.figure
        uid = getattr(fig, "_calpy_uid", None)
        if uid is None:
            uid = next(_figure_uids)
            fig._calpy_uid = uid
        changed = (
            fig.stale
            or getattr(canvas, "_calpy_dirty", False)
            or uid not in _published_uids
            or any(f is fig for f in forced)
        )
        if not changed or _figure_is_empty(fig):
            continue
        buf = _io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
        except Exception as exc:  # never break the user's cell
            print(f"[CalpyGUI] could not render figure {manager.num}: {exc}")
            continue
        finally:
            fig.stale = False
            canvas._calpy_dirty = False
        payload = {
            "uid": uid,
            "num": manager.num,
            "title": _figure_title(fig, manager.num),
            "png": _base64.b64encode(buf.getvalue()).decode("ascii"),
        }
        publish({FIGURE_MIME: payload, "text/plain": f"<Figure {manager.num}>"})
        _published_uids.add(uid)
        count += 1
    return count


def _post_execute() -> None:
    """IPython ``post_execute`` hook: publish figures changed by the cell."""
    try:
        matplotlib = _sys.modules.get("matplotlib")
        if matplotlib is None or matplotlib.get_backend() != BACKEND:
            return  # e.g. "%matplotlib qt": figures live in their own windows
        publish_figures()
    except Exception as exc:
        print(f"[CalpyGUI] figure publication failed: {exc}")


# ---------------------------------------------------------------------------
# Image viewer
# ---------------------------------------------------------------------------


def _to_numpy(obj: Any):
    """Convert *obj* to a numpy (possibly masked) array on the host."""
    import numpy as np

    if hasattr(obj, "asmarray"):  # xupy masked array
        obj = obj.asmarray()
    elif hasattr(obj, "__cuda_array_interface__") and hasattr(obj, "get"):
        obj = obj.get()
    if isinstance(obj, np.ma.MaskedArray):
        return obj
    return np.asarray(obj)


def view(
    obj: Any,
    title: Optional[str] = None,
    publish: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[str]:
    """
    Show an array in the GUI interactive image viewer.

    1-D arrays are shown as a line plot, 2-D arrays as an image and 3-D
    arrays as a cube (opticalib layout: frames along the last axis).
    Masked values are hidden.

    Parameters
    ----------
    obj : array_like
        Data to show (numpy, masked, xupy/cupy arrays are accepted).
    title : str, optional
        Title shown in the viewer.
    publish : callable, optional
        Function receiving the ``display_data`` payload (used by tests).

    Returns
    -------
    str or None
        Path of the temporary file handed over to the GUI.
    """
    import numpy as np

    arr = _to_numpy(obj)
    if arr.ndim not in (1, 2, 3):
        raise ValueError(f"Cannot view a {arr.ndim}-D array (1-D to 3-D only).")
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
        raise TypeError(f"Cannot view an array of dtype {arr.dtype}.")

    path = _os.path.join(_tmp_dir(), f"view_{_os.getpid()}_{next(_view_counter)}.npz")
    data = np.ma.getdata(arr)
    mask = np.ma.getmaskarray(arr) if isinstance(arr, np.ma.MaskedArray) else None
    if mask is not None and mask.any():
        np.savez(path, data=data, mask=mask)
    else:
        np.savez(path, data=data)

    payload = {
        "path": path,
        "title": title or f"{arr.ndim}-D array {tuple(arr.shape)}",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    (publish or _default_publisher)(
        {IMAGE_MIME: payload, "text/plain": f"<view {payload['title']}>"}
    )
    return path


# ---------------------------------------------------------------------------
# Workspace and folders
# ---------------------------------------------------------------------------


def _classify(value: Any) -> str:
    """
    Return the GUI kind of *value* (``'dm'``, ``'array'``, ...).

    Only the *type* of *value* is inspected: probing attributes of the
    instance could run arbitrary code (e.g. a network call on a Pyro4 proxy).
    """
    import numpy as np

    cls = type(value)
    if isinstance(value, np.ndarray) or hasattr(cls, "__cuda_array_interface__"):
        return "array"
    if callable(getattr(cls, "asmarray", None)):  # xupy masked array
        return "array"
    for kind, methods in _DEVICE_METHODS:
        if all(callable(getattr(cls, m, None)) for m in methods):
            return kind
    return "other"


def _summary(value: Any, kind: str) -> str:
    """Return a one-line description of *value*."""
    if kind == "array":
        shape = tuple(getattr(value, "shape", ()))
        dtype = getattr(value, "dtype", "")
        masked = " masked" if hasattr(value, "mask") else ""
        return f"{shape} {dtype}{masked}"
    try:
        return _repr.repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _is_hidden(name: str, value: Any, hidden_ns: Dict[str, Any]) -> bool:
    """Whether *name* should be left out of the workspace listing."""
    if name.startswith("_"):
        return True
    if name in hidden_ns and hidden_ns[name] is value:
        return True
    if _baseline.get(name) == id(value):
        return True
    return isinstance(
        value,
        (
            _types.ModuleType,
            _types.FunctionType,
            _types.BuiltinFunctionType,
            _types.MethodType,
            type,
        ),
    )


def workspace_items(namespace: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Describe the user variables of the kernel namespace.

    Modules, functions, classes, private names and the names defined by the
    calpy bootstrap (see :func:`mark_baseline`) are left out.

    Parameters
    ----------
    namespace : dict, optional
        Namespace to describe; defaults to the IPython user namespace.

    Returns
    -------
    list of dict
        One ``{'name', 'type', 'module', 'kind', 'summary'}`` entry per
        variable.
    """
    hidden_ns: Dict[str, Any] = {}
    if namespace is None:
        shell = _shell()
        namespace = shell.user_ns
        hidden_ns = shell.user_ns_hidden
    items = []
    for name, value in sorted(namespace.items(), key=lambda kv: str(kv[0])):
        if not isinstance(name, str):
            continue
        try:
            if _is_hidden(name, value, hidden_ns):
                continue
            kind = _classify(value)
            item = {
                "name": name,
                "type": type(value).__name__,
                "module": str(type(value).__module__),
                "kind": kind,
                "summary": _summary(value, kind),
            }
        except Exception:  # one odd object must not break the listing
            item = {
                "name": name,
                "type": type(value).__name__,
                "module": "",
                "kind": "other",
                "summary": "<unavailable>",
            }
        items.append(item)
    return items


def workspace() -> str:
    """Return :func:`workspace_items` as a JSON string (queried by the GUI)."""
    return _json.dumps(workspace_items())


def folders() -> str:
    """
    Return the opticalib data folders as a JSON string (queried by the GUI).

    Returns
    -------
    str
        JSON object with ``base`` (data root), ``config`` (configuration
        file), ``categories`` (list of ``[label, path]``) and ``paths``
        (every ``*_FOLDER`` path of :mod:`opticalib.core.root`, by name).
    """
    root = _sys.modules.get("opticalib.core.root")
    if root is None:
        import opticalib.core.root as root
    categories = [
        [label, getattr(root, attr)]
        for label, attr in _DATA_FOLDERS
        if getattr(root, attr, None)
    ]
    paths = {
        name: value
        for name, value in vars(root).items()
        if name.isupper() and name.endswith("FOLDER") and isinstance(value, str)
    }
    return _json.dumps(
        {
            "base": getattr(root, "BASE_DATA_PATH", ""),
            "config": getattr(root, "CONFIGURATION_FILE", ""),
            "categories": categories,
            "paths": paths,
        }
    )


def describe(value: Any, width: int = 60) -> str:
    """
    Return a short, readable description of a (configuration) value.

    Arithmetic integer sequences are shown as ``np.arange(start, stop[, step])``
    and long lists are shortened, so the GUI can show defaults as hints.

    Parameters
    ----------
    value : Any
        The value.
    width : int, optional
        Maximum length of the description.

    Returns
    -------
    str
        The description.
    """
    import numpy as np

    if isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str):
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size > 3 and np.issubdtype(arr.dtype, np.integer):
            steps = np.diff(arr)
            if np.all(steps == steps[0]) and steps[0] != 0:
                stop = int(arr[-1] + steps[0])
                step = "" if steps[0] == 1 else f", {int(steps[0])}"
                return f"np.arange({int(arr[0])}, {stop}{step})"
        if arr.dtype != object:
            value = arr.tolist()
    text = repr(value) if not isinstance(value, str) else value
    return text if len(text) <= width else text[: width - 1] + "…"


def backend() -> str:
    """
    Return the array backend of xupy as a JSON string (queried by the GUI).

    Returns
    -------
    str
        JSON object with ``on_gpu`` (xupy creates CuPy arrays) and
        ``available`` (CuPy could be loaded, so ``xupy.use_gpu()`` works);
        both ``null`` when xupy cannot be imported.
    """
    try:
        import xupy
        from xupy import _core
    except Exception:
        return _json.dumps({"on_gpu": None, "available": None})
    return _json.dumps(
        {
            "on_gpu": bool(getattr(xupy, "on_gpu", False)),
            "available": bool(getattr(_core, "_GPU_AVAILABLE", False)),
        }
    )


def mark_baseline() -> None:
    """Hide the names currently defined (the calpy bootstrap) from the workspace."""
    _baseline.clear()
    _baseline.update({k: id(v) for k, v in _shell().user_ns.items()})


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _shell():
    """Return the running IPython shell."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("opticalib.gui.kernel_side must run inside IPython.")
    return shell


def install(shell=None) -> None:
    """
    Register the figure hook and bind this module as ``_gui``.

    Parameters
    ----------
    shell : InteractiveShell, optional
        The kernel shell; defaults to the running one.
    """
    shell = shell or _shell()
    module = _sys.modules[__name__]
    try:
        shell.events.unregister("post_execute", _post_execute)
    except ValueError:
        pass
    shell.events.register("post_execute", _post_execute)
    shell.user_ns["_gui"] = module
    shell.user_ns_hidden["_gui"] = module
