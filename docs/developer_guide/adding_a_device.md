# Adding a device

OptiCalib talks to hardware through *structural* types, not inheritance from a
single god-class. A new instrument becomes usable everywhere in the library as
soon as it exposes the methods of the matching protocol — see
{doc}`/reference/typings`.

## 1. Pick the protocol you must satisfy

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - You are adding a...
  - Protocol
  - Required members
* - Interferometer
  - `_InterfProtocol`
  - `acquire_map`, `acquire_full_frame`, `capture`, `produce`
* - Wavefront sensor
  - `_WFSProtocol`
  - `acquire_map`, `acquire_pupil`, `acquire_detector`
* - Camera
  - `_CameraProtocol`
  - `acquire_frames`, `set_exptime`, `get_exptime`
* - Deformable mirror
  - `_DMProtocol`
  - `n_acts`, `set_shape`, `get_shape`, `upload_cmd_history`, `run_cmd_history`
```

All of them live in {mod}`opticalib.core._types`. Because they are
{class}`~typing.Protocol` classes marked `runtime_checkable`, you can verify
conformance at any time:

```{code-block} python
from opticalib.core._types import isinstance_

assert isinstance_(my_mirror, "DeformableMirrorDevice")
```

## 2. Subclass the matching abstract base

The bases in {mod}`opticalib.devices._API.base_devices` implement the shared
bookkeeping — configuration lookup, command clamping, logging, data filing —
so you only write the vendor-specific parts:

- {class}`~opticalib.devices._API.base_devices.BaseDeformableMirror`
- {class}`~opticalib.devices._API.base_devices.BaseWavefrontSensor`
- {class}`~opticalib.devices._API.base_devices.BaseCamera`

```{code-block} python
# opticalib/devices/deformable_mirrors.py
from ._API.base_devices import BaseDeformableMirror


class MyVendorDm(BaseDeformableMirror):
    """Docstring becomes the API reference entry -- write it properly."""

    def __init__(self, nacts: int | None = None, **kwargs):
        super().__init__(nacts, **kwargs)
        self._sdk = self._connect()          # vendor SDK handle

    def _set_shape(self, cmd, differential=False):
        self._sdk.SetVector(cmd)

    def _get_shape(self):
        return self._sdk.GetVector()
```

```{note}
Vendor SDKs are not redistributable. Guard the import inside the method or
behind a `try`/`except ImportError`, and add the package name to
`autodoc_mock_imports` in `docs/conf.py` so the documentation build does not
need it.
```

## 3. Register the device in the configuration

Add a section to `DEVICES` in the shipped template
`opticalib/core/_configurations/configuration.yaml`, and document every key in
{doc}`/configuration`. Constructors should read their own defaults from there:

```{code-block} yaml
DEVICES:
  DEFORMABLE.MIRRORS:
    MyVendorDm97:
      serialNumber: MV-0001
      sdk_folder_path: /opt/myvendor/lib
```

and document the new key set in {func}`~opticalib.core.config.get_dm_config`
terms.

## 4. Export it

Expose the class from the subpackage `__init__` and from the top level so users
can write `opt.MyVendorDm()`:

```{code-block} python
# opticalib/devices/deformable_mirrors.py
__all__ = [..., "MyVendorDm"]

# opticalib/devices/__init__.py
from .deformable_mirrors import MyVendorDm

# opticalib/__init__.py
from .devices import MyVendorDm
```

## 5. Provide a simulated twin

Every real device should have a fake counterpart in {mod}`opticalib.simulator`,
subclassing the appropriate base in `opticalib.simulator._API`. This is what
makes the device testable on CI and usable by people without the hardware — see
{doc}`/user_guide/calpy` § *Working without hardware*.

## 6. Add it to the reference

Add the class to the table in {doc}`/reference/devices` (or
{doc}`/reference/simulator`). The `automodule` directive picks up the docstring
automatically; the table is what tells a reader the hardware is supported at
all.

## Checklist

- Protocol conformance asserted with `isinstance_`
- Vendor SDK import guarded and added to `autodoc_mock_imports`
- Configuration template updated
- Every new key documented in {doc}`/configuration`
- Exported from subpackage and top level
- Simulated twin added
- Reference table updated
- Docstring in NumPy style (Napoleon renders it)
