# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is `opticalib`

`opticalib` is a Python framework, developed by the Adaptive Optics Group at INAF - Osservatorio
Astrofisico di Arcetri, for **optical calibration of deformable mirrors (DMs) and adaptive optics (AO)
experimentation** in the laboratory. It grew out of the control/calibration software for the `ELT @ M4`
adaptive mirror and its calibration tower (`OTT`), then was generalized into a standalone package.

It serves two main purposes:

- **Hardware abstraction**: uniform, high-level Python interfaces to lab instrumentation — interferometers
  (4D PhaseCam, AccuFiz), deformable mirrors (Alpao, AdOptica, Splatt, DP, M4AU, PetalMirror), wavefront
  sensors and cameras — over Ethernet/Pyro4/plico/vendor SDKs, with matching software simulators so the
  same code paths run without hardware attached.
- **DM calibration routines**: acquisition and processing of Influence Functions (IFF), interaction/
  reconstruction matrix computation, actuator slaving, DM flattening, alignment, phasing (segmented
  mirrors), and time-series/wavefront analysis of the results.

### Key features

- **Config-driven device access**: a single `configuration.yaml` (SYSTEM / DEVICES / INFLUENCE.FUNCTIONS /
  SYSTEM.ALIGNMENT / STITCHING sections) declares every device and experiment parameter; `opticalib.core.root`
  builds the on-disk data folder tree from it at import time.
- **Tracking-number (`tn`) data model**: every acquisition/processing result is stored under a
  `YYYYMMDD_HHMMSS` timestamp folder (`is_tn`/`newtn` in `ground/osutils.py`) inside a fixed set of
  `OPTData` subfolders (`IFFunctions`, `INTMatrices`, `ModalBases`, `OPDImages`, `OPDSeries`, `Flattening`,
  `Alignment`, `SPL`, ...). This `tn` convention is the thread that ties acquisition, processing and
  analysis code together across the whole package.
- **Simulators as first-class citizens**: `opticalib.simulator` mirrors the real device classes
  (`AlpaoDm`, `DP`, `PetalMirror`, `Fake4DInterf`, ...) with physics-based fake implementations, so
  procedures can be developed/tested without a bench. Simulator reference data is not shipped in the repo —
  it is fetched/cached on demand (`opticalib.simulator.prefetch_simdata`).
- **`calpy` CLI**: the package installs a `calpy` entry point (`setup_calpy.py`) that creates a per-experiment
  config + data tree and drops into an IPython shell (or Qt GUI, via `opticalib.gui`) with `opticalib`
  pre-imported and aliased (see `opticalib/__init_script__/initCalpy.py`).
- **GPU/CPU-agnostic array ops via `xupy`**: performance-sensitive numeric code (reconstructor computation,
  IFF processing, image analysis, actuator slaving, simulator RBF interpolation) uses `xupy`, a
  cupy/numpy backend-switching layer (also authored by the maintainer) that additionally supports masked
  arrays on GPU. Treat `xupy` as numpy-compatible unless you're touching GPU-specific code paths.

## Architecture

```
opticalib/
├── core/            Config & runtime plumbing — imported first, has import-time side effects
├── devices/         High-level real-hardware device classes + low-level `_API` backends
├── simulator/        Fake device classes mirroring devices/, + `_API` physics models & remote sim-data cache
├── dmutils/          DM calibration domain logic (IFF acquisition/processing, slaving, flattening)
├── ground/           Shared low-level tooling (FITS/tn I/O, geometry, ROI, modal decomposition, reconstructor, logger)
├── analyzer/         Image / signal / time-series analysis on acquired data
├── procedures/       High-level orchestration routines composing devices + dmutils + analyzer
├── gui/              PyQt5 `CalpyGUI`, an embedded IPython console for `calpy --gui`
└── visualization.py  Plotting helpers (exposed as `opticalib.vis`)
```

- **`core/root.py`** is the heart of the package: at **import time** it reads `AOCONF` (env var, default
  `~/.opticalib/SysConfig/configuration.yaml`), loads the YAML config, and creates the entire `OPTData` /
  `Logging` / `SysConfig` folder tree on disk. `folders` (a `_Folds` instance) is the canonical object for
  every root path in the package — prefer it over hardcoding paths. Because this runs on import, changing
  which config is active at runtime is *not* a simple attribute set — use
  `opticalib.set_configuration_file(path)` / `unset_configuration_file()`, which update `AOCONF` and
  `importlib.reload()` every already-imported `opticalib.*` runtime module so all cached folder/config
  references stay in sync. Keep this in mind if you add new module-level state derived from config: it must
  be re-derived on reload, not just set once.
- **`core/config.py`** reads/writes sections of `configuration.yaml` (`get_device_config`, `get_iff_config`,
  `get_alignment_config`, ...). IFF acquisitions get their own copied `iffConfig.yaml` per tracking number
  for provenance (`copy_iff_config_file`, `update_iff_config`).
- **`core/_types.py`** defines the package's typing surface: `Protocol`-based structural types
  (`_DMProtocol`, `_InterfProtocol`, `_CameraProtocol`, `_WFSProtocol`, `MatrixLike`, `ImageData`,
  `CubeData`, ...) plus a custom `isinstance_()` duck-typing helper. Device/DM code type-hints against these
  protocols rather than concrete classes — new device classes just need to satisfy the shape, not inherit
  from anything specific (though most also derive from an ABC in `devices/_API/base_devices.py` for shared
  behavior like actuator slaving).
- **`devices/_API/base_devices.py`** defines the ABCs (`BaseDeformableMirror`, `BaseWavefrontSensor`,
  `BaseCamera`) that public device classes in `devices/*.py` implement. `BaseDeformableMirror` centralizes
  actuator-slaving logic (`_apply_slaving`, `_slave_cmdmat`, delegating to `opticalib.dmutils.slaving`) so
  every DM subclass gets consistent `slave=True/"method"` behavior in `set_shape`/`upload_cmd_history` for
  free. Vendor-specific wire protocols (Alpao SDK, plico_dm, 4D `i4d`, PI, Splatt/microgate) live in
  `devices/_API/*API.py`, kept separate from the public class so swapping a backend (e.g. Alpao's recent
  `plico_dm` support) doesn't change the public interface.
- **`simulator/`** mirrors this same split: `simulator/fake_dms.py` / `fake_interf.py` are the public fake
  classes, `simulator/_API/` holds the physics (`base_fake_alpao.py`, `base_fake_adopticadm.py`,
  `base_petalmirror.py`) and GPU-accelerated RBF interpolation (`_rbf_gpu.py`). `simulator/_API/simdata.py`
  fetches/caches reference simulation data on demand instead of shipping it in the repo.
- **`dmutils/iff_processing.py` and `procedures/iff.py`** implement the IFF acquisition → processing
  pipeline: acquire raw OPD images per commanded mode (`OPDImages/<tn>`), differential-process them into
  per-mode FITS + a cube (`IFFunctions/<tn>`, `INTMatrices/<tn>`), optionally register/stack cubes across
  tracking numbers. `ground/reconstructor.py` (`ComputeReconstructor`) consumes an interaction-matrix cube
  to compute the reconstruction matrix (SVD-based, `xupy`-accelerated).
- **`core/fitsarray.py`** (`FitsArray`, a `numpy.ndarray` subclass carrying a FITS header) and
  `ground/osutils.py` (`save_fits`/`load_fits`, `is_tn`/`newtn`, `create_data_folder`) are the shared I/O
  layer almost every other module builds on — prefer these over raw `astropy.io.fits` calls so headers and
  tracking-number folder conventions stay consistent.
- **Top-level `opticalib/__init__.py`** re-exports the common surface (`load_fits`, `save_fits`, `folders`,
  device classes via `devices/__init__.py`, `config`, `typings` = `core._types`, `fits_array`) and imports
  the submodules (`analyzer`, `devices`, `ground`, `dmutils`, `simulator`, `visualization` aliased as `vis`).
  When adding a new public symbol, wire it through here and into `__all__`, matching the existing style.

## Commands

Environment: Python ≥ 3.10, dependencies in `requirements.txt` (includes `xupy`, `arte`, `astropy`,
`Pyro4`, `PyQt5`+`qtconsole` for the GUI, vendor SDKs like `vmbpy`/`pipython`).

```bash
# Install (editable, for development)
pip install -r requirements.txt
pip install -e .

# Run the whole test suite
pytest

# Run one test module / class / function
pytest tests/test_core_exceptions.py
pytest tests/test_ground_logger.py::TestSetUpLogger::test_set_up_logger_creation

# With coverage (matches CI)
pytest --cov=opticalib --cov-report=html

# Deselect slow/integration tests
pytest -m "not slow"

# Lint (matches CI's flake8 invocation)
flake8 . --count --select=E9,F63,F7,F82 --max-complexity=10 --max-line-length=127

# Build the Sphinx docs
cd docs && make html
```

CI (`.github/workflows/tests.yml`) runs this same flake8 + pytest combination across Python 3.10–3.13 on
every push.

Tests live in `tests/`, one `test_<subpackage>_<module>.py` file per source module, mirroring the package
layout (`test_core_*`, `test_ground_*`, `test_dmutils_*`, `test_devices_*`, `test_simulator_*`,
`test_analyzer_*`). Shared fixtures (`temp_dir`, `temp_config_file`, `mock_dm`, `mock_interferometer`,
`sample_image`/`sample_cube`, `sample_iff_folder_structure`, ...) are in `tests/conftest.py` — reuse these
instead of hand-rolling temp dirs or mocks. Hardware-dependent code should be tested against the
`simulator`/mocks, not real devices.

## Rules

- Hardware side effects are real: device classes talk to physical DMs/interferometers over the network.
  Never assume a change is safe to exercise interactively against real hardware — use `opticalib.simulator`
  or the `mock_dm`/`mock_interferometer` fixtures unless explicitly told a real bench is available.
- Respect the `tn` (tracking number) convention everywhere data is written/read: use `ground.osutils.newtn`/
  `is_tn`, and write into the appropriate `folders.<X>_ROOT_FOLDER` rather than inventing new path schemes.
- Prefer `folders` (`opticalib.core.root.folders`) and `core/config.py` accessors over hardcoded paths or
  re-parsing `configuration.yaml` directly.
- `core/root.py` and `core/config.py` module-level code runs at import time and depends on `AOCONF`; be
  careful with import order and with tests that touch configuration (see `conftest.py`'s
  `reset_environment` fixture and `monkeypatch` usage patterns already in the test suite).
- When adding a new device backend (a new `_API` module), keep the vendor SDK/protocol code isolated in
  `devices/_API/` (or `simulator/_API/`) and expose only the ABC-conforming public interface from
  `devices/*.py` (or `simulator/*.py`), following the existing Alpao (`plico_dm` vs. native SDK) pattern.
- Treat `xupy` as the array backend for anything performance-sensitive or GPU-relevant (reconstructor,
  slaving, image processing, simulator interpolation); default to plain `numpy` elsewhere unless there's a
  reason to unify with surrounding `xupy` code in the same module.
- Don't add simulator reference data files to the repo — `simulator/_API/simdata.py`'s on-demand
  fetch/cache mechanism replaced that.
- Always prefer reusable code already available in the `opticalib` package when possible, instead of writing
  new code which does the same thing. Example: use `opticalib.ground.osutils.load_fits` to load a fits file
  instead of using the `astropy.io.fits` primitive.

## Python rules

(Condensed from `.github/instructions/python.instructions.md`, which applies to all `*.py` files.)

- Type-hint everything; use the package's own type aliases/protocols from `opticalib.core._types`
  (`typings` when imported via the top-level package) instead of loose `typing` primitives where an
  equivalent exists (e.g. `MatrixLike`, `ImageData`, `CubeData`, `Reconstructor`).
- Docstrings follow **numpy docstring conventions**, placed immediately after `def`/`class`.
- Follow PEP 8; 4-space indentation.
- Commit messages use the project's own format: `` `<mod> <type>: <subject>` ``, where `<mod>` is the
  module/file/class touched (no path or extension), `<type>` is one of
  `feat|fix|docs|style|refactor|perf|test|chore`, and `<subject>` is a concise, informative description.
  If a function is added, name it in the subject; if `__version__` changes, mention the new version and a
  summary of what's included. Examples:
  - `iff_processing refactor: optimized mode matrix processing for generalized use cases and better performance`
  - `ComputeReconstructor perf: improved performance of reconstructor computation by optimizing matrix operations`
  - `AlpaoDm feat: add plico_dm backend support with IP/PORT configuration`
