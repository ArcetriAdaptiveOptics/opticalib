# Opticalib Test Suite


This directory contains the test suite for the `opticalib` package, focusing on the core and ground modules.

## Running Tests

### Install Dependencies

Make sure you have pytest installed:

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Module

```bash
pytest tests/test_core_exceptions.py
pytest tests/test_ground_logger.py
```

### Run Tests with Coverage

```bash
pytest --cov=opticalib --cov-report=html
```

### Run Tests Verbosely

```bash
pytest -v
```

### Run Specific Test Class or Function

```bash
pytest tests/test_core_exceptions.py::TestDeviceNotFoundError
pytest tests/test_ground_logger.py::TestSetUpLogger::test_set_up_logger_creation
```

## Test Structure

- `conftest.py`: Shared fixtures and pytest configuration
- `test_core_*.py`: Tests for the `core` module
  - `test_core_exceptions.py`: Custom exceptions
  - `test_core_read_config.py`: Configuration file handling
  - `test_core_root.py`: Path and folder management
- `test_ground_*.py`: Tests for the `ground` module
  - `test_ground_logger.py`: Logging utilities
  - `test_ground_osutils.py`: File operations and utilities
  - `test_ground_geo.py`: Geometric operations
  - `test_ground_roi.py`: Region of interest operations
  - `test_ground_computerec.py`: Reconstructor computation

## Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `temp_config_file`: Temporary configuration file
- `sample_image`: Sample masked image for testing
- `sample_cube`: Sample cube for testing
- `circular_mask`: Circular mask for testing
- `tracking_number`: Valid tracking number

## Notes

- Tests use temporary directories to avoid modifying system files
- Some tests may require specific dependencies (e.g., matplotlib for interactive plots)
- Tests are designed to be independent and can run in any order

### GUI tests (`test_gui_*.py`)

- Locally they run without a display (`QT_QPA_PLATFORM=offscreen`, set by `conftest.py`),
  but Qt still needs the OpenGL/EGL system libraries. On Debian/Ubuntu:

  ```bash
  sudo apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3
  ```

- In CI (`.github/workflows/tests.yml`) they run on a **real virtual display** instead:
  the workflow starts `Xvfb` on `DISPLAY=:99` and exports `QT_QPA_PLATFORM=xcb`, which
  wins over the `offscreen` default of `conftest.py` (a `setdefault`). The runner image
  therefore also gets the X11/xcb libraries, Mesa and a set of fonts (see the
  `Install Qt/X11 system libraries` step). To reproduce CI locally:

  ```bash
  sudo apt-get install xvfb x11-utils libxkbcommon-x11-0 libxcb-cursor0
  Xvfb :99 -screen 0 1920x1080x24 -ac > /dev/null 2>&1 &
  DISPLAY=:99 QT_QPA_PLATFORM=xcb pytest -rs
  ```

- When Qt cannot be loaded (missing libraries or packages), the GUI tests are
  skipped, with the reason (e.g. `libEGL.so.1: cannot open shared object file`)
  shown by `pytest -rs`. The workflow treats that as a failure: the
  `Check that the GUI tests really ran` step asserts that no graphical test was skipped.
- Qt settings are redirected to a temporary folder by the `qapp` fixture, so the
  tests never touch the user's configuration.
- `test_gui_app.py` starts the whole application with a real IPython kernel in a
  subprocess (marked `integration`, about 15 s). That subprocess pins
  `QT_QPA_PLATFORM=offscreen` itself, so it is the only graphical test that does not
  draw on the virtual display.

