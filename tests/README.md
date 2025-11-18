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

