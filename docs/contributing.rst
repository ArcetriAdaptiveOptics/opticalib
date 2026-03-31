Contributing
============

We welcome contributions to ``opticalib``!  Please follow these guidelines to
keep the codebase consistent and maintainable.

Getting started
---------------

1. **Fork** the repository on GitHub and clone your fork::

       git clone https://github.com/<your-username>/opticalib.git
       cd opticalib

2. Create a feature branch::

       git checkout -b feature/my-feature

3. Install the package in editable mode together with the development
   dependencies::

       pip install -e ".[dev]"

4. Make your changes, add tests, and open a pull request against ``main``.

Code style
----------

* The project follows **PEP 8**.  Use the ``flake8`` linter to check your code
  before committing.
* Use **type hints** for all public functions and methods.
* Write **NumPy-style docstrings** for all public symbols (functions, classes,
  methods).  See the example below.

Example docstring
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def compute_flat_cmd(
        dm: DeformableMirrorDevice,
        interf: InterfDevice,
        n_modes: int = 50,
    ) -> np.ndarray:
        """
        Compute a flattening command for *dm* using *interf*.

        Parameters
        ----------
        dm : DeformableMirrorDevice
            The deformable mirror to flatten.
        interf : InterfDevice
            The interferometer used to measure the wavefront.
        n_modes : int, optional
            Number of Zernike modes to include in the reconstruction.
            Defaults to 50.

        Returns
        -------
        flat_cmd : np.ndarray
            Command vector of length ``dm.nActs``.

        Raises
        ------
        MatrixError
            If the interaction matrix is singular.
        """

Tests
-----

All new features must be accompanied by unit tests located in the ``tests/``
directory.  Run the test suite with::

    pytest

Run only a targeted subset to speed up iteration::

    pytest tests/test_dmutils_flattening.py -v

Use the following markers to classify your tests:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Marker
     - Description
   * - ``@pytest.mark.unit``
     - Fast, dependency-free unit tests
   * - ``@pytest.mark.integration``
     - Tests that require file I/O or multiple components
   * - ``@pytest.mark.slow``
     - Long-running tests (excluded from the default CI run)

Building the documentation
--------------------------

Documentation sources live in the ``docs/`` folder.  To build the HTML docs
locally::

    cd docs
    pip install -r requirements.txt
    make html

The output is written to ``docs/_build/html/``.  Open
``docs/_build/html/index.html`` in your browser to preview the result.

.. note::
   The docs are built automatically on `Read the Docs`_ on every push to
   ``main``.

.. _Read the Docs: https://readthedocs.org/projects/opticalib/

Reporting issues
----------------

Open an issue on the `GitHub issue tracker`_.  Please include:

* A minimal, reproducible example.
* The version of ``opticalib`` (``python -c "import opticalib; print(opticalib.__version__)"``).
* The Python version and operating system.

.. _GitHub issue tracker: https://github.com/ArcetriAdaptiveOptics/opticalib/issues

License
-------

By contributing you agree that your contributions will be licensed under the
`MIT License <https://opensource.org/licenses/MIT>`_ that covers the project.
