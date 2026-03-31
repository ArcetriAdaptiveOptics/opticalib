Quick Start
===========

This guide shows the most common workflow: connecting to instruments, acquiring
interferometer images, and running a DM calibration.

Setting up an experiment
------------------------

After :doc:`installing <installation>` ``opticalib``, create a workspace with
the ``calpy`` command::

    calpy -f ~/alpao_experiment --create

This generates the following folder tree under ``~/alpao_experiment``:

.. code-block:: text

    alpao_experiment/
    ├── OPTData/
    │   ├── Flattening/
    │   ├── INTMatrices/
    │   ├── ModalBases/
    │   ├── OPDImages/
    │   ├── OPDSeries/
    |   ├── SPL/
    |   |   ├── Fringes/
    │   └── IFFunctions/
    ├── Logging/
    └── SysConfig/
        └── configuration.yaml

Edit ``SysConfig/configuration.yaml`` to describe your hardware (see
:doc:`configuration`).

Activating the environment
--------------------------

Run::

    calpy -f ~/alpao_experiment

``calpy`` will import ``opticalib`` (aliased as ``opt``) and
``opticalib.dmutils`` (aliased as ``dmutils``) and set the data root to your
experiment folder.

Connecting to instruments
--------------------------

.. code-block:: python

    import opticalib as opt

    # Connect to a 4D PhaseCam interferometer
    interf = opt.PhaseCam('192.168.1.10', 8011)

    # Connect to an Alpao deformable mirror (820 actuators)
    dm = opt.AlpaoDm(820)

Acquiring a wavefront image
----------------------------

.. code-block:: python

    # Acquire a single wavefront map (returns a masked numpy array)
    wf = interf.acquire_map()

    import matplotlib.pyplot as plt
    plt.imshow(wf)
    plt.colorbar(label='OPD [m]')
    plt.title('Wavefront map')
    plt.show()

Acquiring Influence Functions
-------------------------------

The :func:`~opticalib.dmutils.iff_module.iffDataAcquisition` function
orchestrates the full push-pull measurement loop.  Parameters that are not
supplied are read from the ``configuration.yaml`` file:

.. code-block:: python

    from opticalib import dmutils

    # Acquire IFF data – returns a tracking number (timestamp string)
    tn = dmutils.iff_module.iffDataAcquisition(dm, interf)
    print(f"IFF data saved under tracking number: {tn}")

Processing Influence Functions
--------------------------------

.. code-block:: python

    from opticalib.dmutils import iff_processing

    # Process the acquired data
    iff_processing.process(tn)

Flattening the deformable mirror
----------------------------------

.. code-block:: python

    from opticalib.dmutils import Flattening

    flat = Flattening(dm, interf)
    flat.applyFlatCommand()

Using simulated devices (no hardware)
---------------------------------------

``opticalib`` ships with a :mod:`~opticalib.simulator` sub-package for
offline testing:

.. code-block:: python

    from opticalib.simulator import AlpaoDm, Fake4DInterf

    dm    = AlpaoDm(nActs=97)
    interf = Fake4DInterf()

    wf = interf.acquire_map()
    print(wf.shape)

Next steps
----------

* :doc:`configuration` – configure your hardware devices.
* :doc:`api` – full API reference.
