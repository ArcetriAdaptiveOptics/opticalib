"""
Ground — reconstruction, modal decomposition, geometry and logging
==================================================================

The computational "ground segment" of the library.  Provides wavefront
reconstruction from influence-function data, modal decomposition
(Zernike, Karhunen–Loève, radial basis functions), pupil geometry and
region-of-interest utilities, plus the logging and file-system helpers
used throughout the library.

"""

from . import reconstructor, roi, modal_decomposer, geometry, osutils
