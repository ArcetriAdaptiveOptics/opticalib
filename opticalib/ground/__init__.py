"""
GROUND module
=============
2024

Author(s)
---------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it
- Marco Xompero : marco.xompero@inaf.it
- Runa Briguglio : runa.briguglio@inaf.it

Description
-----------
This module provides various utility functionalities for the opticalib
package, including functions for wavefront reconstruction, interaction
matrix computation, and device management.

Contents
--------
- computerec.py : module for computing and using the reconstructor from a DM
    calibration. Used in `dmutils.flattening`
- osutils.py : module with various OS utilities, like reading/writing FITS
    files and interferometer maps.
- logger.py : module for logging utilities.
- geo.py : module for geometric utilities, like circular masks creation.
- roi.py : module for region of interest (ROI) management.
- zernike.py : module for Zernike polynomials computation and fitting on images.
"""
