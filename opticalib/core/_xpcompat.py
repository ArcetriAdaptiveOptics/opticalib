"""
Compatibility helpers for :mod:`xupy` (>= 2.0, NumPy 2 namespace).
"""

import xupy as _xp


def compute_float():
    """
    Floating dtype for numerical work on the active xupy backend.

    ``float32`` on GPU (CuPy) for speed, ``float64`` on CPU (NumPy). This is
    what the removed ``xupy.float`` alias used to resolve to; it is evaluated
    at call time, so it follows ``xupy.use_cpu()`` / ``xupy.use_gpu()``.
    """
    return _xp.float32 if _xp.on_gpu else _xp.float64
