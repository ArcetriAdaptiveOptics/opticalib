"""
GPU-accelerated Thin Plate Spline (TPS) and RBF interpolation using CuPy.

This module provides a GPU-optimized alternative to scipy's RBFInterpolator
for cases with many constraint points (>1000) where matrix operations
become the bottleneck.

Theory:
    An RBF interpolant is a linear combination of radial basis functions
    centered at data points y plus a polynomial:
    
        f(x) = Σᵢ aᵢ φ(||x - yᵢ||) + Σⱼ bⱼ pⱼ(x)
    
    where φ is the RBF kernel (e.g., TPS: r² log(r)) and pⱼ are polynomial
    monomials. The coefficients (a, b) are found by solving:
    
        [K + λI   P] [a] = [d]
        [P^T      0] [b]   [0]
    
    where K is the RBF Gram matrix and λ is smoothing.

References:
    [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
    [2] Wahba, G., 1990. Spline Models for Observational Data.
"""

from opticalib import typings as _ot
import numpy as np
import xupy as xp

if xp.on_gpu:

    _DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024 ** 2
    _DEFAULT_MIN_CHUNK_SIZE = 4096

    def _euclidean_distance(
        x: _ot.ArrayLike,
        y: _ot.ArrayLike
    ) -> _ot.ArrayLike:
        """
        Compute Euclidean distance matrix between points.
        
        Parameters
        ----------
        x : array, shape (n, d)
            First set of points.
        y : array, shape (m, d)
            Second set of points.
        
        Returns
        -------
        dist : array, shape (n, m)
            Pairwise Euclidean distances.
        """
        # ||x - y||² = ||x||² + ||y||² - 2 x·y  (more numerically stable)
        x_sqnorm = (x ** 2).sum(axis=1, keepdims=True)  # (n, 1)
        y_sqnorm = (y ** 2).sum(axis=1, keepdims=True)  # (m, 1)
        xy = x @ y.T  # (n, m)
        
        dist_sq = x_sqnorm + y_sqnorm.T - 2 * xy
        dist_sq = xp.clip(dist_sq, 0, None)  # Handle numerical errors
        
        return xp.sqrt(dist_sq)

    def _thin_plate_spline_kernel(r: _ot.ArrayLike
                                ) -> _ot.ArrayLike:
        """
        Thin Plate Spline (TPS) kernel: φ(r) = r² log(r).
        
        Parameters
        ----------
        r : array
            Distances.
        
        Returns
        -------
        kernel : array
            TPS kernel values. For r=0, returns 0 (limit of r²log(r)).
        """
        # Use numpy.errstate for both numpy and cupy arrays
        with np.errstate(divide='ignore', invalid='ignore'):
            result = r ** 2 * xp.log(r)
        
        result = xp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result


    def _polynomial_matrix(
        x: _ot.ArrayLike,
        degree: int = 1
    ) -> _ot.ArrayLike:
        """
        Build polynomial matrix for TPS.
        
        For degree=1 (default for TPS), includes [1, x, y] terms.
        
        Parameters
        ----------
        x : array, shape (n, d)
            Data point coordinates.
        degree : int, optional
            Polynomial degree (default: 1).
        
        Returns
        -------
        P : array, shape (n, n_poly)
            Polynomial evaluation matrix. For 2D points with degree=1:
            P = [1, x₁, x₂, ..., xₙ, y₁, y₂, ..., yₙ]^T (roughly)
        """
        n, d = x.shape
        
        if degree == -1:
            # No polynomial terms
            return xp.empty((n, 0), dtype=x.dtype)
        elif degree == 0:
            # Only constant term
            return xp.ones((n, 1), dtype=x.dtype)
        elif degree == 1:
            # Constant + linear terms: [1, x₁, x₂, ..., xₐ]
            return xp.column_stack([xp.ones(n, dtype=x.dtype), x])
        else:
            raise NotImplementedError(f"Polynomial degree {degree} not implemented")


    class RBFInterpolator:
        """
        GPU-accelerated Radial Basis Function Interpolator using xupy.
        
        Supports TPS (Thin Plate Spline) and other RBF kernels on GPU.
        For problems with >1000 constraint points, typically 5-50x faster than
        scipy's CPU implementation, depending on GPU capability.
        
        Parameters
        ----------
        y : array, shape (n, d)
            Constraint point coordinates. Can be numpy array (moved to GPU)
            or xupy array.
        d : array, shape (n, ...)
            Data values at constraint points.
        kernel : str, optional
            RBF kernel type. Currently only 'thin_plate_spline' is optimized.
            Default: 'thin_plate_spline'.
        smoothing : float or array, shape (n,), optional
            Smoothing parameter (Tikhonov regularization). Larger values reduce
            overfitting at cost of worse fit. Default: 0.0 (exact fit).
        degree : int, optional
            Polynomial degree. Default: 1 for TPS (recommended).
        memory_efficient : bool, optional
            If True, chunk computations during evaluation to reduce GPU memory.
            Default: True.
        
        Examples
        --------
        >>> import cupy as cp
        >>> # Create random constraint points
        >>> y = cp.random.rand(1000, 2)  # 1000 points in 2D
        >>> d = cp.sin(y[:, 0]) * cp.cos(y[:, 1])
        >>>
        >>> # Create interpolator
        >>> rbf = RBFInterpolatorGPU(y, d)
        >>>
        >>> # Evaluate on new points
        >>> x = cp.random.rand(10000, 2)
        >>> values = rbf(x)
        """
        
        def __init__(
            self,
            y: _ot.ArrayLike,
            d: _ot.ArrayLike,
            kernel: str = 'thin_plate_spline',
            smoothing: _ot.Union[float, _ot.ArrayLike] = 0.0,
            degree: int = 1,
            memory_efficient: bool = True,
            max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES,
            min_chunk_size: int = _DEFAULT_MIN_CHUNK_SIZE,
        ):
            """Initialize GPU RBF interpolator."""
            
            # Convert to CuPy if needed
            self.y = xp.asarray(y, dtype=xp.double)
            self._d_was_vector = np.ndim(d) == 1
            self.d = xp.asarray(d, dtype=xp.double)
            
            if self.d.ndim == 1:
                self.d = self.d[:, None]
            
            self.kernel = kernel.lower()
            self.degree = degree
            self.memory_efficient = memory_efficient
            self.max_chunk_bytes = int(max_chunk_bytes)
            self.min_chunk_size = int(min_chunk_size)
            self.output_dim = self.d.shape[1]
            
            if self.kernel != 'thin_plate_spline':
                raise NotImplementedError(
                    f"Kernel '{self.kernel}' not yet implemented. "
                    "Only 'thin_plate_spline' is optimized for GPU."
                )

            if self.max_chunk_bytes <= 0:
                raise ValueError("`max_chunk_bytes` must be positive.")

            if self.min_chunk_size <= 0:
                raise ValueError("`min_chunk_size` must be positive.")
            
            n = self.y.shape[0]
            if isinstance(smoothing, (int, float)):
                self.smoothing = xp.full(n, smoothing, dtype=xp.double)
            else:
                self.smoothing = xp.asarray(smoothing, dtype=xp.double)
            
            # Build and solve the system
            self._build_system()
        
        def _build_system(self) -> None:
            """
            Build and solve the RBF linear system.
            
            Constructs the block matrix:
                [K + λI   P] [a] = [d]
                [P^T      0] [b]   [0]
            
            where K is RBF Gram matrix, P is polynomial matrix, λ is smoothing.
            """
            n = self.y.shape[0]

            # Polynomial matrix
            P = _polynomial_matrix(self.y, self.degree)
            n_poly = P.shape[1]
            self.n_poly = n_poly
            
            # Build block matrix
            n_sys = n + n_poly
            A = xp.zeros((n_sys, n_sys), dtype=xp.double)

            # Compute RBF kernel matrix directly into the system matrix to avoid
            # keeping both K and A resident at the same time.
            dist = _euclidean_distance(self.y, self.y)
            A[:n, :n] = _thin_plate_spline_kernel(dist)
            del dist

            # Add smoothing (Tikhonov regularization)
            A[:n, :n] += xp.diag(self.smoothing)

            b = xp.zeros((n_sys, self.output_dim), dtype=xp.double)
            
            # Fill blocks
            if n_poly > 0:
                A[:n, n:] = P
                A[n:, :n] = P.T
            
            b[:n] = self.d
            
            # Solve: [a, b]^T = A⁻¹ [d, 0]^T
            try:
                coeffs = xp.linalg.solve(A, b)
            except xp.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"Failed to solve RBF system (singular matrix): {e}. "
                    "Try increasing smoothing parameter."
                )
            
            self.coeffs_rbf = coeffs[:n]
            self.coeffs_poly = coeffs[n:] if n_poly > 0 else None

        def _resolve_chunk_size(
            self,
            n_eval: int,
            chunk_size: _ot.Optional[int],
        ) -> int:
            """
            Resolve the evaluation chunk size.

            Parameters
            ----------
            n_eval : int
                Number of evaluation points.
            chunk_size : int or None
                Explicit chunk size. If None, estimate a size from the configured
                memory budget.

            Returns
            -------
            int
                Effective chunk size for the evaluation.
            """
            if chunk_size is not None:
                return max(1, min(n_eval, int(chunk_size)))

            if not self.memory_efficient:
                return n_eval

            bytes_per_float = np.dtype(np.float64).itemsize
            bytes_per_eval = bytes_per_float * (
                4 * self.y.shape[0] + self.output_dim + self.n_poly + self.y.shape[1]
            )
            estimated_chunk = self.max_chunk_bytes // max(bytes_per_eval, 1)
            estimated_chunk = max(self.min_chunk_size, int(estimated_chunk))
            return max(1, min(n_eval, estimated_chunk))
        
        def __call__(
            self,
            x: _ot.ArrayLike,
            chunk_size: _ot.Optional[int] = None
        ) -> _ot.ArrayLike:
            """
            Evaluate interpolant at points x.
            
            Parameters
            ----------
            x : array, shape (m, d)
                Evaluation points.
            chunk_size : int, optional
                Chunk size for memory-efficient evaluation. If None, computes
                in one batch. Useful for large evaluation sets on limited GPU.
            
            Returns
            -------
            f : array, shape (m, ...)
                Interpolant values at x. Same shape as d (except first axis).
            """
            x = xp.asarray(x, dtype=xp.double)
            m = x.shape[0]

            resolved_chunk_size = self._resolve_chunk_size(m, chunk_size)
            if resolved_chunk_size >= m:
                out = self._evaluate_batch(x)
            else:
                out = xp.empty((m, self.output_dim), dtype=xp.double)
                for i in range(0, m, resolved_chunk_size):
                    j = min(i + resolved_chunk_size, m)
                    out[i:j] = self._evaluate_batch(x[i:j])

            if self._d_was_vector:
                return out[:, 0]

            return out
        
        def _evaluate_batch(self, x: _ot.ArrayLike) -> _ot.ArrayLike:
            """Evaluate interpolant on a batch of points."""
            # RBF contribution: Σ aᵢ φ(||x - yᵢ||)
            dist = _euclidean_distance(x, self.y)
            K = _thin_plate_spline_kernel(dist)
            del dist
            rbf_vals = K @ self.coeffs_rbf  # (m, d)
            del K
            
            # Polynomial contribution: Σ bⱼ pⱼ(x)
            if self.coeffs_poly is not None and self.coeffs_poly.shape[0] > 0:
                P = _polynomial_matrix(x, self.degree)
                rbf_vals += P @ self.coeffs_poly  # (m, d)
            
            return rbf_vals
        
        def to_cpu(self) -> dict:
            """
            Convert coefficients to CPU (numpy) for storage or CPU evaluation.
            
            Returns
            -------
            state : dict
                Dictionary with 'y', 'coeffs_rbf', 'coeffs_poly' as numpy arrays.
            """
            return {
                'y': xp.asnumpy(self.y),
                'coeffs_rbf': xp.asnumpy(self.coeffs_rbf),
                'coeffs_poly': (xp.asnumpy(self.coeffs_poly)
                            if self.coeffs_poly is not None else None),
                'degree': self.degree,
            }

else:
    # Fallback to CPU implementation if not on GPU
    from scipy.interpolate import RBFInterpolator
