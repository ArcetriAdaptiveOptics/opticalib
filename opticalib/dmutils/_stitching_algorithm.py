import numpy as np
import xupy as xp
from tqdm import tqdm, trange
from opticalib.ground.modal_decomposer import ZernikeFitter
from opticalib import typings as t

f32 = xp.float


def map_stitching(
    image_vector: t.CubeData,
    fullmask: t.ImageData,
    zern2fit: list[int],
    mp_chunk_size: int = 128,
) -> t.ImageData:
    """
    Stitching algorithm.

    Parameters
    ----------
    image_vector : np.ndarray
        3D array of images to be stitched, shape (N, H, W).
    fullmask : np.ndarray
        2D array representing the full mask, shape (H, W).
    zern2fit : list
        List of Zernike indices to fit.
    mp_chunk_size : int, optional
        Chunk size for multiprocessing (on CPU only), by default 128.

    Returns
    -------
    np.ndarray
        2D array of the stitched image after removing specified Zernike terms.
    """
    print("Computing Zernike basis...", end="\r", flush=True)
    N = image_vector.shape[0]
    MM = fullmask.copy().astype(xp.uint8)
    _zfit = ZernikeFitter(MM)
    M = len(zern2fit)
    p = xp.asarray(
        xp.asnumpy([_zfit.makeSurface(i) for i in [[1], [1, 2], [1, 2, 3]]]), dtype=f32
    )
    Qo = xp.tile(p, (M, 1, 1))
    v_order = xp.reshape(
        xp.reshape(xp.arange(M**2, dtype=xp.int8), (M, M)).T, (1, M**2)
    )
    q = Qo * Qo[v_order[0]]

    print("Setting up stitching algorithm...", end="\r", flush=True)
    # Pre-extract masks and data for efficiency
    masks = xp.array([img.mask for img in image_vector], dtype=xp.uint8)
    data = xp.array([img.data for img in image_vector], dtype=f32)

    # Prepare all (ii, jj) pairs
    pairs = [(ii, jj) for ii in range(N) for jj in range(N)]

    Q = xp.zeros((N, N, M**2), dtype=f32)
    P = xp.zeros((N, N, M), dtype=f32)

    pbar = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"

    if xp.on_gpu:
        # GPU computation, if available
        # Move arrays to GPU in single precision float
        for ii in trange(
            N,
            desc="P-Q Computation",
            ncols=65,
            colour="green",
            unit="img",
            bar_format=pbar,
        ):
            for jj in range(N):
                mm = xp.logical_or(masks[ii], masks[jj])
                if ii == jj:
                    Q_val = xp.zeros(M**2, dtype=f32)
                else:
                    Q_val = xp.sum(q * (~mm), axis=(1, 2))
                img = data[ii] - data[jj]
                P_val = xp.nansum(p * (~mm) * img, axis=(1, 2))
                Q[ii, jj, :] = Q_val
                P[ii, jj, :] = P_val

    else:
        from multiprocessing import Pool, cpu_count

        # Back to CPU computation
        block_compute = _BlockCompute(masks, data, M, p, q)
        with Pool(processes=cpu_count()) as pool:
            for (ii, jj), Q_val, P_val in tqdm(
                pool.imap_unordered(
                    block_compute.compute_block, pairs, chunksize=mp_chunk_size
                ),
                desc="P-Q Computation",
                total=len(pairs),
                ncols=65,
                unit="pair",
                colour="green",
                bar_format=pbar,
            ):
                Q[ii, jj, :] = Q_val
                P[ii, jj, :] = P_val

    print("Computing stitched image...", end="\r", flush=True)
    P1 = xp.reshape(P, (N, N, M))
    Pt = xp.array([xp.sum(P1[ii], axis=0) for ii in range(N)], dtype=f32)
    PP = xp.reshape(Pt, M * N)
    Q1 = xp.reshape(Q, (N, N, M**2))
    QQ = xp.reshape(Q1, (N, N, M, M))
    temp = xp.vstack(
        [xp.hstack([QQ[ii, jj, :, :] for ii in range(N)]) for jj in range(N)]
    )
    QQ = temp.copy()
    QD = xp.zeros_like(QQ, dtype=f32)
    for ii in range(N):
        temp = xp.sum(Q1[ii], axis=0)
        temp = xp.reshape(temp, (M, M))
        QD[M * ii : M * (ii + 1), M * ii : M * (ii + 1)] = -temp
    QF = QD + QQ
    X = xp.linalg.lstsq(QF, PP, rcond=None)[0]
    zzc = xp.ma.empty_like(image_vector, dtype=f32)
    c = xp.reshape(X, (N, M))
    for ii in range(N):
        img = xp.asarray(image_vector[ii, :, :].data, dtype=f32)
        mm = xp.asarray(image_vector[ii, :, :].mask, dtype=xp.int8)
        res = xp.zeros_like(MM, dtype=f32)
        for ki in range(M):
            res += p[ki] * c[ii, ki]
        zzc[ii, :, :] = xp.ma.masked_array(
            (img + res) * (-1 * mm + 1), mm.astype(bool), dtype=f32
        )
    print(f"Removing zernike modes {zern2fit}...", end="\r", flush=False)
    ZZ = xp.ma.mean(zzc, axis=0)
    ZZ = _zfit.removeZernike(ZZ, zern2fit)
    return ZZ


class _BlockCompute:

    def __init__(
        self,
        masks: t.ImageData,
        data: t.ImageData,
        M: int,
        p: t.ArrayLike,
        q: t.ArrayLike,
    ):
        self.masks = masks
        self.data = data
        self.M = M
        self.p = p
        self.q = q

    def compute_block(
        self, args: tuple[int, int]
    ) -> tuple[tuple[int, int], t.MatrixLike, t.MatrixLike]:
        ii, jj = args
        mm = np.logical_or(self.masks[ii], self.masks[jj])
        if ii == jj:
            Q_val = np.zeros(self.M**2)
        else:
            Q_val = np.sum(self.q * (~mm).astype(np.int8), axis=(1, 2))
        img = self.data[ii] - self.data[jj]
        P_val = np.array(np.nansum(self.p * (~mm).astype(np.int8) * img, axis=(1, 2)))
        return (ii, jj), Q_val, P_val
