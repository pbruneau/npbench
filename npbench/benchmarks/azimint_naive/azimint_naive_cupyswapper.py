import numpy as np
import cupy as cp


def azimint_naive(data, radius, npt):
    if isinstance(npt, np.ndarray):
        npt = cp.asarray(npt)
    if isinstance(radius, np.ndarray):
        radius = cp.asarray(radius)
    if isinstance(data, np.ndarray):
        data = cp.asarray(data)
    rmax = radius.max()
    res = cp.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = cp.logical_and(cp.asarray(r1) <= cp.asarray(radius), cp.
            asarray(radius) < cp.asarray(r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    return res
