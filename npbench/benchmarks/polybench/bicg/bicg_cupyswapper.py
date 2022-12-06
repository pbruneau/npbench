import numpy as np
import cupy as cp


def kernel(A, p, r):
    if isinstance(r, np.ndarray):
        r = cp.asarray(r)
    if isinstance(p, np.ndarray):
        p = cp.asarray(p)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    return r @ A, A @ p
