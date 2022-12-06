import numpy as np
import cupy as cp


def kernel(A):
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    A[:] = cp.linalg.cholesky(cp.asarray(A)) + cp.triu(A, k=1)
