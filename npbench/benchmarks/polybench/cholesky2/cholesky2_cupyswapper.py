import numpy as np
import cupy as cp


def kernel(A):
    A = cp.asarray(A)
    A[:] = cp.linalg.cholesky(A) + cp.asarray(np.triu(cp.asnumpy(A), k=1))
