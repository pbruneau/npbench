import numpy as np
import cupy as cp


def kernel(A, p, r):
    r = cp.asarray(r)
    p = cp.asarray(p)
    A = cp.asarray(A)
    return cp.asnumpy(r @ A), cp.asnumpy(A @ p)
