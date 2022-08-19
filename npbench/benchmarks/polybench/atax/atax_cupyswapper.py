import numpy as np
import cupy as cp


def kernel(A, x):
    x = cp.asarray(x)
    A = cp.asarray(A)
    return cp.asnumpy(A @ x @ A)
