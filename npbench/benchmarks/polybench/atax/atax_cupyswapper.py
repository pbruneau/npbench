import numpy as np
import cupy as cp


def kernel(A, x):
    if isinstance(x, np.ndarray):
        x = cp.asarray(x)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    return A @ x @ A
