import numpy as np
import cupy as cp


def kernel(alpha, beta, C, A, B):
    if isinstance(B, np.ndarray):
        B = cp.asarray(B)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    if isinstance(C, np.ndarray):
        C = cp.asarray(C)
    if isinstance(beta, np.ndarray):
        beta = cp.asarray(beta)
    if isinstance(alpha, np.ndarray):
        alpha = cp.asarray(alpha)
    C[:] = alpha * A @ B + beta * C
