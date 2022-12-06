import numpy as np
import cupy as cp


def kernel(alpha, beta, A, B, x):
    if isinstance(x, np.ndarray):
        x = cp.asarray(x)
    if isinstance(B, np.ndarray):
        B = cp.asarray(B)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    if isinstance(beta, np.ndarray):
        beta = cp.asarray(beta)
    if isinstance(alpha, np.ndarray):
        alpha = cp.asarray(alpha)
    return alpha * A @ x + beta * B @ x
