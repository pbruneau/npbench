import numpy as np
import cupy as cp


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    if isinstance(z, np.ndarray):
        z = cp.asarray(z)
    if isinstance(y, np.ndarray):
        y = cp.asarray(y)
    if isinstance(x, np.ndarray):
        x = cp.asarray(x)
    if isinstance(w, np.ndarray):
        w = cp.asarray(w)
    if isinstance(v2, np.ndarray):
        v2 = cp.asarray(v2)
    if isinstance(u2, np.ndarray):
        u2 = cp.asarray(u2)
    if isinstance(v1, np.ndarray):
        v1 = cp.asarray(v1)
    if isinstance(u1, np.ndarray):
        u1 = cp.asarray(u1)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    if isinstance(beta, np.ndarray):
        beta = cp.asarray(beta)
    if isinstance(alpha, np.ndarray):
        alpha = cp.asarray(alpha)
    A += cp.outer(cp.asarray(u1), cp.asarray(v1)) + cp.outer(cp.asarray(u2),
        cp.asarray(v2))
    x += beta * y @ A + z
    w += alpha * A @ x
