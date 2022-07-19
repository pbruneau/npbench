import numpy as np
import cupy as cp


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    v2 = cp.asarray(v2)
    u2 = cp.asarray(u2)
    v1 = cp.asarray(v1)
    u1 = cp.asarray(u1)
    A = cp.asarray(A)
    A += cp.outer(u1, v1) + cp.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
