import numpy as np
import cupy as cp


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    z = cp.asarray(z)
    y = cp.asarray(y)
    x = cp.asarray(x)
    A = cp.asarray(A)
    beta = cp.asarray(beta)
    alpha = cp.asarray(alpha)
    A += cp.outer(u1, v1) + cp.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
