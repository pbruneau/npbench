import numpy as np
import cupy as cp


def kernel(alpha, beta, C, A, B):
    B = cp.asarray(B)
    A = cp.asarray(A)
    C = cp.asarray(C)
    beta = cp.asarray(beta)
    alpha = cp.asarray(alpha)
    C[:] = alpha * A @ B + beta * C
