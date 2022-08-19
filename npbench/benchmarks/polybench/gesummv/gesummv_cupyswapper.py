import numpy as np
import cupy as cp


def kernel(alpha, beta, A, B, x):
    x = cp.asarray(x)
    B = cp.asarray(B)
    A = cp.asarray(A)
    beta = cp.asarray(beta)
    alpha = cp.asarray(alpha)
    return cp.asnumpy(alpha * A @ x + beta * B @ x)
