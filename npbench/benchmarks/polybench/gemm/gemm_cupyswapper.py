import numpy as np
import cupy as cp


def kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C
