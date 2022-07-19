import numpy as np
import cupy as cp


def kernel(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x
