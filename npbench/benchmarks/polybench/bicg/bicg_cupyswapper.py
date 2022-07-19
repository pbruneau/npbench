import numpy as np
import cupy as cp


def kernel(A, p, r):
    return r @ A, A @ p
