import numpy as np
import cupy as cp


def kernel(A, x):
    return A @ x @ A
