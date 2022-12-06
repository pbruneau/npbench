import numpy as np
import cupy as cp


def compute(array_1, array_2, a, b, c):
    if isinstance(c, np.ndarray):
        c = cp.asarray(c)
    if isinstance(b, np.ndarray):
        b = cp.asarray(b)
    if isinstance(a, np.ndarray):
        a = cp.asarray(a)
    if isinstance(array_2, np.ndarray):
        array_2 = cp.asarray(array_2)
    if isinstance(array_1, np.ndarray):
        array_1 = cp.asarray(array_1)
    return cp.clip(cp.asarray(array_1), 2, 10) * a + array_2 * b + c
