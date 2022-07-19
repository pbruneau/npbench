import numpy as np
import cupy as cp


def compute(array_1, array_2, a, b, c):
    c = cp.asarray(c)
    b = cp.asarray(b)
    a = cp.asarray(a)
    array_2 = cp.asarray(array_2)
    array_1 = cp.asarray(array_1)
    return cp.asnumpy(cp.clip(array_1, 2, 10) * a + array_2 * b + c)
