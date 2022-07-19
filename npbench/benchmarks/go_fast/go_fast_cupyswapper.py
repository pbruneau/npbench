import numpy as np
import cupy as cp


def go_fast(a):
    a = cp.asarray(a)
    trace = 0.0
    for i in range(a.shape[0]):
        trace += cp.tanh(a[i, i])
    return cp.asnumpy(a + trace)
