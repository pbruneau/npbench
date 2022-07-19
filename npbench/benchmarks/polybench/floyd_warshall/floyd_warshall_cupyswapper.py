import numpy as np
import cupy as cp


def kernel(path):
    path = cp.asarray(path)
    for k in range(path.shape[0]):
        path[:] = cp.minimum(path[:], cp.add.outer(path[:, (k)], path[(k), :]))
