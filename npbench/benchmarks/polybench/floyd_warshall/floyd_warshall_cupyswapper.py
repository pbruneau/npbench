import numpy as np
import cupy as cp


def kernel(path):
    if isinstance(path, np.ndarray):
        path = cp.asarray(path)
    for k in range(path.shape[0]):
        path[:] = cp.minimum(path[:], cp.asarray(np.add.outer(cp.asnumpy(
            path[:, (k)]), cp.asnumpy(path[(k), :]))))
