import cupy as np


def kernel(A, p, r):

    return np.asnumpy(r @ A), np.asnumpy(A @ p)
