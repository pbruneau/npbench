import cupy as np


def kernel(A, x):

    return np.asnumpy((A @ x) @ A)
