import cupy as np


def kernel(alpha, beta, A, B, x):

    return np.asnumpy(alpha * A @ x + beta * B @ x)
