import numpy as np
import cupy as cp


def kernel(alpha, A, B):
    if isinstance(B, np.ndarray):
        B = cp.asarray(B)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    if isinstance(alpha, np.ndarray):
        alpha = cp.asarray(alpha)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += cp.dot(A[i + 1:, (i)], B[i + 1:, (j)])
    B *= alpha
