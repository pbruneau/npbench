import numpy as np
import cupy as cp


def kernel(alpha, A, B):
    B = cp.asarray(B)
    A = cp.asarray(A)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += cp.dot(A[i + 1:, (i)], B[i + 1:, (j)])
    B *= alpha
