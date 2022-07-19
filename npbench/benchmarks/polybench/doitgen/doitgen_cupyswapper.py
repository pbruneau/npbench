import numpy as np
import cupy as cp


def kernel(NR, NQ, NP, A, C4):
    C4 = cp.asarray(C4)
    A = cp.asarray(A)
    NP = cp.asarray(NP)
    NQ = cp.asarray(NQ)
    NR = cp.asarray(NR)
    A[:] = cp.reshape(cp.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
