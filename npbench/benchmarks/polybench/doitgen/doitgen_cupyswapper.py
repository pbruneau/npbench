import numpy as np
import cupy as cp


def kernel(NR, NQ, NP, A, C4):
    if isinstance(C4, np.ndarray):
        C4 = cp.asarray(C4)
    if isinstance(A, np.ndarray):
        A = cp.asarray(A)
    if isinstance(NP, np.ndarray):
        NP = cp.asarray(NP)
    if isinstance(NQ, np.ndarray):
        NQ = cp.asarray(NQ)
    if isinstance(NR, np.ndarray):
        NR = cp.asarray(NR)
    A[:] = cp.reshape(cp.reshape(cp.asarray(A), (int(NR), int(NQ), 1, int(
        NP))) @ cp.asarray(C4), (int(NR), int(NQ), int(NP)))
