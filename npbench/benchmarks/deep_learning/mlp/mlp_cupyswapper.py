import numpy as np
import cupy as cp


def relu(x):
    return cp.maximum(cp.asarray(x), 0)


def softmax(x):
    tmp_max = cp.max(cp.asarray(x), axis=-1, keepdims=True)
    tmp_out = cp.exp(cp.asarray(x) - tmp_max)
    tmp_sum = cp.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def mlp(input, w1, b1, w2, b2, w3, b3):
    if isinstance(b3, np.ndarray):
        b3 = cp.asarray(b3)
    if isinstance(w3, np.ndarray):
        w3 = cp.asarray(w3)
    if isinstance(b2, np.ndarray):
        b2 = cp.asarray(b2)
    if isinstance(w2, np.ndarray):
        w2 = cp.asarray(w2)
    if isinstance(b1, np.ndarray):
        b1 = cp.asarray(b1)
    if isinstance(w1, np.ndarray):
        w1 = cp.asarray(w1)
    if isinstance(input, np.ndarray):
        input = cp.asarray(input)
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)
    return x
