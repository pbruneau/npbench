import numpy as np
import cupy as cp


def relu(x):
    return cp.maximum(x, 0)


def softmax(x):
    tmp_max = cp.max(x, axis=-1, keepdims=True)
    tmp_out = cp.exp(x - tmp_max)
    tmp_sum = cp.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


def mlp(input, w1, b1, w2, b2, w3, b3):
    b3 = cp.asarray(b3)
    w3 = cp.asarray(w3)
    b2 = cp.asarray(b2)
    w2 = cp.asarray(w2)
    b1 = cp.asarray(b1)
    w1 = cp.asarray(w1)
    input = cp.asarray(input)
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)
    return cp.asnumpy(x)
