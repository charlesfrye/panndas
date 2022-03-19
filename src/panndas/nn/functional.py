"""Functional interface."""
import math

import pandas as pd


def _pointwise_map(p, f):
    if isinstance(p, pd.Series):
        return p.apply(f)
    else:
        return p.applymap(f)


def _columnwise_map(p, f):
    return p.apply(f, axis=0)


def _rowwise_map(p, f):
    return p.apply(f, axis=1)


def relu(p):
    return _pointwise_map(p, _relu)


def _relu(x):
    return max(x, 0.0)


def softplus(p, beta=1.0, threshold=20.0):
    return _pointwise_map(p, _softplus)


def _softplus(x, beta, threshold):
    if x * beta > threshold:
        return x
    else:
        return 1.0 / beta * math.log(1 + math.exp(beta * x))


def tanh(p):
    return _pointwise_map(p, _tanh)


def _tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def mish(p):
    return _pointwise_map(p, _mish)


def _mish(x):
    return softplus(tanh(x))


def sigmoid(p):
    return _pointwise_map(p, _sigmoid)


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(p):
    if isinstance(p, pd.DataFrame):
        return _columnwise_map(p, _softmax)
    else:
        return _softmax(p)


def _softmax(s):
    exps = s.apply(math.exp)
    return exps / exps.sum()


def entropy(*args, **kwargs):
    raise NotImplementedError


def squared_difference(pX, pY):
    """Computes the squared differences between pX and pY."""
    return _pointwise_map(pX - pY)(lambda r: r**2)


def absolute_difference(pX, pY):
    """Computes the absolute differences between pX and pY."""
    return _pointwise_map(pX - pY)(lambda r: abs(r))
