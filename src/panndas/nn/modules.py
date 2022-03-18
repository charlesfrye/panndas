from abc import abstractmethod

import pandas as pd

from . import math


class Module(object):
    """An object that is callable via its .forward method."""

    def __call__(self, xs):
        return self.forward(xs)

    @abstractmethod
    def forward(self, xs):
        """Minimally, define this method to define a Module."""
        raise NotImplementedError

    def show(self):
        return str(self.__class__.__name__)


class Sequential(Module):
    """A Module that applies an iterable of Modules sequentially."""

    def __init__(self, *modules):
        self.modules = modules

    def forward(self, xs):
        for module in self:
            xs = module(xs)
        return xs

    def show(self):
        return f"{super().show()}{[self.show(module) for module in self]}"

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, key):
        return self.modules[key]


class Linear(Module):
    """A Module that multiplies its inputs by the weights_df and adds the bias_series.

    Input 'tensors'  can be at most 2-D here: feature (rows) and batch/sequence (columns).

    The weights dataframe should have the input feature space as its column index
    and the output feature space as its row index."""

    def __init__(self, weights_df, bias_series=-1):
        self.w = weights_df
        if not isinstance(bias_series, pd.Series):
            bias_series = pd.Series(
                [bias_series] * len(weights_df), index=weights_df.index
            )
        self.b = bias_series

    def forward(self, xs):
        return (self.w @ xs).add(self.b, axis=0)

    def show(self):
        return f"{super().show()}:: {self.w.columns.name} -> {self.w.index.name}"


class AdditiveSkip(Module):
    """A Module that applies an additive "skip" connection around the provided Module."""

    def __init__(self, block):
        self.block = block

    def forward(self, xs):
        return xs + self.block(xs)

    def show(self):
        return f"[{super().show()}[{self.block.show()}]"


class Attention(Module):
    """Literally all you need, so we're done here."""

    def __init__(self, queries_df, keys_df, values_df):
        self.w_q = queries_df
        self.w_k = keys_df
        self.w_v = values_df

    def forward(self, xs):
        Q = self.w_q @ xs
        K = self.w_k @ xs
        V = self.w_v @ xs

        return Q, K, V


class LinearAttention(Attention):
    """The most basic version of an attention layer."""

    def forward(self, xs):
        V, Q, K = super().forward(xs)

        A = Q.T @ K

        zs = V @ A
        return zs


class SoftmaxAttention(Attention):
    """The best-known version of an attention layer."""

    def forward(self, xs):
        V, Q, K = super().forward(xs)

        A = math.softmax(Q.T @ K)

        zs = V @ A
        return zs
