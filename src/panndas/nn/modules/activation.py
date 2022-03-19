from .module import Module
from ..functional import softmax


class ReLU(Module):
    """Ol' ReLU-iable."""

    def forward(self, xs):
        return xs.applymap(lambda x: max(x, 0.0))


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

        A = softmax(Q.T @ K)

        zs = V @ A
        return zs
