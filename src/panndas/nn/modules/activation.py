import panndas.nn.functional as F

from .module import Module


class ReLU(Module):
    """Ol' ReLU-iable.

    Applies the recitified linear function, elementwise."""

    def forward(self, xs):
        return F.relu(xs)


class Mish(Module):
    """Applies the Mish function, element-wise.

    For details, see Mish: A Self-Regularized Non-Monotonic
    Neural Activation Function."""

    def forward(self, xs):
        return F.mish(xs)


class Sigmoid(Module):
    """Applies the sigmoid function, element-wise."""

    def forward(self, xs):
        return F.sigmoid(xs)


class Tanh(Module):
    """Applies the hyperbolic tangent function, element-wise."""

    def forward(self, xs):
        return F.tanh(xs)


class Softplus(Module):
    """Applies the softplus function, element-wise."""

    def forward(self, xs):
        return F.softplus(xs)


class Softmax(Module):
    """Applies softmax function, column-wise."""

    def forward(self, xs):
        return F.softmax(xs)


class Attention(Module):
    """Literally all you need."""

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
    """The most basic version of an attention layer.

    Combines queries, keys, and values linearly."""

    def forward(self, xs):
        Q, K, V = super().forward(xs)

        A = Q.T @ K
        zs = V @ A

        return zs


class SoftmaxAttention(Attention):
    """The best-known version of an attention layer.

    Uses a softmax over the sequence dim to select which values to attend to."""

    def forward(self, xs):
        Q, K, V = super().forward(xs)

        A = F.softmax((Q.T @ K).T).T
        zs = V @ A

        return zs
