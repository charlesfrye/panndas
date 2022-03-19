import panndas.nn.functional as F

from .module import Module


class ReLU(Module):
    """Ol' ReLU-iable."""

    def forward(self, xs):
        return F.relu(xs)


class Mish(Module):
    """Applies the Mish function, element-wise.

    For details, see Mish: A Self-Regularized Non-Monotonic
    Neural Activation Function."""

    def forward(self, xs):
        """Applies the Mish function, element-wise."""
        return F.mish(xs)


class Sigmoid(Module):
    """Applies the sigmoid function, element-wise."""

    def forward(self, xs):
        """Applies the sigmoid function, element-wise."""
        return F.sigmoid(xs)


class Tanh(Module):
    """Applies the hyperbolic tangent function, element-wise."""

    def forward(self, xs):
        """Applies the hyperbolic tangent function, element-wise."""
        return F.tanh(xs)


class Softplus(Module):
    """Applies the softplus function, element-wise."""

    def forward(self, xs):
        """Applies the softplus function, element-wise."""
        return F.softplus(xs)


class Softmax(Module):
    """Applies softmax function, column-wise."""

    def forward(self, xs):
        """Applies softmax function, column-wise."""
        return F.softmax(xs)


class Attention(Module):
    """Literally all you need."""

    def __init__(self, queries_df, keys_df, values_df):
        self.w_q = queries_df
        self.w_k = keys_df
        self.w_v = values_df

    def forward(self, xs):
        """Computes queries, keys, and values given inputs."""
        Q = self.w_q @ xs
        K = self.w_k @ xs
        V = self.w_v @ xs

        return Q, K, V


class LinearAttention(Attention):
    """The most basic version of an attention layer."""

    def forward(self, xs):
        """Combines queries, keys, and values linearly."""
        V, Q, K = super().forward(xs)

        A = Q.T @ K
        zs = V @ A

        return zs


class SoftmaxAttention(Attention):
    """The best-known version of an attention layer."""

    def forward(self, xs):
        """Uses a softmax over the sequence dim to select which values to attend to."""
        V, Q, K = super().forward(xs)

        A = F.softmax((Q.T @ K).T).T
        zs = V @ A

        return zs
