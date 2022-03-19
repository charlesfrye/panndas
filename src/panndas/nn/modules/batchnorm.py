from .module import Module


class LayerMaxNorm(Module):
    """Normalize across the feature dimension with respect to the infinity norm."""

    def forward(self, xs):
        return xs.divide(xs.abs().max(axis=0), axis=1).fillna(0.0)


class BatchNorm1d(Module):
    """Normalize across the sequence dimension with respect to the 2 norm."""

    def __init__(self, eps=1e-05, gamma=1.0, beta=0.0):
        self.eps = eps
        self.gamma = gamma
        self.beta = beta

    def forward(self, xs):
        mu = xs.mean()
        sigma = (xs.var() + self.eps).sqrt()
        return (xs - mu) / (sigma) * self.gamma + self.beta
