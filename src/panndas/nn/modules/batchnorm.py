from .module import Module


class LayerMaxNorm(Module):
    """Normalize across the feature dimension with respect to the infinity norm."""

    def forward(self, xs):
        return xs.divide(xs.abs().max(axis=0), axis=1).fillna(0.0)
