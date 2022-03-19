import random

import panndas.nn.functional as F

from .module import Module


class AlphaDropout(Module):
    """A Module that multiplies its inputs by the weights_df and adds the bias_series.

    Input 'tensors'  can be at most 2-D here: feature (rows) and batch/sequence (columns).

    The weights dataframe should have the input feature space as its column index
    and the output feature space as its row index."""

    def __init__(self, p, alpha=0.0):
        self.p = p
        self.alpha = alpha

    def forward(self, xs):
        xs = F._pointwise_apply(lambda x: x if random.random() > self.p else self.alpha)
        return xs

    def show(self):
        return f"{super().show()}:: {self.w.columns.name} -> {self.w.index.name}"


class Dropout(AlphaDropout):
    def __init__(self, p):
        super().__init__(alpha=0.0, p=p)
