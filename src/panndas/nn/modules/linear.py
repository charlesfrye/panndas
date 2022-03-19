import pandas as pd

from .module import Module


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


class Identity(Module):
    """A Module that returns its inputs unaltered."""

    def forward(self, xs):
        return xs
