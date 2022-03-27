import pandas as pd

from .module import Module


class Linear(Module):
    """A Module that multiplies its inputs by the weights_df and adds the bias_series.

    Input 'tensors'  can be at most 2-D here: feature (rows) and batch/sequence (columns).

    Args:
        weights_df: Weights for the affine transform. Column index
                    is the input feature space and row index is the
                    output feature space.
        bias_series: Biases for the affine transform. If not a pd.Series,
                     presumed to be a single element that is promoted to a Series.

    Examples:
    >>> import pandas as pd
    >>> import panndas.nn as nn
    >>> w = pd.DataFrame([[0.0, 1.0],[1.0, 0.0]])              # reflection matrix
    >>> w.columns = pd.Index(["left", "right"], name="inputs")
    >>> w.index = pd.Index(["right", "left"], name="outputs")  # reflection mirrors inputs
    >>> l = nn.Linear(weights_df=w, bias_series=0.0)
    >>> s = pd.Series([1.0, 2.0], index=w.columns)
    >>> s
    inputs
    left    1.0
    right   2.0
    dtype: float64
    >>> l(s)
    outputs
    right    2.0
    left     1.0
    dtype: float64
    """

    def __init__(self, weights_df, bias_series=-1.0):
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
