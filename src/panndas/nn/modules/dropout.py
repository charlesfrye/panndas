import random

import panndas.nn.functional as F

from .module import Module


class AlphaDropout(Module):
    """A Module that randomly replaces a fraction of its inputs with a fixed value.

    (Pseudo-)random values are drawn from the random standard library module.

    Args:
        p: The probability that any value is masked on a single call.
           Sensible values are between 0.0 and 1.0, but this is not checked.
        alpha: The value that replaces the masked value. Typically set to the
               neutral value for a following or preceding non-linearity, e.g. 0.0
               for ReLU or sigmoid.
    """

    def __init__(self, p, alpha=0.0):
        self.p = p
        self.alpha = alpha

    def forward(self, xs):
        xs = F._pointwise_map(
            xs, lambda x: x if random.random() > self.p else self.alpha
        )
        return xs

    def show(self):
        return f"{super().show()}(p={self.p}, alpha={self.alpha})"


class Dropout(AlphaDropout):
    """An AlphaDropout Module with alpha set to 0.0.

    See AlphaDropout for details."""

    def __init__(self, p):
        super().__init__(alpha=0.0, p=p)

    def show(self):
        return f"{super(AlphaDropout, self).show()}(p={self.p})"
