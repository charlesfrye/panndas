from .module import Module


class AdditiveSkip(Module):
    """A Module that applies an additive "skip" connection around the provided Module."""

    def __init__(self, block):
        self.block = block

    def forward(self, xs):
        return xs + self.block(xs)

    def show(self):
        return f"[{super().show()}[{self.block.show()}]"


class TransformerEncoderLayer(Module):
    pass
