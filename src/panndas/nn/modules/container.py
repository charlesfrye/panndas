from .module import Module


class Sequential(Module):
    """A Module that applies an iterable of Modules sequentially."""

    def __init__(self, modules):
        self.modules = modules

    def forward(self, xs):
        for module in self:
            xs = module(xs)
        return xs

    def show(self):
        return f"{super().show()}{[module.show() for module in self]}"

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, key):
        return self.modules[key]
