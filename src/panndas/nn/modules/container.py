from .module import Module


class Sequential(Module):
    def __init__(self, modules):
        """A Module that applies an iterable of Modules sequentially.

        Args:
            modules: An interable of panndas.nn Modules.

        Examples:
            >>> import panndas.nn as nn
            >>> m1, m2 = [nn.Module(), nn.Module()]
            >>> sequential = nn.Sequential([m1, m2])
        """
        self.modules = modules

    def forward(self, xs):
        for module in self:
            xs = module(xs)
        return xs

    def show(self):
        return f"{super().show()}[{','.join([module.show() for module in self])}]"

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, key):
        return self.modules[key]
