from abc import abstractmethod


class Module(object):
    def __init__(self):
        """An object that is callable via its .forward method."""

    def __call__(self, xs):
        return self.forward(xs)

    @abstractmethod
    def forward(self, xs):
        """Applies the Module to its input."""
        raise NotImplementedError

    def show(self):
        """Displays the Module in a human-friendly format."""
        return str(self.__class__.__name__)
