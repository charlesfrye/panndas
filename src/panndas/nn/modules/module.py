from abc import abstractmethod


class Module(object):
    """An object that is callable via its .forward method."""

    def __call__(self, xs):
        return self.forward(xs)

    @abstractmethod
    def forward(self, xs):
        """Minimally, define this method to define a Module."""
        raise NotImplementedError

    def show(self):
        """Displays the Module."""
        return str(self.__class__.__name__)
