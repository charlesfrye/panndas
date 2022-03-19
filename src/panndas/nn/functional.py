def softmax(*args, **kwargs):
    raise NotImplementedError


def entropy(*args, **kwargs):
    raise NotImplementedError


def squared_difference(dfX, dfY):
    """Computes the squared differences between dfX and dfY."""
    return (dfX - dfY).applymap(lambda r: r**2)


def absolute_difference(dfX, dfY):
    """Computes the absolute differences between dfX and dfY."""
    return (dfX - dfY).applymap(lambda r: abs(r))
