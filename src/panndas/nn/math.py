import numpy as np
import pandas as pd


def softmax(*args, **kwargs):
    raise NotImplementedError


def entropy(*args, **kwargs):
    raise NotImplementedError


def squared_difference(dfX, dfY):
    return (dfX - dfY).applymap(lambda r: r**2)


def eye(index):
    return pd.DataFrame(np.eye(len(index)), index=index, columns=index)
