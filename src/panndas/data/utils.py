import numpy as np
import pandas as pd


def eye(index):
    return pd.DataFrame(np.eye(len(index)), index=index, columns=index)
