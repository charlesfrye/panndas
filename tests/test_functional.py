"""Test functional interface by comparing with PyTorch."""
import numpy as np
import pandas as pd
import panndas.nn.functional as F
import torch
import torch.nn.functional as tF


class TestFunctional:
    def assert_allclose(self, df, tensor):
        np.testing.assert_allclose(df.to_numpy(), tensor.numpy())


class TestPointwise(TestFunctional):
    df = pd.DataFrame(data=np.arange(-30.0, 30.0, step=1).reshape(12, 5))
    t = torch.Tensor(df.to_numpy())

    def test_relu(self):
        self.assert_allclose(F.relu(self.df), tF.relu(self.t))
